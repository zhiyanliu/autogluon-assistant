#!/usr/bin/env python3
"""
Regenerate condensed_tutorials/ for one or more tools using a fresh LLM.

Scope:
  This script ONLY rewrites condensed_tutorials/. It never modifies tutorials/.
  Rationale: TutorialIndexer maintains two independent FAISS indices per tool
  (one over tutorials/ summaries, one over condensed_tutorials/ summaries),
  and the default agent flow (condense_tutorials: True) only queries the
  condensed index. So refreshing condensed quality does not require touching
  the pristine tutorials/ source.

Why a separate script (instead of calling ToolsRegistry.add_tool_tutorials):
  add_tool_tutorials() also re-stamps tutorials/<file>.md with a fresh
  "Summary: ..." line. Pointing it at the existing tutorials/ would
  double-stamp (the file already begins with a "Summary: ..." line). This
  script avoids that by reading each tutorial, stripping any leading
  "Summary: ..." paragraph to recover the raw body, condensing the body, and
  writing only condensed_tutorials/<file>.md.

Configuration:
  The LLM config path is hardcoded as the module-level constant `CONFIG_PATH`
  near the top of this file (style-aligned with other tools/*.py scripts in
  this repo, which prefer module constants over CLI flags for static paths).
  To use a different LLM, edit `CONFIG_PATH` to point at another yaml under
  src/autogluon/assistant/configs/, or copy one of those yamls and edit its
  `llm.model` line.

Usage examples:
  # Regenerate all tools.
  python tools/regenerate_condensed.py

  # Regenerate only autogluon.tabular and autogluon.timeseries.
  python tools/regenerate_condensed.py \\
      --tool autogluon.tabular --tool autogluon.timeseries

  # Dry run (list files that would be regenerated, no LLM calls).
  python tools/regenerate_condensed.py --dry-run

  # Only regenerate condensed files that are missing (skip existing ones).
  # Useful for incremental top-up after adding new tutorials.
  python tools/regenerate_condensed.py --only-missing

Environment:
  Set the appropriate API credentials for your provider, e.g.
    export ANTHROPIC_API_KEY=...
    export AWS_DEFAULT_REGION=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from autogluon.assistant.llm import ChatLLMFactory
from autogluon.assistant.tools_registry import ToolsRegistry
from autogluon.assistant.tools_registry.utils import (
    extract_title_from_markdown,
    split_markdown_into_chunks,
)


# --------------------------------------------------------------------------- #
# Configuration (edit these constants instead of passing CLI flags)
# --------------------------------------------------------------------------- #
# Path to the yaml whose top-level `llm:` block configures the LLM used to
# condense tutorials. Resolved relative to the repo root so it works from
# any cwd. To switch model, point this at a different yaml under
# src/autogluon/assistant/configs/, or copy one of those and edit `llm.model`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = (
    _REPO_ROOT / "src" / "autogluon" / "assistant" / "configs" / "bedrock.yaml"
)

# DEFAULT_CHUNK_SIZE matches registry.add_tool_tutorials. A tutorial body is
# only split when len(body) > 2 * DEFAULT_CHUNK_SIZE (i.e. > 16384 chars), so
# in practice only a handful of long tutorials (e.g. tabular-indepth,
# customization, forecasting-indepth) are actually chunked.
DEFAULT_CHUNK_SIZE = 8192

# DEFAULT_MAX_LENGTH bounds the *condensed_content* (not the whole file).
# Raised from registry's default of 9999 to 12288: at 9999 the two most
# complex tutorials (customization.md, tabular-indepth.md) were being
# truncated, which loses information. Headroom check: downstream prompt
# concatenates up to max_num_tutorials=5 condensed files, capped at
# max_tutorial_length=32768 in default.yaml -- 12288 still leaves the
# system functional even if all 5 retrieved tutorials are at the cap.
DEFAULT_MAX_LENGTH = 12288

# A leading "Summary: ..." paragraph (one or more lines) terminated by a blank
# line. We use re.DOTALL so the "..." can span lines until the blank line.
LEADING_SUMMARY_RE = re.compile(r"^\s*Summary:.*?(?:\n\s*\n)", re.DOTALL)

logger = logging.getLogger("regenerate_condensed")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate condensed_tutorials/ using a fresh LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tool",
        action="append",
        default=None,
        help="Tool name to regenerate (repeatable). Default: all tools in the registry.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Markdown chunk size for condensation. Default: {DEFAULT_CHUNK_SIZE}.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max length of final condensed content. Default: {DEFAULT_MAX_LENGTH}.",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip tutorials whose condensed_tutorials/<rel>.md already exists. "
        "Useful for incrementally filling in newly added tutorials without "
        "re-condensing the ones already done.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be regenerated and exit. No LLM calls.",
    )
    p.add_argument(
        "--token-report",
        type=str,
        default=None,
        help="Optional path to write a JSON token-usage report at the end.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging (DEBUG)."
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def _load_llm_config() -> DictConfig:
    """Load the `llm:` block from CONFIG_PATH, matching what ChatLLMFactory expects."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}. "
            f"Edit CONFIG_PATH at the top of {Path(__file__).name} to point at a valid yaml."
        )
    full_cfg = OmegaConf.load(CONFIG_PATH)
    if "llm" not in full_cfg:
        raise ValueError(f"Config {CONFIG_PATH} has no top-level `llm:` section")
    llm_cfg = OmegaConf.create(OmegaConf.to_container(full_cfg.llm, resolve=True))

    # Required by the chunked condense flow (chunks share context within one
    # tutorial). Force-on regardless of what the source yaml had.
    llm_cfg.multi_turn = True

    return llm_cfg


# --------------------------------------------------------------------------- #
# Tutorial body / summary handling
# --------------------------------------------------------------------------- #
def _strip_existing_summary(content: str) -> Tuple[str, Optional[str]]:
    """
    If `content` starts with a `Summary: ...` paragraph (terminated by a blank
    line), strip it off and return (body, old_summary). Otherwise return
    (content, None).
    """
    m = LEADING_SUMMARY_RE.match(content)
    if not m:
        return content, None
    old_summary = m.group(0).strip()
    body = content[m.end() :]
    return body, old_summary


# --------------------------------------------------------------------------- #
# Condense pipeline (mirrors registry.add_tool_tutorials, minus double-stamp)
# --------------------------------------------------------------------------- #
#
# Developer reference: per-tutorial condense + summary flow.
# (Internal note only -- not exposed via __doc__ / --help.)
#
#       raw tutorial body
#             │
#             ▼
#  ┌──────────────────┐
#  │ split_markdown_  │ ──→  chunk 1, chunk 2, ..., chunk N
#  │  into_chunks     │       N = 1 when body ≤ 2 * chunk_size
#  └──────────────────┘       (i.e. no actual splitting happens)
#             │
#             ▼  one LLM session per tutorial, multi_turn=True
#  ┌──────────────────────────────────────────────────────┐
#  │ for i in 0..N-1:                                     │
#  │     user      → raw chunk i                          │
#  │     assistant → condensed chunk i                    │
#  │ # multi_turn keeps tone, terminology and section     │
#  │ # structure consistent across chunks of one tutorial.│
#  └──────────────────────────────────────────────────────┘
#             │
#             ▼
#  condensed_content = "\n\n".join(all condensed chunks)
#             │
#             ▼  same session continues (turn N+1)
#  ┌──────────────────────────────────────────────────────┐
#  │ user      → summary_prompt + condensed_content       │
#  │ assistant → "Summary: ..."                           │
#  └──────────────────────────────────────────────────────┘
#             │
#             ▼  truncate condensed_content to DEFAULT_MAX_LENGTH
#                (preserving section / paragraph boundaries)
#             │
#             ▼  written to disk; tutorials/ is NEVER touched
#  condensed_tutorials/<rel>.md
#  ┌──────────────────────────────┐
#  │ # Condensed: <title>         │
#  │                              │
#  │ Summary: ...                 │ ← FAISS indexes this line
#  │                              │
#  │ *This is a condensed ...*    │
#  │                              │
#  │ <condensed_content>          │ ← all condensed chunks joined
#  └──────────────────────────────┘
#
def _condense_one_file(
    tutorial_file: Path,
    tool_name: str,
    llm_cfg: DictConfig,
    chunk_size: int,
    max_length: int,
) -> Tuple[str, str, str]:
    """
    Returns (new_summary, condensed_content, body_without_summary).
    """
    raw = tutorial_file.read_text(encoding="utf-8")
    body, _ = _strip_existing_summary(raw)

    # One LLM session per tutorial, multi-turn so chunks share context.
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    session_name = f"regen_{tool_name}_{tutorial_file.stem}_{timestamp}"
    llm = ChatLLMFactory.get_chat_model(llm_cfg, session_name=session_name)

    if len(body) > 2 * chunk_size:
        chunks = split_markdown_into_chunks(body, max_chunk_size=chunk_size)
    else:
        chunks = [body]

    condensed_chunks: List[str] = []
    for i, chunk in enumerate(chunks):
        context = "This is a continuation of the previous chunk. " if i > 0 else ""
        chunk_prompt = (
            f"{context}Condense this portion of the tutorial while preserving essential "
            f"implementation details, code samples, and key concepts. Focus on:\n\n"
            f"1. Implementation details and techniques\n"
            f"2. Code snippets with necessary context\n"
            f"3. Critical configurations and parameters\n"
            f"4. Important warnings and best practices\n\n"
            f"Chunk {i + 1}/{len(chunks)}:\n{chunk}\n\n"
            f"Provide the condensed content in markdown format."
        )
        condensed_chunks.append(llm.assistant_chat(chunk_prompt))

    condensed_content = "\n\n".join(condensed_chunks)

    summary_prompt = (
        "Generate a concise summary (within 100 words) of this tutorial that helps a code "
        "generation LLM understand:\n"
        "1. What specific implementation knowledge or techniques it can find in this tutorial\n"
        "2. What coding tasks this tutorial can help with\n"
        "3. Key features or functionalities covered\n\n"
        f"Tutorial content:\n{condensed_content}\n\n"
        'Provide the summary in a single paragraph starting with "Summary: ".'
    )
    summary = llm.assistant_chat(summary_prompt)
    if not summary.startswith("Summary: "):
        summary = "Summary: " + summary

    # Truncate condensed content if too long, preserving structure.
    if len(condensed_content) > max_length:
        cut = condensed_content[:max_length].rfind("\n#")
        if cut <= 0:
            cut = condensed_content[:max_length].rfind("\n\n")
            if cut == -1:
                cut = max_length
        condensed_content = condensed_content[:cut] + "\n\n...(truncated)"

    return summary, condensed_content, body


def _write_outputs(
    condensed_file: Path,
    summary: str,
    condensed_content: str,
    title: str,
) -> None:
    """Write condensed_tutorials/<file>.md only. tutorials/ is never touched."""
    condensed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(condensed_file, "w", encoding="utf-8") as f:
        f.write(f"# Condensed: {title}\n\n")
        f.write(f"{summary}\n\n")
        f.write(
            "*This is a condensed version that preserves essential implementation details and context.*\n\n"
        )
        f.write(condensed_content)


# --------------------------------------------------------------------------- #
# Per-tool driver
# --------------------------------------------------------------------------- #
def _process_tool(
    tool_name: str,
    registry: ToolsRegistry,
    llm_cfg: DictConfig,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    """Returns (succeeded, failed) file counts."""
    tool_path = registry.get_tool_path(tool_name)
    if tool_path is None:
        logger.warning(f"[{tool_name}] not found in registry, skipping")
        return (0, 0)

    tutorials_dir = tool_path / "tutorials"
    condensed_dir = tool_path / "condensed_tutorials"

    if not tutorials_dir.exists():
        logger.warning(f"[{tool_name}] no tutorials/ directory, skipping")
        return (0, 0)

    md_files = sorted(tutorials_dir.rglob("*.md"))
    if not md_files:
        logger.warning(f"[{tool_name}] no .md files under tutorials/, skipping")
        return (0, 0)

    # When --only-missing is set, drop any tutorial whose condensed counterpart
    # already exists. The relative path under condensed_tutorials/ mirrors the
    # one under tutorials/ exactly, so a simple existence check is enough.
    skipped_existing = 0
    if args.only_missing:
        filtered: List[Path] = []
        for f in md_files:
            rel = f.relative_to(tutorials_dir)
            if (condensed_dir / rel).exists():
                skipped_existing += 1
                logger.debug(f"  skip (already condensed): {rel}")
            else:
                filtered.append(f)
        md_files = filtered

    if not md_files:
        logger.info(
            f"[{tool_name}] nothing to do "
            f"({skipped_existing} already condensed, --only-missing set)"
        )
        return (0, 0)

    extra = (
        f" ({skipped_existing} skipped as already condensed)"
        if skipped_existing
        else ""
    )
    logger.info(f"[{tool_name}] found {len(md_files)} tutorial(s) to regenerate{extra}")
    if args.dry_run:
        for f in md_files:
            rel = f.relative_to(tutorials_dir)
            logger.info(f"  would regenerate: condensed_tutorials/{rel}")
        return (len(md_files), 0)

    succeeded = 0
    failed = 0
    for md in md_files:
        rel = md.relative_to(tutorials_dir)
        condensed_path = condensed_dir / rel
        try:
            logger.info(f"  -> {rel}")
            summary, condensed_content, body = _condense_one_file(
                tutorial_file=md,
                tool_name=tool_name,
                llm_cfg=llm_cfg,
                chunk_size=args.chunk_size,
                max_length=args.max_length,
            )
            title = extract_title_from_markdown(body)
            _write_outputs(
                condensed_file=condensed_path,
                summary=summary,
                condensed_content=condensed_content,
                title=title,
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.error(f"     FAILED: {e}")
            logger.debug(traceback.format_exc())
            raise  # fail-fast: don't waste tokens on partial runs

    logger.info(f"[{tool_name}] done: {succeeded} ok, {failed} failed")
    return (succeeded, failed)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    llm_cfg = _load_llm_config()
    if not args.dry_run:
        logger.info(f"Loaded LLM config from {CONFIG_PATH}")
        logger.info(
            f"LLM: provider={llm_cfg.provider} model={llm_cfg.model} "
            f"max_tokens={llm_cfg.get('max_tokens')} temp={llm_cfg.get('temperature')}"
        )

    registry = ToolsRegistry()
    all_tools = registry.list_tools()
    if args.tool:
        unknown = [t for t in args.tool if t not in all_tools]
        if unknown:
            logger.error(f"Unknown tool(s): {unknown}. Known: {all_tools}")
            return 2
        tool_list = args.tool
    else:
        tool_list = all_tools
    logger.info(f"Processing tools: {tool_list}")

    total_ok = 0
    total_fail = 0
    for tool_name in tool_list:
        ok, fail = _process_tool(tool_name, registry, llm_cfg, args)
        total_ok += ok
        total_fail += fail

    logger.info("=" * 60)
    logger.info(f"TOTAL: {total_ok} succeeded, {total_fail} failed")

    # Token usage report (skipped in dry-run since no LLM calls were made).
    if not args.dry_run:
        try:
            usage = ChatLLMFactory.get_total_token_usage(save_path=args.token_report)
            logger.info(f"Token usage: {usage}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not collect token usage: {e}")

    if total_ok == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
