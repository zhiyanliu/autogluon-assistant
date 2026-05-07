import logging
import os
from pathlib import Path
from typing import List

from ..prompts import DescriptionFileRetrieverPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)

# File extensions treated as text-based description / documentation candidates
# for the glob-fallback safety net (see __call__ below).
_DESCRIPTION_FILE_EXTENSIONS = ("*.md", "*.txt", "*.rst")


class DescriptionFileRetrieverAgent(BaseAgent):
    """
    Identify potential description files from the data prompt.
    Only identifies files, does not read content.

    Agent Input:
    - data_prompt: Text string containing data prompt

    Agent Output:
    - List[str]: List of identified description filenames
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.description_file_retriever_llm_config = llm_config
        self.description_file_retriever_prompt_template = prompt_template

        self.description_file_retriever_prompt = DescriptionFileRetrieverPrompt(
            llm_config=self.description_file_retriever_llm_config,
            manager=self.manager,
            template=self.description_file_retriever_prompt_template,
        )

        if self.description_file_retriever_llm_config.multi_turn:
            self.description_file_retriever_llm = init_llm(
                llm_config=self.description_file_retriever_llm_config,
                agent_name="description_file_retriever",
                multi_turn=self.description_file_retriever_llm_config.multi_turn,
            )

    def __call__(self) -> List[str]:
        self.manager.log_agent_start("DescriptionFileRetrieverAgent: identifying description files from data prompt.")

        # Build prompt for identifying description files
        prompt = self.description_file_retriever_prompt.build()

        if not self.description_file_retriever_llm_config.multi_turn:
            self.description_file_retriever_llm = init_llm(
                llm_config=self.description_file_retriever_llm_config,
                agent_name="description_file_retriever",
                multi_turn=self.description_file_retriever_llm_config.multi_turn,
            )

        response = self.description_file_retriever_llm.assistant_chat(prompt)

        description_files = self.description_file_retriever_prompt.parse(response)

        # Defensive merge: union LLM's selections with all .md / .txt / .rst files that
        # actually exist in input_data_folder, then drop any path that doesn't resolve to a
        # real file. This protects against two LLM failure modes that have been observed
        # with non-English (e.g. Chinese) filenames:
        #
        #   (a) Hallucinated paths — LLM translates "05_AutoGluon经验与避坑.md" to
        #       "05_工具使用指南.md" or fabricates entries like "06_反模式清单.md" that
        #       don't exist on disk. These would trigger silent read-failure warnings in
        #       TaskDescriptorPrompt.get_description_files_contents() and silently drop
        #       critical task-defining content. The os.path.isfile filter below removes them.
        #
        #   (b) Over-restrictive selection — LLM picks only one of N description files
        #       (saw this happen when our 5-file doc package collapsed to a single file).
        #       The glob fallback ensures every real .md / .txt / .rst in the input folder
        #       is included regardless of LLM's filtering judgment.
        #
        # Net result: LLM's job becomes "exclude irrelevant files" (which it does well via
        # the existing prompt), while file discovery for description-style docs becomes
        # deterministic.
        try:
            input_dir = Path(self.manager.input_data_folder)
            glob_paths: List[str] = []
            for pattern in _DESCRIPTION_FILE_EXTENSIONS:
                glob_paths.extend(str(p) for p in input_dir.rglob(pattern))

            llm_paths = list(description_files) if description_files else []

            combined = sorted(set(llm_paths) | set(glob_paths))
            existing_files = [p for p in combined if os.path.isfile(p)]

            hallucinated = [p for p in llm_paths if not os.path.isfile(p)]
            added_by_glob = [p for p in glob_paths if p not in llm_paths and os.path.isfile(p)]

            if hallucinated:
                logger.warning(
                    "DescriptionFileRetrieverAgent: dropping %d non-existent path(s) returned by "
                    "the LLM (likely translated/hallucinated filenames): %s",
                    len(hallucinated),
                    hallucinated,
                )
            if added_by_glob:
                logger.info(
                    "DescriptionFileRetrieverAgent: glob fallback added %d file(s) the LLM did "
                    "not select: %s",
                    len(added_by_glob),
                    added_by_glob,
                )

            # Persist the post-merge list so the on-disk audit trail matches what downstream
            # agents actually see (the raw LLM response is still preserved in
            # description_file_retriever_response.txt by parse()).
            self.manager.save_and_log_states(
                content=existing_files,
                save_name="description_files.txt",
                per_iteration=False,
                add_uuid=False,
            )

            description_files = existing_files
        except Exception as e:
            logger.error(
                "DescriptionFileRetrieverAgent: glob-fallback merge failed (%s); "
                "falling back to LLM output as-is.",
                e,
            )

        self.manager.log_agent_end("DescriptionFileRetrieverAgent: description file list extracted.")

        return description_files
