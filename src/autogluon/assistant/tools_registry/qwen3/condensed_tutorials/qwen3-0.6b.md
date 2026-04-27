# Condensed: Qwen3-0.6B

Summary: This tutorial covers implementing Qwen3-0.6B, a causal LM with thinking/non-thinking dual modes, using HuggingFace Transformers (>=4.51.0). It provides code for model loading, chat template application with `enable_thinking` toggle, parsing `<think>` blocks via token ID 151668, and deployment via SGLang/vLLM. It details dynamic per-turn thinking control using `/think` and `/no_think` soft switches, agentic tool-calling setup with Qwen-Agent (MCP servers, built-in tools), and critical sampling parameters per mode (e.g., Temperature 0.6/0.7). Key warnings include avoiding greedy decoding in thinking mode and excluding thinking content from multi-turn history.

*This is a condensed version that preserves essential implementation details and context.*

# Qwen3-0.6B - Implementation Guide

## Model Specs
- Parameters: 0.6B (0.44B non-embedding)
- Layers: 28 | Attention Heads: 16 Q / 8 KV (GQA)
- Context Length: 32,768

## Requirements
- `transformers>=4.51.0` (otherwise: `KeyError: 'qwen3'`)

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True  # Default is True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Parse thinking content (151668 = </think> token)
try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

## Deployment

```shell
# SGLang (>=0.4.6.post1)
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3

# vLLM (>=0.8.5)
vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
```

## Thinking Mode Control

### Hard Switch via `enable_thinking`

- **`enable_thinking=True`** (default): Generates `<think>...</think>` block followed by response.
- **`enable_thinking=False`**: No thinking content, behaves like Qwen2.5-Instruct.

### Soft Switch via User Input (`/think` and `/no_think`)

When `enable_thinking=True`, append `/think` or `/no_think` to user messages to dynamically toggle per-turn. Model follows the most recent instruction.

> **Important**: With `enable_thinking=True`, output always includes `<think>...</think>` block (may be empty). With `enable_thinking=False`, soft switches are ignored entirely.

## Agentic Use (Qwen-Agent)

```python
from qwen_agent.agents import Assistant

llm_cfg = {
    'model': 'Qwen3-0.6B',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}

tools = [
    {'mcpServers': {
        'time': {'command': 'uvx', 'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}
    }},
    'code_interpreter',
]

bot = Assistant(llm=llm_cfg, function_list=tools)
messages = [{'role': 'user', 'content': 'Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
```

## Best Practices

| Setting | Thinking Mode | Non-Thinking Mode |
|---------|--------------|-------------------|
| Temperature | 0.6 | 0.7 |
| TopP | 0.95 | 0.8 |
| TopK | 20 | 20 |
| MinP | 0 | 0 |

**Critical warnings:**
- **DO NOT use greedy decoding** in thinking mode — causes degradation and endless repetitions.
- Set `presence_penalty` to 1.5 if encountering endless repetitions (values 0–2; higher values may cause language mixing).
- Use `max_new_tokens=32768` for most queries; `38,912` for complex math/coding benchmarks.
- **Multi-turn conversations**: Historical assistant messages should exclude thinking content (only final output). The Jinja2 template handles this automatically.

**Benchmark prompts:**
- Math: append "Please reason step by step, and put your final answer within \boxed{}."
- MCQ: append "Please show your choice in the `answer` field with only the choice letter, e.g., `\"answer\": \"C\"`."