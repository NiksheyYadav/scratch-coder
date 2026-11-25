"""Chat template shared by pretraining, agent fine-tuning and the agent harness.

The agent protocol copies mini-swe-agent, which is the format of the open agent
sessions we pretrain on: the assistant writes a THOUGHT and exactly one ```bash```
block; the shell output comes back as a user message wrapped in
<returncode>/<output> tags; the agent finishes by echoing FINAL_MARKER.
"""
from __future__ import annotations

import re

EOT = "<|endoftext|>"
END = "<|end|>"
PAD = "<|pad|>"
ROLES = ("system", "user", "assistant")
SPECIAL_TOKENS = [EOT, PAD, END] + [f"<|{r}|>" for r in ROLES] + [f"<|reserved_{i}|>" for i in range(10)]

FINAL_MARKER = "MINI_SWE_AGENT_FINAL_OUTPUT"
SUBMIT_COMMAND = f"echo {FINAL_MARKER} && git add -A && git diff --cached"

SYSTEM_PROMPT = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

FORMAT_ERROR = """Please always provide EXACTLY ONE action in triple backticks, found {n} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that)."""

_BASH_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)


def header(role: str) -> str:
    return f"<|{role}|>\n"


def render_message(role: str, content: str) -> str:
    return f"{header(role)}{content}{END}\n"


def render(messages: list[dict], add_generation_prompt: bool = False, eot: bool = True) -> str:
    """Render a conversation to text. Roles other than system/assistant become user."""
    out = []
    for m in messages:
        role = m["role"] if m["role"] in ROLES else "user"
        out.append(render_message(role, m["content"]))
    if add_generation_prompt:
        out.append(header("assistant"))
    elif eot:
        out.append(EOT)
    return "".join(out)


def encode_with_mask(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids plus a 0/1 mask that is 1 only on assistant content and its END token."""
    ids, mask = [], []
    for m in messages:
        role = m["role"] if m["role"] in ROLES else "user"
        h = tok.encode(header(role)).ids
        body = tok.encode(m["content"] + END).ids
        nl = tok.encode("\n").ids
        ids += h + body + nl
        train = 1 if role == "assistant" else 0
        mask += [0] * len(h) + [train] * len(body) + [0] * len(nl)
    eot = tok.token_to_id(EOT)
    ids.append(eot)
    mask.append(0)
    return ids, mask


def parse_actions(text: str) -> list[str]:
    return [a.strip() for a in _BASH_RE.findall(text)]


def format_observation(returncode: int, output: str, limit: int = 4000) -> str:
    if len(output) > limit:
        half = limit // 2
        output = output[:half] + f"\n... [{len(output) - limit} characters elided] ...\n" + output[-half:]
    return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
