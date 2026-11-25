"""Model presets and the training data mix, all in one place."""
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_mult: int = 4
    seq_len: int = 2048
    logit_softcap: float = 15.0

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # 2.2M params (0.1M in the transformer body): CPU smoke tests, and small enough to read every weight
    "nano": ModelConfig(n_layer=2, n_head=2, d_model=64, seq_len=256),
    # 11.5M params: quick CPU experiments
    "micro": ModelConfig(n_layer=4, n_head=4, d_model=256, seq_len=512),
    # 42M params: fits a smaller GPU budget
    "small": ModelConfig(n_layer=8, n_head=8, d_model=512, seq_len=2048),
    # 110M params (85M in the transformer body): the $1.81 run
    "base": ModelConfig(n_layer=12, n_head=12, d_model=768, seq_len=2048),
}

# Languages taken from Stack-Edu (educational-quality code filtered from The Stack v2).
STACK_EDU_REPO = "utter-project/stack-edu-HF"
AGENT_REPO = "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"

# Each source: (kind, selector). "stack_edu" selects by language, "agent" renders chat trajectories.
SOURCES = {
    "python": ("stack_edu", "Python"),
    "javascript": ("stack_edu", "JavaScript"),
    "typescript": ("stack_edu", "TypeScript"),
    "shell": ("stack_edu", "Shell"),
    "markdown": ("stack_edu", "Markdown"),
    "c": ("stack_edu", "C"),
    "cpp": ("stack_edu", "C++"),
    "java": ("stack_edu", "Java"),
    "go": ("stack_edu", "Go"),
    "rust": ("stack_edu", "Rust"),
    "sql": ("stack_edu", "SQL"),
    "agent": ("agent", AGENT_REPO),
}

# Share of training sequences per source. "pretrain" is used for most of the run;
# "anneal" takes over during the learning-rate cooldown and boosts agent sessions,
# Python and shell, which is what the coding agent leans on most.
MIXES = {
    "pretrain": {
        "python": 0.28, "javascript": 0.09, "typescript": 0.06, "shell": 0.05,
        "markdown": 0.12, "c": 0.05, "cpp": 0.06, "java": 0.06, "go": 0.04,
        "rust": 0.04, "sql": 0.02, "agent": 0.13,
    },
    "anneal": {
        "python": 0.30, "javascript": 0.05, "typescript": 0.03, "shell": 0.08,
        "markdown": 0.08, "c": 0.02, "cpp": 0.02, "java": 0.02, "go": 0.02,
        "rust": 0.02, "sql": 0.01, "agent": 0.35,
    },
    # small mix for the CPU smoke test: few sources so it downloads little
    "smoke": {"python": 0.5, "shell": 0.2, "agent": 0.3},
}

# The cooldown (and the anneal mix) covers the last part of the time budget.
COOLDOWN_FRAC = 0.25
