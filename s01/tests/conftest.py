import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY",      "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

S01_DIR = Path(__file__).parent.parent
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(S01_DIR))
