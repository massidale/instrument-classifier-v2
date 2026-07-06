import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ablation import ABLATION_COMBOS, combo_name, format_ablation_table  # noqa: E402,F401
