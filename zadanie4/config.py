"""Single source of truth: paths, search/budget params, function list, DESABOA params."""
from pathlib import Path

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CONV_DIR = RESULTS_DIR / "convergence"
BOX_DIR = RESULTS_DIR / "boxplots"

# ---- experiment setup ------------------------------------------------------
SEED = 42
D = 10                       # problem dimension
POP = 30                     # population size
MAX_FES = 10_000 * D         # CEC2017 standard budget = 100_000 for D=10
NUM_RUNS = 51                # independent runs per (algorithm, function)
N_CHECKPOINTS = 100          # convergence curve samples (evenly spaced over FEs)

# CEC2017 function IDs (F2 skipped: unstable, officially excluded).
# Coverage: F1 unimodal | F4,F5,F9 multimodal | F10,F14,F17 hybrid | F20,F21,F24,F27,F29 composition.
FUNCTIONS = [1, 4, 5, 9, 10, 14, 17, 20, 21, 24, 27, 29]

FUNCTION_CATEGORY = {
    1: "unimodalna",
    4: "multimodalna", 5: "multimodalna", 9: "multimodalna",
    10: "hybrydowa", 14: "hybrydowa", 17: "hybrydowa",
    20: "złożona", 21: "złożona", 24: "złożona", 27: "złożona", 29: "złożona",
}

# Algorithms to run. First is the statistical baseline for Wilcoxon pairing.
ALGORITHMS = ["SABOA", "DESABOA"]
BASELINE = "SABOA"

# ---- algorithm hyper-parameters -------------------------------------------
BOA_C0 = 0.01                # initial sensory modality
BOA_A = 0.1                  # power exponent
SWITCH_P = 0.8               # probability of local search (rand > p -> global)

# DESABOA diversity-enhancement parameters
DESABOA = {
    "theta": 0.10,           # trigger mutation when diversity < theta * D0
    "m_frac": 0.20,          # fraction of worst individuals to perturb
    "levy_beta": 1.5,        # Mantegna Levy stability index
    "levy_alpha": 0.01,      # Levy step scale (times (ub-lb))
    "cauchy_gamma": 0.10,    # Cauchy long-jump scale (times (ub-lb))
    "stagnation_iters": 5,   # also trigger if g* unchanged this many iters
    "mut_mix": 0.5,          # P(Levy) vs Cauchy per perturbed individual
    "use_obl": True,         # opposition-based learning at init
}

PLOT_DPI = 120


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, RUNS_DIR, CONV_DIR, BOX_DIR):
        d.mkdir(parents=True, exist_ok=True)
