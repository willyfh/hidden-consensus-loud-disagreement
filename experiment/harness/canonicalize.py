"""Shared measure-choice canonicalization, used by analyze.py, analyze_cross_model.py, and
analyze_haiku_supplement.py so all three stay in sync.

Uniform operationalization-choice rule (fixed 2026-09-30, prompted by a reviewer): every
abstract hypothesis that leaves the model unspecified (H1, H3, H4, H6) tracks which model(s)
are the target of the analysis as part of the canonical framing, not just H1. Before this
fix, H1's model-pair choice was tracked but H3's/H4's/H6's equivalent choice ("which model do
I test/explain/calibrate?") was not, which underweighted their real diversity -- confirmed
empirically: several H3/H4/H6 replicates state the model only in methodological_choices, not
in primary_metric_name, so a name-only rule invisibly flattened real variation (H6 alone goes
from 1 to 6 unique framings once this is fixed). Named entities that are OUTPUTS of the
analysis rather than choices made before running it (H3's specific top feature) are still not
tracked, since that conflates measure-choice diversity with result diversity.
"""
import re

MODEL_ALIASES = {
    "histgradientboosting": "hgb", "hist gradient boosting": "hgb", "hgb": "hgb",
    "gradientboosting": "gb", "gradient boosting": "gb", "gbm": "gb",
    "randomforest": "rf", "random forest": "rf", "rf": "rf",
    "logisticregression": "logreg", "logistic regression": "logreg", "logreg": "logreg",
    "knn": "knn", "k-nearest neighbors": "knn",
    "gaussiannb": "nb", "naive bayes": "nb",
    "best tree ensemble": "best_tree_ensemble",
    "best": "best", "worst": "worst", "max": "best", "min": "worst",
}

# Full sklearn class names too, for scanning free-text methodological_choices where the
# metric name itself doesn't state the model.
CLASS_ALIASES = {
    "randomforestclassifier": "rf", "random forest": "rf", "randomforest": "rf",
    "histgradientboostingclassifier": "hgb", "hist gradient boosting": "hgb", "histgradientboosting": "hgb",
    "gradientboostingclassifier": "gb", "gradient boosting": "gb", "gradientboosting": "gb",
    "logisticregression": "logreg", "logistic regression": "logreg",
    "kneighborsclassifier": "knn", "k-nearest neighbors": "knn",
    "gaussiannb": "nb", "naive bayes": "nb",
}

MULTI_MARKERS = ["averaged", "average over", "both", "across logreg", "across logistic",
                  "best of", "best imbalance", "best model", "best-", "best tree ensemble",
                  "best -", "max -", "min -", "worst"]


def base_metric(name: str) -> str:
    """The underlying statistical quantity, independent of wording, CV/split details, or
    which specific model/feature is named."""
    s = name.lower()
    if "feature importance" in s or "permutation importance" in s or "decrease in impurity" in s or "feature_importance" in s:
        s_noseed = re.sub(r"averaged? (over|across) \d+ (seeds?|reruns?|folds?|repeats?)", "", s)
        if any(k in s_noseed for k in ("consensus", "aggregated", "combined")) or \
           ("average" in s_noseed and "normalized" in s_noseed):
            return "feat_importance:consensus_multi_method"
        if "permutation" in s:
            return "feat_importance:permutation"
        if "impurity" in s or "gini" in s:
            return "feat_importance:impurity_mdi"
        return "feat_importance:unspecified"
    if "balanced accuracy" in s or "balanced-accuracy" in s:
        return "balanced_accuracy"
    if "macro-f1" in s or "macro f1" in s or "f1-macro" in s or "f1 macro" in s:
        return "macro_f1"
    if re.search(r"\bf1\b", s) and ">50k" in s.replace(" ", ""):
        return "f1_minority"
    if re.search(r"\bf1\b", s):
        return "macro_f1"  # bare "F1" in H4 context means macro-F1
    if "roc-auc" in s or "roc auc" in s or re.search(r"\bauc\b", s):
        return "roc_auc"
    if "calibration error" in s or re.search(r"\bece\b", s):
        return "ece"
    return "other:" + s[:40]


def h1_comparison_target(name: str) -> frozenset:
    """Which pair of model families H1 compares (order-independent)."""
    s = name.lower()
    m = re.search(r"\(([^)]*)\)", s)
    inside = m.group(1) if m else s
    inside = re.split(r",", inside)[0]
    parts = re.split(r"\s*-\s*|\sminus\s", inside)
    norm = [MODEL_ALIASES.get(p.strip()) for p in parts]
    norm = [p for p in norm if p]
    return frozenset(norm) if norm else frozenset(["unresolved"])


def _extract_models_from_text(text: str) -> frozenset:
    s = text.lower()
    return frozenset(v for k, v in CLASS_ALIASES.items() if k in s)


def single_model_target(name: str, methodological_choices) -> str:
    """For H3/H4/H6: which model(s) are the operationalization target -- a single
    normalized model tag, 'multi:<models>' if the replicate explicitly compares/averages
    across more than one, or 'unspecified' if genuinely never stated anywhere."""
    mc = "" if methodological_choices is None else str(methodological_choices)
    if mc.lower() == "nan":
        mc = ""
    full_text = f"{name} {mc}".lower()
    is_multi = any(marker in full_text for marker in MULTI_MARKERS)

    name_models = _extract_models_from_text(name)
    if name_models and not is_multi:
        if len(name_models) == 1:
            return next(iter(name_models))
        return "multi:" + "+".join(sorted(name_models))

    mc_models = _extract_models_from_text(mc)
    if is_multi and len(mc_models) >= 2:
        return "multi:" + "+".join(sorted(mc_models))
    if is_multi and name_models:
        return "multi:" + "+".join(sorted(name_models))
    if len(mc_models) == 1:
        return next(iter(mc_models))
    if len(mc_models) >= 2:
        for k, v in CLASS_ALIASES.items():
            if k in mc.lower():
                return v
    return "unspecified"


def canonicalize(hyp: str, name: str, methodological_choices=None) -> str:
    bm = base_metric(name)
    if hyp == "H1":
        return f"{bm}|{'+'.join(sorted(h1_comparison_target(name)))}"
    if hyp in ("H3", "H4", "H6"):
        return f"{bm}|{single_model_target(name, methodological_choices)}"
    return bm
