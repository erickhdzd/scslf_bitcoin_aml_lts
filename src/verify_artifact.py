"""Verify the stored scientific evidence of the anonymous Bitcoin AML artifact."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
EXPECTED_SEEDS = [42, 43, 44, 45, 46]
TOL = 1e-12

EXPECTED = {
    "mlp_target": {
        "f1": (0.5107157116722826, 0.024540276962721725),
        "mcc": (0.5069289154771042, 0.02926285251541851),
        "roc_auc": (0.8501057935328762, 0.014446373919187523),
        "pr_auc": (0.44974540046945394, 0.018507022501374593),
    },
    "transformer": {
        "f1": (0.545173344606537, 0.02157689251658203),
        "mcc": (0.5523400802494638, 0.03811643718343104),
        "roc_auc": (0.8400615404468439, 0.01655452198846706),
        "pr_auc": (0.5154245495060659, 0.025339630079801775),
    },
    "hybrid_transformer": {
        "f1": (0.5722724814135765, 0.008379900086773478),
        "mcc": (0.587509861035251, 0.011358263307661189),
        "roc_auc": (0.8373838014987804, 0.010840901098623764),
        "pr_auc": (0.5275578136165826, 0.00820540106929969),
    },
    "graphsage": {
        "f1": (0.4452320261473569, 0.009073401044439837),
        "mcc": (0.43529963513868725, 0.022621700327296997),
        "roc_auc": (0.8339193830028158, 0.022034846880084606),
        "pr_auc": (0.4441471800252209, 0.013103200588184594),
    },
    "transformer_graphsage": {
        "f1": (0.4807927018929144, 0.018446589711344073),
        "mcc": (0.47692955382505103, 0.025779436058607455),
        "roc_auc": (0.8385824945417513, 0.021276543341255044),
        "pr_auc": (0.44829044450565136, 0.027111060032478002),
    },
    "late_fusion": {
        "f1": (0.568582846307151, 0.008072136796411923),
        "mcc": (0.588012779646811, 0.012448581296172467),
        "roc_auc": (0.850889944499085, 0.00993092600473163),
        "pr_auc": (0.5376217663616172, 0.006850801829633646),
    },
}

IDENTIFYING_PATTERNS = [
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]

TEXT_EXTENSIONS = {
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".tsv",
}


class Verification:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"{label:<38} {status}{suffix}")
        if not condition:
            self.failures += 1


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def model_seeds(summary: dict) -> list[int]:
    if "seeds" in summary and isinstance(summary["seeds"], list):
        return [int(value) for value in summary["seeds"]]
    metrics = summary.get("metrics", {})
    f1_values = metrics.get("f1", {}).get("values", [])
    if len(f1_values) == 5:
        return EXPECTED_SEEDS
    return []


def verify_model(v: Verification, model: str) -> None:
    path = RESULTS / model / "aggregate_summary.json"
    v.check(f"{model}: aggregate summary", path.is_file())
    if not path.is_file():
        return

    data = load_json(path)
    seeds = model_seeds(data)
    if model == "graphsage":
        seed_ok = (
            data.get("seed_count") == 5
            and len(data.get("metrics", {}).get("f1", {}).get("values", [])) == 5
        )
    elif model == "late_fusion":
        seed_ok = [int(x) for x in data.get("seeds", [])] == EXPECTED_SEEDS
    else:
        seed_ok = seeds == EXPECTED_SEEDS

    v.check(f"{model}: seeds", seed_ok, "42–46")

    for metric, (expected_mean, expected_std) in EXPECTED[model].items():
        block = data.get("metrics", {}).get(metric, {})
        ok = close(block.get("mean", float("nan")), expected_mean) and close(
            block.get("standard_deviation", float("nan")), expected_std
        )
        v.check(f"{model}: {metric}", ok)


def verify_alpha(v: Verification) -> None:
    summary_path = RESULTS / "late_fusion" / "aggregate_summary.json"
    search_path = RESULTS / "late_fusion" / "global_alpha_search.csv"
    if not summary_path.is_file() or not search_path.is_file():
        v.check("Late fusion alpha evidence", False, "required files missing")
        return

    data = load_json(summary_path)
    alpha_h = float(data.get("selected_alpha_hybrid", float("nan")))
    alpha_g = float(data.get("selected_alpha_graphsage", float("nan")))
    val_f1 = float(
        data.get("selected_validation_summary", {}).get(
            "mean_validation_f1", float("nan")
        )
    )

    v.check("Selected alpha: Hybrid", close(alpha_h, 0.60))
    v.check("Selected alpha: GraphSAGE", close(alpha_g, 0.40))
    v.check("Selected validation mean F1", close(val_f1, 0.8461237810650355))

    rows = []
    with search_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def get_float(row: dict, candidates: tuple[str, ...]) -> float:
        for key in candidates:
            if key in row and row[key] not in ("", None):
                return float(row[key])
        raise KeyError(candidates)

    try:
        best = max(
            rows,
            key=lambda row: (
                get_float(row, ("mean_validation_f1",)),
                get_float(row, ("mean_validation_mcc",)),
                get_float(row, ("mean_validation_pr_auc",)),
                -get_float(row, ("std_validation_f1",)),
                -abs(get_float(row, ("alpha_hybrid", "alpha")) - 1.0),
            ),
        )
        best_alpha = get_float(best, ("alpha_hybrid", "alpha"))
        best_f1 = get_float(best, ("mean_validation_f1",))
        v.check("Alpha search maximum", close(best_alpha, 0.60))
        v.check("Alpha search F1", close(best_f1, 0.8461237810650355))
    except (KeyError, ValueError):
        v.check("Alpha search parse", False, "unexpected CSV schema")


def verify_checks(v: Verification) -> None:
    files = {
        "Publication check": RESULTS / "paper_assets_final" / "PUBLICATION_CHECK.txt",
        "Package check": RESULTS
        / "paper_assets_final"
        / "delivery"
        / "PACKAGE_CHECK.txt",
    }
    for label, path in files.items():
        if not path.is_file():
            v.check(label, False, "file missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        v.check(label, "PASS" in text and "FAIL" not in text)


def verify_anonymity(v: Verification) -> None:
    findings = []
    for root in (RESULTS, PROJECT_ROOT / "src"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for pattern in IDENTIFYING_PATTERNS:
                if pattern.search(text):
                    findings.append(path.relative_to(PROJECT_ROOT).as_posix())
                    break
    v.check("Anonymous text metadata", not findings, f"{len(findings)} finding(s)")
    if findings:
        for item in findings[:10]:
            print(f"  residual identifier: {item}")


def main() -> int:
    print("Artifact verification")
    print("=====================")
    print()

    v = Verification()

    for model in EXPECTED:
        verify_model(v, model)

    verify_alpha(v)
    verify_checks(v)
    verify_anonymity(v)

    print()
    if v.failures:
        print(f"Artifact verification: FAIL ({v.failures} failed check(s))")
        return 1

    print("Artifact verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
