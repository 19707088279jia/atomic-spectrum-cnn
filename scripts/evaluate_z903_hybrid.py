#!/usr/bin/env python
"""Run validation-only ablations and shortcut checks for saved hybrid models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from atomic_spectrum_ai.hybrid_windows import HybridWindowDataset
from atomic_spectrum_ai.models.z903_hybrid import HybridZ903CNN

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
RAW = dict(zip(TARGETS, (100, 500, 0.05, 5, 25, 10), strict=True))
PROB = dict(zip(TARGETS, (0.705, 0.235, 0.5, 0.355, 0.505, 0.405), strict=True))


def collate(batch):
    return (
        torch.stack([item[0] for item in batch]),
        [torch.stack([item[1][i] for item in batch]) for i in range(6)],
        torch.stack([item[2] for item in batch]),
        torch.stack([item[3] for item in batch]),
    )


def scores(frame, probabilities):
    rows = []
    for index, target in enumerate(TARGETS):
        raw = pd.to_numeric(
            frame[f"{target}_raw" if target != "Mg" else "MgO_raw"], errors="coerce"
        ).to_numpy()
        known = frame[f"{target}_mask"].to_numpy() > 0
        truth = (raw[known] > RAW[target]).astype(int)
        rows.append(
            {
                "target": target,
                "pr_auc": average_precision_score(truth, probabilities[known, index]),
                "f1": f1_score(truth, probabilities[known, index] >= PROB[target], zero_division=0),
            }
        )
    return rows


def main() -> None:
    dataset = HybridWindowDataset(
        ROOT / "data/processed/z903_val.csv", ROOT / "data/processed/z903_target_windows.csv"
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)
    full, branches, _, _ = next(iter(loader))
    lengths = [len(dataset.indices[target]) for target in TARGETS]
    output = ROOT / "outputs/z903/hybrid_eval"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("z903_hybrid", "z903_hybrid_physics"):
        model = HybridZ903CNN(lengths)
        model.load_state_dict(
            torch.load(ROOT / f"models/{name}_best.pt", map_location="cpu")["model_state_dict"]
        )
        model.eval()
        with torch.no_grad():
            normal = torch.sigmoid(model(full, branches)).numpy()
            disabled_global = torch.sigmoid(model(full, branches, disable_global=True)).numpy()
            disabled_lines = torch.sigmoid(model(full, branches, disable_lines=True)).numpy()
        for label, probabilities in (
            ("full", normal),
            ("global_disabled", disabled_global),
            ("line_disabled", disabled_lines),
        ):
            for row in scores(dataset.frame, probabilities):
                rows.append({"variant": name, "ablation": label, **row})
        occlusion = []
        for index, target in enumerate(TARGETS):
            altered = [value.clone() for value in branches]
            altered[index].zero_()
            with torch.no_grad():
                changed = torch.sigmoid(model(full, altered)).numpy()
            for output_index, output_target in enumerate(TARGETS):
                occlusion.append(
                    {
                        "variant": name,
                        "occluded_target": target,
                        "output": output_target,
                        "probability_change": float(
                            np.mean(changed[:, output_index] - normal[:, output_index])
                        ),
                    }
                )
        pd.DataFrame(occlusion).to_csv(output / f"{name}_occlusion.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "ablation_metrics.csv", index=False)
    selected = pd.DataFrame(rows)
    report = [
        "# Z-903 Hybrid Physics-Informed CNN",
        "",
        "All hybrid development metrics below use validation data only. The historical test split was not used for architecture, window, consistency-weight, or threshold selection.",
        "",
        "## Empirical Lines",
        "",
    ]
    windows = pd.read_csv(ROOT / "data/processed/z903_empirical_target_lines.csv")
    for target in TARGETS:
        report.append(
            f"- {target}: {int(((windows.target == target) & windows.selected).sum())} selected empirically ranked lines/windows"
        )
    report.extend(
        [
            "",
            "## Validation Comparison",
            "",
            "| Variant | Macro PR-AUC | Macro F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for variant in (
        "z903_cnn_robust_best",
        "z903_physics_constrained_best",
        "z903_hybrid",
        "z903_hybrid_physics",
    ):
        if variant == "z903_cnn_robust_best":
            data = pd.read_csv(ROOT / "outputs/z903/z903_cnn_robust/validation_metrics.csv")
            name = "full-spectrum"
        elif variant == "z903_physics_constrained_best":
            data = pd.read_csv(ROOT / "outputs/z903/physics_constrained/validation_metrics.csv")
            name = "hard-window"
        else:
            data = selected[(selected.variant == variant)].query("ablation == 'full'")
            name = variant
        report.append(f"| {name} | {data.pr_auc.mean():.4f} | {data.f1.mean():.4f} |")
    report.extend(
        [
            "",
            "## Hybrid Per-Target Validation",
            "",
            "| Target | Hybrid PR-AUC | Hybrid F1 | Physics-loss PR-AUC | Physics-loss F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in TARGETS:
        a = selected[
            (selected.variant == "z903_hybrid")
            & (selected.ablation == "full")
            & (selected.target == target)
        ].iloc[0]
        b = selected[
            (selected.variant == "z903_hybrid_physics")
            & (selected.ablation == "full")
            & (selected.target == target)
        ].iloc[0]
        report.append(f"| {target} | {a.pr_auc:.4f} | {a.f1:.4f} | {b.pr_auc:.4f} | {b.f1:.4f} |")
    report.extend(
        [
            "",
            "## Branch Ablations",
            "",
            "Global-disabled and line-disabled PR-AUC/F1 are in `outputs/z903/hybrid_eval/ablation_metrics.csv`.",
            "",
            "## Target-Branch Occlusion",
            "",
            "For each hybrid variant, zeroing one target branch was evaluated against all six outputs. Full results are in the variant occlusion CSVs.",
            "",
            "## Matrix-Matched Context",
            "",
            "The full-spectrum matrix-matched validation PR-AUC/F1 are retained from the prior audit; the hybrid matrix-matched subset should be interpreted from the same validation-only composition matching procedure before claiming reduced matrix dependence.",
            "",
            "## Assessment",
            "",
            "The hybrid preserves full-spectrum context while limiting each line branch to its own empirical windows. The first validation result is below the full-spectrum baseline, so target-line constraints have not yet improved predictive performance. Zn and Cu remain the most concerning targets; Cd remains uncertain because of its small known-label count.",
        ]
    )
    (ROOT / "reports/z903_hybrid_physics_cnn.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(selected.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
