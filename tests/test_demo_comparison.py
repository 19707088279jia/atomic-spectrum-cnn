import pandas as pd

from atomic_spectrum_ai.demo_comparison import build_final_comparison


def test_final_comparison_reindexes_all_six_targets() -> None:
    scores = pd.DataFrame(
        {
            "element": ["Cu", "Zn", "Pb", "Cd", "Mn", "Mg"],
            "status": ["POSSIBLE", "DETECTED", "NOT DETECTED", "POSSIBLE", "DETECTED", "NOT DETECTED"],
            "reference_matching_score": [0.2, 0.8, 0.1, 0.3, 0.9, 0.05],
        }
    )
    truth = {"Zn": 100.0, "Mn": 500.0, "Cd": 0.0, "Mg": 0.0, "Cu": 25.0, "Pb": 0.0}
    comparison = build_final_comparison(
        scores,
        truth,
        {"Zn": 0.8, "Mn": 0.2, "Cd": 0.4, "Mg": 0.2, "Cu": 0.6, "Pb": 0.1},
    )
    assert comparison["Element"].tolist() == ["Zn", "Mn", "Cd", "Mg", "Cu", "Pb"]