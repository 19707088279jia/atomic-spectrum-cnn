import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_z903_spectrum.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_z903_spectrum", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load plotting script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_real_z903_csv_has_fixed_png_dimensions(tmp_path: Path) -> None:
    module = load_plot_module()
    input_path = ROOT / "data" / "raw" / "pds_z903" / "plibs_z903_7d16.csv"
    output_path = module.plot_z903_spectrum(input_path, tmp_path / "spectrum.png")

    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.size == (1500, 600)
        assert image.format == "PNG"


def test_expand_inputs_accepts_multiple_files() -> None:
    module = load_plot_module()
    paths = module.expand_inputs(
        [
            "data/raw/pds_z903/plibs_z903_7d16.csv",
            "data/raw/pds_z903/plibs_z903_agv1a.csv",
        ]
    )
    assert [path.name for path in paths] == ["plibs_z903_7d16.csv", "plibs_z903_agv1a.csv"]