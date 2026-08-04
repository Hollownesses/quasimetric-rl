from pathlib import Path

import matplotlib.pyplot as plt

from minimal_qrl.visualize_diagnostic_scenarios import (
    visualize_diagnostic_scenarios,
)


def test_visualizer_writes_three_maps_and_overview(tmp_path: Path):
    paths = visualize_diagnostic_scenarios(
        tmp_path,
        split="validation",
        sample_index=1,
        communication_resolution=40,
        dpi=60,
    )
    assert set(paths) == {
        "u_trap",
        "comm_shadow_corridor",
        "easy_open",
        "overview",
    }
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 1_000
        image = plt.imread(path)
        assert image.shape[0] > 100
        assert image.shape[1] > 100
