"""Check that profile windows stay within one section and ignore labels."""

import pytest

try:
    import torch
except (ImportError, OSError) as exc:
    pytest.skip(f"Optional CPU PyTorch unavailable: {exc}", allow_module_level=True)

import numpy as np
import pandas as pd

from pipetwin.agent2_physics.case2_neural_models import POINTS, ProfileNet, local_windows


def test_window_does_not_cross_section_or_depend_on_labels_and_row_order():
    frame = pd.DataFrame({"exposure_id": ["a", "b", "a", "b"], "center_m": [0, 0, 20, 20],
                          "altitude_relative_m": [1, 1000, 3, 1010], "y_mm": [2, 3, 4, 5]})
    windows = local_windows(frame, 100)
    np.testing.assert_array_equal(windows[0, 4:7], [[0, 0], [0, 1], [2, 1]])
    np.testing.assert_array_equal(windows[1, 4:7], [[0, 0], [0, 1], [10, 1]])
    shifted = frame.loc[[3, 0, 2, 1]].assign(y_mm=500)
    np.testing.assert_array_equal(local_windows(shifted, 100), windows[[3, 0, 2, 1]])


@pytest.mark.parametrize("architecture", ["mlp", "cnn", "transformer"])
def test_networks_handle_unavailable_profile_without_nonfinite_thickness(architecture):
    model = ProfileNet(architecture).eval()
    with torch.inference_mode():
        out = model(torch.zeros(3, POINTS, 2), torch.zeros(3, 2))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()
