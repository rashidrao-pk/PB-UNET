import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portable_bridge_unet.plotting import plot_training_history, plot_evaluation_metrics

def test_training_artifacts(tmp_path):
    history = [dict(epoch=1, train_loss=.8, val_loss=.9, train_dice=.2, val_dice=.1, lr=.001)]
    plot_training_history(history, tmp_path)
    for extension in ("png", "pdf"):
        assert (tmp_path / f"training_curves.{extension}").stat().st_size > 0
    assert not plt.get_fignums()

def test_evaluation_with_missing_hd95_and_nonfinite_scores(tmp_path):
    frame = pd.DataFrame(dict(dice=[.2, .5, np.nan], iou=[.1, .3, .4]))
    for suffix in ("raw", "postprocessed"):
        plot_evaluation_metrics(frame, tmp_path, suffix)
        for name in ("metric_distributions", "metric_summary"):
            for extension in ("png", "pdf"):
                assert (tmp_path / f"{name}_{suffix}.{extension}").stat().st_size > 0
    assert not plt.get_fignums()

def test_empty_results(tmp_path):
    plot_training_history([], tmp_path)
    plot_evaluation_metrics(pd.DataFrame(), tmp_path)
    assert not list(tmp_path.iterdir())
