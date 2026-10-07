import numpy as np
import pandas as pd
import pytest
from portable_bridge_unet.comparison import read_paired, paired_statistics, bootstrap_differences

def test_alignment_and_rejection(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame(dict(image_path=["a", "b"], dice=[.9, .3])).to_csv(a, index=False)
    pd.DataFrame(dict(image_path=["b", "a"], dice=[.1, .8])).to_csv(b, index=False)
    pb, un = read_paired(a, b)
    np.testing.assert_allclose(pb.dice-un.dice, [.1, .2])
    pd.DataFrame(dict(image_path=["a", "a"], dice=[.1, .8])).to_csv(b, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_paired(a, b)
    pd.DataFrame(dict(image_path=["a", "c"], dice=[.1, .8])).to_csv(b, index=False)
    with pytest.raises(ValueError, match="identities differ"):
        read_paired(a, b)

def test_hd95_direction_and_nonfinite():
    pb = pd.DataFrame(dict(hd95=[1., 4., np.inf], dice=[.5, .5, np.nan]))
    un = pd.DataFrame(dict(hd95=[2., 3., 0.], dice=[.5, .5, .7]))
    hd, dice = paired_statistics(pb, un, ["hd95", "dice"])
    assert hd["pb_wins"] == hd["unet_wins"] == 1
    assert hd["excluded_nonfinite"] == 1
    assert dice["ties"] == 2 and dice["wilcoxon_p"] == 1

def test_bootstrap_reproducible_and_constant():
    pb, un = pd.DataFrame(dict(dice=[1., 1.])), pd.DataFrame(dict(dice=[.5, .5]))
    first, means, medians = bootstrap_differences(pb, un, resamples=100)
    second, _, _ = bootstrap_differences(pb, un, resamples=100)
    assert first == second
    assert first["mean_ci95"] == first["median_ci95"] == [.5, .5]
    assert (means == .5).all() and (medians == .5).all()
    with pytest.raises(ValueError, match="no finite"):
        bootstrap_differences(pd.DataFrame(dict(dice=[np.nan])), pd.DataFrame(dict(dice=[.5])))
