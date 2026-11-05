"""Connectivity budgets and simulation behaviour on random and silent input. Run: pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vc_rsc.config import load_config  # noqa: E402
from vc_rsc.connectivity import build_network  # noqa: E402
from vc_rsc.simulate import simulate  # noqa: E402

NET = ROOT / "configs/networks/vc_rsc_v2.yaml"
GEN1 = ROOT / "configs/cameras/gen1_240x304.yaml"
GH, GW = 120, 152  # Gen1 at group_size 2


def _network(cfg, f2=0.6, f3=0.3):
    return build_network(cfg, GH, GW, np.random.default_rng(cfg.random_seed), f2, f3)


def test_synapse_budget_per_population():
    cfg = load_config(NET, GEN1)
    net = _network(cfg)
    assert net["rsc_src"].shape == (100, 75)
    fb_only = [n for n in range(275) if not net["is_sens"][n]]
    sens = [n for n in range(275) if net["is_sens"][n]]
    assert all(len(net["l1_fb"][n]) == 75 for n in fb_only)
    assert all(len(net["pix_groups"][i]) + len(net["l1_fb"][n]) == 75
               for i, n in enumerate(net["sens_idx"]))
    # rsc_fraction_L2 = 0.6 -> 45 RSC + 30 L1; rsc_fraction_L3 = 0.3 -> round(22.5) = 22 RSC + 53 L2
    assert len(net["l2_fb"][0]) == 45 and len(net["l2_ff"][0]) == 30
    assert len(net["l3_fb"][0]) == 22 and len(net["l3_ff"][0]) == 53
    assert sum(len(g) for g in net["pix_groups"]) == GH * GW


def test_random_input_runs_and_is_finite():
    cfg = load_config(NET, GEN1)
    cfg.analysis.cut_steps = [0]
    rng = np.random.default_rng(0)
    pooled = (rng.random((6, GH * GW)) < 0.01).astype(np.float32) * rng.random((6, GH * GW)).astype(np.float32)
    res = simulate(cfg, pooled, _network(cfg))
    assert res["S"]["L1"].shape == (6, 275)
    assert res["S"]["RSC"].shape == (6, 100)
    assert np.isfinite(res["V"]["L1"]).all()


def test_silent_input_gives_no_spikes():
    cfg = load_config(NET, GEN1)
    cfg.analysis.cut_steps = []
    pooled = np.zeros((6, GH * GW), np.float32)
    res = simulate(cfg, pooled, _network(cfg))
    assert all(res["S"][k].sum() == 0 for k in ["L1", "L2", "L3", "RSC"])
