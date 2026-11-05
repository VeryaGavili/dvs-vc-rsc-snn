"""Configuration checks. Run with: pytest tests -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vc_rsc.config import load_config, validate  # noqa: E402

NET = ROOT / "configs/networks/vc_rsc_v2.yaml"
GEN1 = ROOT / "configs/cameras/gen1_240x304.yaml"
GENX320 = ROOT / "configs/cameras/genx320_320x320.yaml"


def test_gen1_config_loads_and_derives_dt():
    cfg = load_config(NET, GEN1)
    assert cfg.time.dt_ms == pytest.approx(50.0 / 3.0)
    assert cfg.synapses.syn_per_neuron == 75
    assert cfg.camera.grouping.group_size == 2


def test_genx320_default_group_size_is_valid():
    load_config(NET, GENX320)


def test_genx320_with_group_size_2_exceeds_capacity():
    cfg = load_config(NET, GENX320)
    cfg.camera.grouping.group_size = 2
    with pytest.raises(ValueError, match="exceed the capacity"):
        validate(cfg)


def test_layer_sizes_must_match_group_sizes():
    cfg = load_config(NET, GEN1)
    cfg.groups.sizes.L1 = [25, 100]
    with pytest.raises(ValueError, match="groups.sizes.L1"):
        validate(cfg)


def test_fanin_must_sum_to_syn_per_neuron():
    cfg = load_config(NET, GEN1)
    cfg.synapses.rsc_fanin.L1 = 40
    with pytest.raises(ValueError, match="rsc_fanin"):
        validate(cfg)
