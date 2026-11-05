"""Load and validate the YAML configuration (network + camera)."""
from pathlib import Path
import os

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """Dictionary with attribute access. Nested dictionaries are converted too."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(network_path, camera_path, data_path=None):
    """Merge the network and camera YAML files into one validated Config.

    data_path, if given, overrides camera.input.path (useful for a local copy of the recording).
    Relative paths are resolved against the repository root.
    """
    net = yaml.safe_load(Path(network_path).read_text())
    cam = yaml.safe_load(Path(camera_path).read_text())
    cfg = _wrap({**net, "camera": cam})

    if data_path is not None:
        cfg.camera.input.path = str(data_path)
    if cfg.camera.input.path is not None:
        p = Path(os.path.expanduser(str(cfg.camera.input.path)))
        cfg.camera.input.path = str(p if p.is_absolute() else REPO_ROOT / p)

    validate(cfg)
    return cfg


def validate(cfg):
    """Raise ValueError with all problems found, so a bad config fails before a long run."""
    P, S, G, cam = cfg.populations, cfg.synapses, cfg.groups, cfg.camera
    errors = []

    for layer, n in [("L1", P.n_L1), ("L2", P.n_L2), ("L3", P.n_L3), ("RSC", P.n_RSC)]:
        sizes = G.sizes[layer]
        if isinstance(sizes, list) and sum(sizes) != n:
            errors.append(f"groups.sizes.{layer} sums to {sum(sizes)}, the layer has {n} neurons")

    if sum(S.rsc_fanin.values()) != S.syn_per_neuron:
        errors.append(f"rsc_fanin sums to {sum(S.rsc_fanin.values())}, syn_per_neuron is {S.syn_per_neuron}")
    for key in ["rsc_fraction_L2", "rsc_fraction_L3"]:
        if not 0.0 <= S[key] <= 1.0:
            errors.append(f"synapses.{key} must be in [0, 1]")
    if P.n_L1_sensory > P.n_L1:
        errors.append("populations.n_L1_sensory cannot exceed n_L1")

    g = cam.grouping.group_size
    H, W = cam.sensor.height, cam.sensor.width
    if H % g or W % g:
        errors.append(f"sensor {H}x{W} is not divisible by group_size {g}")
    else:
        n_groups = (H // g) * (W // g)
        capacity = P.n_L1_sensory * S.syn_per_neuron
        if n_groups > capacity:
            errors.append(
                f"{n_groups} pixel groups exceed the capacity of the sensory L1 neurons "
                f"({P.n_L1_sensory} x {S.syn_per_neuron} = {capacity}); "
                f"use a larger group_size or more sensory neurons")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    cfg.time["dt_ms"] = cfg.time.bin_ms / cfg.time.substeps_per_bin
    return cfg
