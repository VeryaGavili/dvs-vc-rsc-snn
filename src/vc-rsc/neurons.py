"""Neuron model (adaptive LIF) and per-group parameter sampling."""
import numpy as np


def neuron_constants(cfg):
    return {
        "dt": cfg.time.dt_ms,
        "tau_adapt": cfg.neuron.tau_adapt_ms,
        "refractory": cfg.neuron.refractory_ms,
        "reset": cfg.neuron.reset,
    }


def layer_groups(cfg, layer, n):
    sizes = cfg.groups.sizes[layer]
    if isinstance(sizes, int):
        sizes = [sizes] * (n // sizes) + ([n % sizes] if n % sizes else [])
    sizes = list(sizes)
    if sum(sizes) != n:
        raise ValueError(f"{layer}: group sizes sum to {sum(sizes)}, the layer has {n} neurons")
    return sizes


def spatial_groups(centres, sizes):
    """Compact groups: group g is the nearest free neurons to the first free neuron."""
    free = np.ones(len(centres), bool)
    group_id = np.full(len(centres), -1)
    for g, size in enumerate(sizes):
        seed = np.flatnonzero(free)[0]
        d2 = ((centres - centres[seed]) ** 2).sum(1)
        d2[~free] = np.inf
        idx = np.argsort(d2)[:size]
        group_id[idx] = g
        free[idx] = False
    return group_id


def build_params(cfg, layer, n, rng, centres):
    """Sample tau, threshold and adapt_gain for each neuron inside its group's range."""
    sizes = layer_groups(cfg, layer, n)
    G = len(sizes)
    group_id = spatial_groups(centres, sizes)
    out = {"group": group_id}
    for name, ranges in cfg.groups.param_ranges[layer].items():
        if isinstance(ranges[0], (int, float)):          # one range for every group
            ranges = [ranges] * G
        if len(ranges) != G:
            raise ValueError(f"{layer} {name}: {len(ranges)} ranges for {G} groups")
        vals = np.empty(n)
        for g, (lo, hi) in enumerate(ranges):
            idx = group_id == g
            vals[idx] = rng.uniform(lo, hi, idx.sum())
        out[name] = vals
    return out


def new_state(n):
    return {"v": np.zeros(n, np.float32), "refr": np.zeros(n), "adapt": np.zeros(n, np.float32)}


def lif_update(st, I, p, nc):
    """One adaptive-LIF step for a population.

    Returns (spikes uint8, membrane before reset, effective threshold).
    """
    dt = nc["dt"]
    active = st["refr"] <= 0
    v = st["v"]
    v[active] += (dt / p["tau"][active]) * (-v[active] + I[active])
    v[~active] = nc["reset"]
    st["refr"] = np.maximum(0.0, st["refr"] - dt)
    th = p["threshold"] + p["adapt_gain"] * st["adapt"]
    v_pre = v.copy()
    spk = (v >= th) & active
    v[spk] = nc["reset"]
    st["refr"][spk] = nc["refractory"]
    st["adapt"] *= np.exp(-dt / nc["tau_adapt"])
    st["adapt"][spk] += 1.0
    return spk.astype(np.uint8), v_pre, th.copy()
