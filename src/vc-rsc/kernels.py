"""Kernels: temporal Gaussian, RSC tuning templates, and distance-based source sampling."""
import numpy as np


def temporal_kernel(cfg):
    """k_tau = exp(-tau^2 / (2 sigma^2)) for tau = 0 .. L-1 (sub-steps into the past)."""
    K = cfg.kernels
    tau = np.arange(K.temporal_history_steps)
    return np.exp(-(tau ** 2) / (2 * K.sigma_time_substeps ** 2)).astype(np.float32)


def tuning_template(offsets, pref, cfg):
    """Preference template T(d) in [0, 1] for offsets d = source - RSC centre."""
    T = cfg.templates
    x, y = offsets[:, 0], offsets[:, 1]
    if pref.startswith("edge"):
        th = np.deg2rad(float(pref[4:]))
        perp = -x * np.sin(th) + y * np.cos(th)
        return np.exp(-perp ** 2 / (2 * T.edge_width_pix ** 2))
    r = np.sqrt(x ** 2 + y ** 2)
    if pref == "circle":
        return np.exp(-(r - T.circle_radius_pix) ** 2 / (2 * T.circle_width_pix ** 2))
    phase = 0.0 if pref == "rot_cw" else np.pi
    return 0.5 * (1 + np.sin(np.arctan2(y, x) - phase))


def sample_sources(target_xy, source_xy, k, sigma, rng, template=None):
    """Draw k distinct sources with probability exp(-d^2 / 2 sigma^2), times the template if given."""
    if k <= 0:
        return np.empty(0, np.int64)
    d = (source_xy - target_xy).astype(np.float64)
    d2 = (d ** 2).sum(1)
    p = np.exp(-(d2 - d2.min()) / (2 * sigma ** 2))
    if template is not None:
        p = p * (template(d) + 1e-3)
    p = p / p.sum()
    return rng.choice(len(source_xy), size=min(k, len(source_xy)), replace=False, p=p)
