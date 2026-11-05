"""Connectivity: which sources each neuron reads, and with which spatial weights."""
import numpy as np

from .geometry import assign_points_capacity, group_centres, regular_grid_centres
from .kernels import sample_sources, tuning_template
from .neurons import build_params


def build_network(cfg, gh, gw, rng, frac_l2, frac_l3):
    """Build all sources, weights, E/I signs, preferences and neuron parameters.

    frac_l2 and frac_l3 are the RSC feedback shares of L2 and L3. Everything except those two
    shares is taken from cfg, so sweep runs differ only in the shares.
    """
    P, S, K = cfg.populations, cfg.synapses, cfg.kernels
    g = cfg.camera.grouping.group_size
    H, W = gh * g, gw * g
    sig = K.sigma_inter_pop_pix

    gpts = group_centres(gh, gw, g)
    c_l1 = regular_grid_centres(P.n_L1, H, W)
    c_l2 = regular_grid_centres(P.n_L2, H, W)
    c_l3 = regular_grid_centres(P.n_L3, H, W)
    c_rsc = regular_grid_centres(P.n_RSC, H, W)

    # Sensory L1 neurons are spread evenly over the L1 list, so they cover the whole image.
    sens_idx = np.round(np.linspace(0, P.n_L1 - 1, P.n_L1_sensory)).astype(int)
    assert len(np.unique(sens_idx)) == P.n_L1_sensory
    is_sens = np.zeros(P.n_L1, bool)
    is_sens[sens_idx] = True

    # Pixel groups -> sensory L1 neurons (capacity-limited nearest assignment).
    owner = assign_points_capacity(gpts, c_l1[sens_idx], S.syn_per_neuron, k=S.capacity_candidates)
    pix_groups, pix_w = [], []
    for i, n in enumerate(sens_idx):
        grp = np.flatnonzero(owner == i)
        w = np.exp(-((gpts[grp] - c_l1[n]) ** 2).sum(1) / (2 * K.sigma_space_pix ** 2)).astype(np.float32)
        pix_groups.append(grp)
        pix_w.append(w)

    # RSC: excitatory/inhibitory identity and tuning preference.
    n_inh = int(round(P.n_RSC * P.rsc_inhibitory_fraction))
    rsc_inh = np.zeros(P.n_RSC, bool)
    rsc_inh[rng.choice(P.n_RSC, n_inh, replace=False)] = True
    rsc_sign = np.where(rsc_inh, -1.0, 1.0)
    prefs = cfg.templates.preferences
    rsc_pref = [prefs[j % len(prefs)] for j in range(P.n_RSC)]

    # RSC -> L1 feedback: sensory neurons fill their remaining slots, feedback-only neurons get all.
    pos_of = {n: i for i, n in enumerate(sens_idx)}
    l1_fb = []
    for n in range(P.n_L1):
        k = S.syn_per_neuron - len(pix_groups[pos_of[n]]) if is_sens[n] else S.syn_per_neuron
        l1_fb.append(sample_sources(c_l1[n], c_rsc, k, sig, rng))

    # L1 -> L2, L2 -> L3 feedforward; RSC -> L2, RSC -> L3 feedback.
    k_fb2 = int(round(S.syn_per_neuron * frac_l2))
    k_ff2 = S.syn_per_neuron - k_fb2
    k_fb3 = int(round(S.syn_per_neuron * frac_l3))
    k_ff3 = S.syn_per_neuron - k_fb3
    l2_ff = [sample_sources(c_l2[m], c_l1, k_ff2, sig, rng) for m in range(P.n_L2)]
    l2_fb = [sample_sources(c_l2[m], c_rsc, k_fb2, sig, rng) for m in range(P.n_L2)]
    l3_ff = [sample_sources(c_l3[m], c_l2, k_ff3, sig, rng) for m in range(P.n_L3)]
    l3_fb = [sample_sources(c_l3[m], c_rsc, k_fb3, sig, rng) for m in range(P.n_L3)]

    # VC -> RSC: each RSC neuron samples from L1, L2 and L3, weighted by its preference template.
    c_vc_all = np.vstack([c_l1, c_l2, c_l3])
    bases = {"L1": 0, "L2": P.n_L1, "L3": P.n_L1 + P.n_L2}
    layer_cent = {"L1": c_l1, "L2": c_l2, "L3": c_l3}
    fanin = S.rsc_fanin
    assert sum(fanin.values()) == S.syn_per_neuron
    rsc_src = np.zeros((P.n_RSC, S.syn_per_neuron), np.int64)
    for j in range(P.n_RSC):
        pref = rsc_pref[j]
        parts = []
        for name in ["L1", "L2", "L3"]:
            parts.append(bases[name] + sample_sources(
                c_rsc[j], layer_cent[name], fanin[name], sig, rng,
                template=lambda d, p=pref: tuning_template(d, p, cfg)))
        rsc_src[j] = np.concatenate(parts)

    # Neuron parameters use their own generator, so changing them does not change connectivity.
    rng_p = np.random.default_rng(cfg.random_seed + 1)
    params = {"L1": build_params(cfg, "L1", P.n_L1, rng_p, c_l1),
              "L2": build_params(cfg, "L2", P.n_L2, rng_p, c_l2),
              "L3": build_params(cfg, "L3", P.n_L3, rng_p, c_l3),
              "RSC": build_params(cfg, "RSC", P.n_RSC, rng_p, c_rsc)}

    return {
        "params": params,
        "sens_idx": sens_idx, "is_sens": is_sens,
        "pix_groups": pix_groups, "pix_w": pix_w,
        "l1_fb": l1_fb, "fb_all": l1_fb + l2_fb + l3_fb,
        "l2_ff": l2_ff, "l2_fb": l2_fb, "l3_ff": l3_ff, "l3_fb": l3_fb,
        "rsc_src": rsc_src, "rsc_sign": rsc_sign, "rsc_inh": rsc_inh, "rsc_pref": rsc_pref,
        "c_l1": c_l1, "c_l2": c_l2, "c_l3": c_l3, "c_rsc": c_rsc, "c_vc_all": c_vc_all,
    }
