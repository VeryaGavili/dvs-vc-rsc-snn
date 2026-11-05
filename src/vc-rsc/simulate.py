"""Time-stepped simulation of the VC-RSC network.

Order inside one sub-step t:
  L1 uses pixel input and RSC spikes from t-1;
  L2 and L3 use the spikes of the layer below at t, and RSC spikes from t-1;
  RSC uses the spikes of all VC layers at t.
The RSC spikes of step t become the feedback for step t+1.
"""
import numpy as np

from .kernels import temporal_kernel
from .neurons import lif_update, neuron_constants, new_state


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def simulate(cfg, pooled, net):
    P, GN = cfg.populations, cfg.gains
    nc = neuron_constants(cfg)
    T, G = pooled.shape
    kern = temporal_kernel(cfg)
    hist = np.zeros((len(kern), G), np.float32)

    N = {"L1": P.n_L1, "L2": P.n_L2, "L3": P.n_L3, "RSC": P.n_RSC}
    st = {k: new_state(n) for k, n in N.items()}
    S = {k: np.zeros((T, n), np.uint8) for k, n in N.items()}
    V = {k: np.zeros((T, n), np.float32) for k, n in N.items()}
    TH = {k: np.zeros((T, n), np.float32) for k, n in N.items()}
    C = {"pix_L1": np.zeros((T, P.n_L1)), "fb_L1": np.zeros((T, P.n_L1)),
         "ff_L2": np.zeros((T, P.n_L2)), "fb_L2": np.zeros((T, P.n_L2)),
         "ff_L3": np.zeros((T, P.n_L3)), "fb_L3": np.zeros((T, P.n_L3)),
         "rsc_in": np.zeros((T, P.n_RSC))}
    cut = {"pix_raw": {}, "pix_before": {}, "pix_after": {}, "vc_rsc": {}, "rsc_vc": {}}
    cut_steps = set(cfg.analysis.cut_steps)

    prev_rsc = np.zeros(P.n_RSC, np.float32)
    rsc_sign = net["rsc_sign"]
    sens_idx = net["sens_idx"]
    syn = cfg.synapses.syn_per_neuron

    for t in range(T):
        hist[1:] = hist[:-1]
        hist[0] = pooled[t]
        temporal = kern @ hist

        # ---- L1: pixel groups (3-D kernel output) + RSC feedback ----
        pix_syn = np.zeros(G, np.float32)
        pix_after = np.zeros(G, np.float32)
        pix_cur = np.zeros(P.n_L1, np.float32)
        for i, (grp, w) in enumerate(zip(net["pix_groups"], net["pix_w"])):
            out = temporal[grp] * w
            pix_cur[sens_idx[i]] = out.mean()
            pix_syn[grp] = out
            pix_after[grp] = GN.pixel_L1 * out / max(len(grp), 1)
        fb1 = np.array([_mean(rsc_sign[s] * prev_rsc[s]) for s in net["l1_fb"]], np.float32)
        cur1 = GN.pixel_L1 * pix_cur + GN.rsc_to_vc * fb1
        S["L1"][t], V["L1"][t], TH["L1"][t] = lif_update(st["L1"], cur1, net["params"]["L1"], nc)
        s1 = S["L1"][t].astype(np.float32)

        # ---- L2: feedforward from L1 + feedback from RSC ----
        ff2 = np.array([_mean(s1[s]) for s in net["l2_ff"]], np.float32)
        fb2 = np.array([_mean(rsc_sign[s] * prev_rsc[s]) for s in net["l2_fb"]], np.float32)
        cur2 = GN.feedforward_vc * ff2 + GN.rsc_to_vc * fb2
        S["L2"][t], V["L2"][t], TH["L2"][t] = lif_update(st["L2"], cur2, net["params"]["L2"], nc)
        s2 = S["L2"][t].astype(np.float32)

        # ---- L3: feedforward from L2 + feedback from RSC ----
        ff3 = np.array([_mean(s2[s]) for s in net["l3_ff"]], np.float32)
        fb3 = np.array([_mean(rsc_sign[s] * prev_rsc[s]) for s in net["l3_fb"]], np.float32)
        cur3 = GN.feedforward_vc * ff3 + GN.rsc_to_vc * fb3
        S["L3"][t], V["L3"][t], TH["L3"][t] = lif_update(st["L3"], cur3, net["params"]["L3"], nc)
        s3 = S["L3"][t].astype(np.float32)

        # ---- RSC: input from VC spikes at step t ----
        vc_all = np.concatenate([s1, s2, s3])
        rsc_in = GN.vc_to_rsc * vc_all[net["rsc_src"]].mean(axis=1)
        S["RSC"][t], V["RSC"][t], TH["RSC"][t] = lif_update(st["RSC"], rsc_in, net["params"]["RSC"], nc)

        C["pix_L1"][t], C["fb_L1"][t] = pix_cur, fb1
        C["ff_L2"][t], C["fb_L2"][t] = ff2, fb2
        C["ff_L3"][t], C["fb_L3"][t] = ff3, fb3
        C["rsc_in"][t] = rsc_in

        if t in cut_steps:
            cut["pix_raw"][t] = pooled[t].copy()
            cut["pix_before"][t] = pix_syn.copy()
            cut["pix_after"][t] = pix_after.copy()
            cut["vc_rsc"][t] = GN.vc_to_rsc * vc_all[net["rsc_src"]] / syn
            cut["rsc_vc"][t] = [GN.rsc_to_vc * rsc_sign[s] * prev_rsc[s] / max(len(s), 1)
                                for s in net["fb_all"]]

        prev_rsc = S["RSC"][t].astype(np.float32)

    return {"S": S, "V": V, "TH": TH, "C": C, "cut": cut}
