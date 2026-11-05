"""Printed summaries: synapse budget, firing rates, group behaviour."""
import numpy as np

POPS = ("L1", "L2", "L3", "RSC")


def mean_rates(res, duration_s):
    """Mean rate (Hz) per population over all its neurons."""
    return {k: res["S"][k].sum() / (res["S"][k].shape[1] * duration_s) for k in POPS}


def print_stats(cfg, res, net, pooled):
    P, S_cfg = cfg.populations, cfg.synapses
    dur = pooled.shape[0] * cfg.time.dt_ms / 1000.0
    S = res["S"]
    sens = np.flatnonzero(net["is_sens"])

    print("\nSYNAPSE BUDGET (average per neuron)")
    print(f"L1 sensory ({len(sens)}): {np.mean([len(g) for g in net['pix_groups']]):.1f} pixel-group "
          f"+ {np.mean([len(net['l1_fb'][n]) for n in sens]):.1f} RSC")
    print(f"L1 feedback-only ({P.n_L1 - len(sens)}): 0 pixel + {S_cfg.syn_per_neuron} RSC")
    print(f"L2 ({P.n_L2}): {len(net['l2_ff'][0])} from L1 + {len(net['l2_fb'][0])} from RSC")
    print(f"L3 ({P.n_L3}): {len(net['l3_ff'][0])} from L2 + {len(net['l3_fb'][0])} from RSC")
    print(f"RSC ({P.n_RSC}): {S_cfg.syn_per_neuron} from VC {dict(S_cfg.rsc_fanin)}")
    print(f"Pixel groups assigned: {sum(len(g) for g in net['pix_groups'])} / {pooled.shape[1]}")

    print("\nFIRING STATISTICS")
    for name in POPS:
        rates = S[name].sum(0) / dur
        print(f"{name}: spikes={int(S[name].sum())}, mean={rates.mean():.2f} Hz, "
              f"active={(rates > 0).sum()}/{len(rates)}, max={rates.max():.2f} Hz")
    r1 = S["L1"].sum(0) / dur
    print(f"L1 sensory mean={r1[net['is_sens']].mean():.2f} Hz | "
          f"feedback-only mean={r1[~net['is_sens']].mean():.2f} Hz")
    rr = S["RSC"].sum(0) / dur
    print(f"RSC excitatory mean={rr[~net['rsc_inh']].mean():.2f} Hz | "
          f"inhibitory mean={rr[net['rsc_inh']].mean():.2f} Hz")


def print_group_summary(res, net, cfg):
    dur = res["S"]["L1"].shape[0] * cfg.time.dt_ms / 1000.0
    for pop in POPS:
        groups = net["params"][pop]["group"]
        rates = res["S"][pop].sum(0) / dur
        G = int(groups.max()) + 1
        means = [rates[groups == g].mean() for g in range(G)]
        print(f"{pop}: {G} groups | " + ", ".join(
            f"g{g} n={int((groups == g).sum())} {means[g]:.1f}Hz" for g in range(G)))
