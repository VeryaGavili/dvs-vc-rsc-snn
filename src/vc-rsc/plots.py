"""Figures 01-11. Each function writes one PNG into the output folder."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finish(cfg, fig, out, name):
    fig.tight_layout()
    if cfg.output.save_figures:
        fig.savefig(Path(out) / name, dpi=150, bbox_inches="tight")
    if cfg.output.show_figures:
        plt.show()
    else:
        plt.close(fig)


def plot_group_raster(cfg, pooled, out):
    dt, g = cfg.time.dt_ms, cfg.camera.grouping.group_size
    t_idx, g_idx = np.nonzero(pooled > 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(t_idx * dt, g_idx, s=1, marker="|")
    ax.set(xlabel="Time (ms)", ylabel="Pixel-group index", title=f"Input raster: {g}x{g} groups with events")
    _finish(cfg, fig, out, "01_input_group_raster.png")


def plot_layer_rasters(cfg, res, net, out):
    S = res["S"]
    t_ms = np.arange(S["L1"].shape[0]) * cfg.time.dt_ms
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    ti, ni = np.nonzero(S["L1"])
    axes[0].scatter(t_ms[ti], ni, s=3, marker="|", c=np.where(net["is_sens"][ni], "black", "red"))
    axes[0].set_title("L1 (black = sensory, red = RSC feedback only)")

    for ax, name in [(axes[1], "L2"), (axes[2], "L3")]:
        ti, ni = np.nonzero(S[name])
        ax.scatter(t_ms[ti], ni, s=3, marker="|", c="black")
        ax.set_title(name)

    ti, ni = np.nonzero(S["RSC"])
    axes[3].scatter(t_ms[ti], ni, s=3, marker="|", c=np.where(net["rsc_inh"][ni], "red", "blue"))
    axes[3].set_title("RSC (blue = excitatory, red = inhibitory)")

    for ax in axes:
        ax.set_ylabel("Neuron index")
    axes[-1].set_xlabel("Time (ms)")
    _finish(cfg, fig, out, "02_spike_rasters_all_layers.png")


def plot_rate_histograms(cfg, res, out):
    dur = res["S"]["L1"].shape[0] * cfg.time.dt_ms / 1000.0
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, name in zip(axes, ["L1", "L2", "L3", "RSC"]):
        rates = res["S"][name].sum(0) / dur
        ax.hist(rates, bins=25)
        ax.set(title=f"{name} firing-rate distribution", xlabel="Rate (Hz)", ylabel="Neurons")
    _finish(cfg, fig, out, "03_firing_rate_histograms.png")


def plot_input_decomposition(cfg, res, out):
    C = res["C"]
    t_ms = np.arange(C["pix_L1"].shape[0]) * cfg.time.dt_ms
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t_ms, C["pix_L1"].mean(1), label="pixel-group input (L1)")
    axes[0].plot(t_ms, C["fb_L1"].mean(1), label="RSC feedback (L1)")
    axes[1].plot(t_ms, C["ff_L2"].mean(1), label="L1 feedforward (L2)")
    axes[1].plot(t_ms, C["fb_L2"].mean(1), label="RSC feedback (L2)")
    axes[2].plot(t_ms, C["ff_L3"].mean(1), label="L2 feedforward (L3)")
    axes[2].plot(t_ms, C["fb_L3"].mean(1), label="RSC feedback (L3)")
    axes[2].plot(t_ms, C["rsc_in"].mean(1), label="VC input (RSC)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_ylabel("Mean input")
        ax.grid(alpha=0.25)
    axes[0].set_title("Mean input components per layer")
    axes[-1].set_xlabel("Time (ms)")
    _finish(cfg, fig, out, "04_input_decomposition.png")


def plot_membrane_examples(cfg, res, net, out):
    V, TH, S = res["V"], res["TH"], res["S"]
    t_ms = np.arange(V["L1"].shape[0]) * cfg.time.dt_ms
    picks = [("L1 sensory", "L1", int(net["sens_idx"][0])), ("L2", "L2", 0),
             ("L3", "L3", 0), ("RSC excitatory", "RSC", int(np.flatnonzero(~net["rsc_inh"])[0]))]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, (label, pop, n) in zip(axes, picks):
        p = net["params"][pop]
        ax.plot(t_ms, V[pop][:, n], lw=0.9, label="membrane v (before reset)")
        ax.plot(t_ms, TH[pop][:, n], "k--", lw=0.8, label="effective threshold")
        ti = np.flatnonzero(S[pop][:, n])
        ax.scatter(t_ms[ti], TH[pop][ti, n], marker="v", s=40, c="red", label="spikes")
        ax.set_title(f"{label}, neuron {n}: group {p['group'][n]}, tau={p['tau'][n]:.1f} ms, "
                     f"threshold={p['threshold'][n]:.2f}, adapt gain={p['adapt_gain'][n]:.2f}")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    _finish(cfg, fig, out, "05_membrane_examples.png")


def plot_group_behaviour(cfg, res, net, out):
    dur = res["S"]["L1"].shape[0] * cfg.time.dt_ms / 1000.0
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    for row, pop in enumerate(["L1", "L2", "L3", "RSC"]):
        p = net["params"][pop]
        rates = res["S"][pop].sum(0) / dur
        groups = p["group"]
        G = int(groups.max()) + 1
        g_mean = np.array([rates[groups == g].mean() for g in range(G)])
        g_std = np.array([rates[groups == g].std() for g in range(G)])
        axes[row, 0].bar(np.arange(G), g_mean, yerr=g_std, capsize=3)
        axes[row, 0].set(title=f"{pop}: mean rate per group", xlabel="group", ylabel="Hz")
        axes[row, 1].scatter(p["threshold"], rates, c=groups, cmap="tab20", s=18)
        axes[row, 1].set(title=f"{pop}: threshold vs rate", xlabel="threshold", ylabel="Hz")
        axes[row, 2].scatter(p["tau"], rates, c=groups, cmap="tab20", s=18)
        axes[row, 2].set(title=f"{pop}: tau vs rate", xlabel="tau (ms)", ylabel="Hz")
    _finish(cfg, fig, out, "11_group_behaviour.png")


def plot_cuts(cfg, res, net, gh, gw, out):
    cut = res["cut"]
    steps = cfg.analysis.cut_steps
    dt, P = cfg.time.dt_ms, cfg.populations
    panels = len(steps)

    fig, axes = plt.subplots(3, panels, figsize=(5 * panels, 13), squeeze=False)
    for col, t in enumerate(steps):
        for row, key, label in [(0, "pix_raw", "real input (pixel groups)"),
                                (1, "pix_before", "after 3-D kernel, before gain"),
                                (2, "pix_after", "after 3-D kernel and gain")]:
            ax = axes[row, col]
            im = ax.imshow(cut[key][t].reshape(gh, gw), cmap="viridis")
            ax.set_title(f"step {t} ({t * dt:.2f} ms)\n{label}")
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Pixel group -> L1: real input, 3-D kernel, and gain")
    _finish(cfg, fig, out, "06_cut_pixel_to_L1.png")

    n1, n2 = P.n_L1, P.n_L2
    N = P.n_L1 + P.n_L2 + P.n_L3
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4.5), squeeze=False)
    src = net["rsc_src"]
    tgt = np.repeat(np.arange(P.n_RSC), cfg.synapses.syn_per_neuron).reshape(P.n_RSC, -1)
    for ax, t in zip(axes[0], steps):
        vals = cut["vc_rsc"][t]
        keep = vals > 0
        sc = ax.scatter(src[keep], tgt[keep], c=vals[keep], s=3, cmap="magma")
        ax.set(title=f"step {t}: {int(keep.sum())} active",
               xlabel=f"VC source (0-{n1 - 1} L1, {n1}-{n1 + n2 - 1} L2, {n1 + n2}-{N - 1} L3)",
               ylabel="RSC target")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("VC -> RSC synapse output")
    _finish(cfg, fig, out, "07_cut_VC_to_RSC.png")

    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4.5), squeeze=False)
    for ax, t in zip(axes[0], steps):
        xs, ys, vals = [], [], []
        for i, (s, v) in enumerate(zip(net["fb_all"], cut["rsc_vc"][t])):
            xs.append(s)
            ys.append(np.full(len(s), i))
            vals.append(v)
        xs, ys, vals = np.concatenate(xs), np.concatenate(ys), np.concatenate(vals)
        keep = vals != 0
        sc = ax.scatter(xs[keep], ys[keep], c=vals[keep], s=3, cmap="coolwarm")
        ax.set(title=f"step {t}: {int(keep.sum())} active", xlabel="RSC source", ylabel="VC target (all layers)")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("RSC -> VC feedback synapse output (red = excitatory, blue = inhibitory)")
    _finish(cfg, fig, out, "08_cut_RSC_to_VC.png")


def plot_rsc_tuning(cfg, net, out):
    j = cfg.analysis.tuning_example_neuron
    P = cfg.populations
    src = net["rsc_src"][j]
    pos = net["c_vc_all"][src]
    centre = net["c_rsc"][j]
    fig, ax = plt.subplots(figsize=(7, 6))
    for lo, hi, name, col in [(0, P.n_L1, "L1", "tab:blue"),
                              (P.n_L1, P.n_L1 + P.n_L2, "L2", "tab:green"),
                              (P.n_L1 + P.n_L2, P.n_L1 + P.n_L2 + P.n_L3, "L3", "tab:purple")]:
        m = (src >= lo) & (src < hi)
        ax.scatter(pos[m, 0], pos[m, 1], c=col, s=30, label=f"{name} sources")
    ax.scatter([centre[0]], [centre[1]], marker="x", c="red", s=120, label="RSC centre")
    ax.set_xlim(0, net["c_rsc"][:, 0].max() * 1.2)
    ax.invert_yaxis()
    ax.set(title=f"RSC neuron {j}: preference = {net['rsc_pref'][j]}", xlabel="x (pixel)", ylabel="y (pixel)")
    ax.legend(fontsize=8)
    _finish(cfg, fig, out, "09_rsc_tuning_example.png")


def plot_sweep(cfg, results, out):
    labels = [f"{f2:.1f},{f3:.1f}" for (f2, f3), _ in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(results))
    for k in ["L1", "L2", "L3", "RSC"]:
        ax.plot(xs, [r[k] for _, r in results], "o-", label=k)
    ax.set_xticks(xs, labels)
    ax.set(xlabel="RSC feedback share (L2, L3)", ylabel="Mean rate (Hz)", title="Sweep: rates vs feedback share")
    ax.legend()
    _finish(cfg, fig, out, "10_sweep_feedback_share.png")
