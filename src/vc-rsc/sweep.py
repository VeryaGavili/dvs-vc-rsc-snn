"""Feedback-share sweep: rebuild and rerun the network for each (L2, L3) pair."""
import numpy as np

from .connectivity import build_network
from .plots import plot_sweep
from .simulate import simulate
from .stats import POPS, mean_rates


def run_sweep(cfg, pooled, gh, gw, out, main_rates):
    dur = pooled.shape[0] * cfg.time.dt_ms / 1000.0
    results = []
    for f2, f3 in [tuple(p) for p in cfg.analysis.sweep_fractions]:
        rng = np.random.default_rng(cfg.random_seed)
        net = build_network(cfg, gh, gw, rng, f2, f3)
        res = simulate(cfg, pooled, net)
        results.append(((f2, f3), mean_rates(res, dur)))

    print("\nSWEEP: RSC feedback share (L2, L3)")
    print(f"{'L2,L3':>10} {'L1 Hz':>8} {'L2 Hz':>8} {'L3 Hz':>8} {'RSC Hz':>8}")
    for (f2, f3), r in results:
        print(f"{f2:>4.2f},{f3:<4.2f} {r['L1']:>8.2f} {r['L2']:>8.2f} {r['L3']:>8.2f} {r['RSC']:>8.2f}")

    main_pair = (cfg.synapses.rsc_fraction_L2, cfg.synapses.rsc_fraction_L3)
    match = [r for pair, r in results if pair == main_pair]
    if match:
        same = all(np.isclose(main_rates[k], match[0][k]) for k in POPS)
        print(f"CHECK main run {main_pair} vs sweep row: {'MATCH' if same else 'MISMATCH'}")
    else:
        print(f"CHECK: main run pair {main_pair} is not in analysis.sweep_fractions")

    plot_sweep(cfg, results, out)
    return results
