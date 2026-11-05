#!/usr/bin/env python3
"""Run the VC-RSC network on one recording: statistics, figures 01-11, and the feedback sweep.

Examples
  python scripts/run_network.py
  python scripts/run_network.py --data /path/to/recording.h5 --no-sweep
  python scripts/run_network.py --camera configs/cameras/genx320_320x320.yaml   # not runnable yet
"""
import argparse
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--network", default=ROOT / "configs/networks/vc_rsc_v2.yaml")
parser.add_argument("--camera", default=ROOT / "configs/cameras/gen1_240x304.yaml")
parser.add_argument("--data", default=None, help="recording .h5 (overrides camera.input.path)")
parser.add_argument("--out", default=None, help="output folder (default: output.dir from the network config)")
parser.add_argument("--start-bin", type=int, default=None, help="first bin (overrides camera config)")
parser.add_argument("--end-bin", type=int, default=None, help="last bin, exclusive (overrides camera config)")
parser.add_argument("--no-sweep", action="store_true", help="skip the feedback-share sweep")
parser.add_argument("--show", action="store_true", help="open figure windows instead of only saving")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")

from vc_rsc import plots  # noqa: E402  (imported after the backend is set)
from vc_rsc.config import load_config  # noqa: E402
from vc_rsc.connectivity import build_network  # noqa: E402
from vc_rsc.data import load_sample, make_group_input  # noqa: E402
from vc_rsc.simulate import simulate  # noqa: E402
from vc_rsc.stats import mean_rates, print_group_summary, print_stats  # noqa: E402
from vc_rsc.sweep import run_sweep  # noqa: E402


def main():
    import numpy as np

    cfg = load_config(args.network, args.camera, args.data)
    if args.start_bin is not None:
        cfg.camera.input.start_bin = args.start_bin
    if args.end_bin is not None:
        cfg.camera.input.end_bin = args.end_bin
    if args.show:
        cfg.output["show_figures"] = True
    if args.no_sweep:
        cfg.analysis["run_sweep"] = False

    out = ROOT / (args.out or cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)
    sample = load_sample(cfg)
    pooled, gh, gw = make_group_input(sample, cfg.camera.grouping.group_size)
    print(f"Pooled input: {pooled.shape} (steps x groups, grid {gh}x{gw})")

    net = build_network(cfg, gh, gw, rng, cfg.synapses.rsc_fraction_L2, cfg.synapses.rsc_fraction_L3)
    res = simulate(cfg, pooled, net)
    print_stats(cfg, res, net, pooled)
    print_group_summary(res, net, cfg)

    dur = pooled.shape[0] * cfg.time.dt_ms / 1000.0
    main_rates = mean_rates(res, dur)

    plots.plot_group_raster(cfg, pooled, out)
    plots.plot_layer_rasters(cfg, res, net, out)
    plots.plot_rate_histograms(cfg, res, out)
    plots.plot_input_decomposition(cfg, res, out)
    plots.plot_membrane_examples(cfg, res, net, out)
    plots.plot_group_behaviour(cfg, res, net, out)
    plots.plot_cuts(cfg, res, net, gh, gw, out)
    plots.plot_rsc_tuning(cfg, net, out)

    if cfg.analysis.run_sweep:
        run_sweep(cfg, pooled, gh, gw, out, main_rates)

    print(f"\nFinished. Figures saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
