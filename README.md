# Veryakilaneh

A spiking VC–RSC network driven by event-camera (DVS) input. Input is pooled into pixel groups, passed through spatiotemporal Gaussian receptive fields into a three-layer visual cortex model (VC), and fed back from a recurrent-spiking-circuit population (RSC) with tuning templates.

Everything is configured in YAML. The Python code is generic: change a file under `configs/`, not the code.

## Status

| Component | Status |
|---|---|
| Gen1 240×304 event-cube input | Implemented and runnable |
| VC L1/L2/L3 + RSC network (`vc_rsc_v2`) | Implemented, ported from `vc_rsc_v2.py` |
| Spatiotemporal receptive fields, tuning templates | Implemented |
| Per-group neuron parameters (τ, θ, adaptation gain) | Implemented |
| Feedback-share sweep and consistency check | Implemented |
| GenX320 320×320 (EVT2.1 events) | Config written; loader and event-to-cube converter not implemented |
| PV / SST / PYR / VIP neuron types | Planned, not implemented |
| STDP learning (no backprop) | Planned, not implemented |
| Reservoir readout (mean/max of selected neurons) | Planned, not implemented |
| FPGA / ASIC mapping | Out of scope for this repository |

Only the first five rows are claimed by the code. Nothing in the planned rows exists yet.

## Repository layout

```
Veryakilaneh/
├── configs/
│   ├── cameras/        sensor size, input file, grouping   (one YAML per camera)
│   └── networks/       populations, synapses, kernels, neurons, gains, analysis
├── src/vc_rsc/         library: config, data, kernels, connectivity, neurons, simulate, stats, plots, sweep
├── scripts/            command-line entry points
├── tests/              configuration checks and simulation smoke tests
├── data/               place recordings here (not tracked by git)
└── results/            generated figures and printed statistics (not tracked by git)
```

Each folder has its own README.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests -q                                       # configuration and simulation checks
python scripts/run_network.py --data /path/to/recording.h5
```

Outputs go to `results/vc_rsc_v2/`: figures `01`–`11` and printed statistics.

Useful flags: `--no-sweep` skips the feedback-share sweep, `--show` opens windows, `--out` changes the folder.

## Configuration

The network YAML and the camera YAML are merged at load time. `validate()` checks them before any simulation starts.

| Key | Meaning |
|---|---|
| `populations.*` | Layer sizes (L1 total and sensory count, L2, L3, RSC), RSC inhibitory fraction |
| `synapses.syn_per_neuron` | Synapses per neuron, all sources together (75) |
| `synapses.rsc_fraction_L2/L3` | Feedback share of L2 and L3 (swept in `analysis`) |
| `synapses.rsc_fanin` | RSC sources per VC layer (must sum to `syn_per_neuron`) |
| `kernels.*` | Spatial σ, inter-population σ, temporal σ and history length |
| `templates.*` | RSC preferences and template widths |
| `time.*` | Bin length and sub-steps per bin; dt is derived |
| `neuron.*` | Adaptation time constant, refractory period, reset value |
| `groups.sizes` | Neuron group sizes per layer (must sum to the layer size) |
| `groups.param_ranges` | τ, threshold and adaptation-gain ranges, one range or one per group |
| `gains.*` | Input gains for pixel→L1, feedforward VC, RSC→VC, VC→RSC |
| `analysis.*` | Cut steps, tuning-example neuron, sweep pairs |
| `output.*` | Output folder and figure options |

Camera files set `sensor.height/width`, `input.*` (path, bins, channels) and `grouping.group_size`.

## Method in brief

1. **Input.** Each 50 ms bin is split into three sub-steps. Each sub-step is the mean of ON and OFF events, averaged over 2×2 pixel groups.
2. **Temporal kernel.** A Gaussian over the last 8 sub-steps, σ = 6 sub-steps.
3. **Spatial kernel.** A Gaussian over the distance from a group to its L1 neuron, σ = 6 px.
4. **Connectivity.** Inter-population sampling uses a Gaussian of σ = 20 px. RSC neurons multiply it by a tuning template (edge at 0/45/90/135°, ring, or rotation).
5. **Neurons.** Adaptive LIF: leaky integration with τ per neuron, reset to zero on spike, threshold rising with an adaptation variable.
6. **Feedback.** RSC spikes at step t−1 feed back into VC. VC spikes at step t feed the RSC.
7. **Sweep.** Five (L2, L3) feedback shares. The main run must match its sweep row.

The derivation and equations are in the paper sections, not in this repository.

## Known limitations

- **Gains are not tuned.** On the synthetic test recording the network produces no spikes. The strongest L1 membrane reaches about 0.2 against a threshold near 1.7, so the pixel gain needs to rise roughly 8–15×. Tune on real data before reporting results.
- **Membrane time constant.** With dt = 16.7 ms and τ ≈ 15 ms, dt/τ ≈ 1.1, so the membrane is almost memoryless within one step. Check whether this is intended.
- **Refractory period.** 0.1 ms is shorter than one step, so a spiking neuron skips one integration step.
- **Rounding of feedback counts.** `round(75 × share)` uses Python's round-half-to-even. For L3 at 0.3, 22.5 becomes 22 RSC synapses, so the effective share is 0.293.
- **Event-cube input only.** GenX320 event streams need a converter first.

## Data

Recordings are not included. Put the Gen1 event-cube `.h5` file under `data/raw/` or pass `--data`. See `data/README.md`.

## License

MIT. See `LICENSE`.
