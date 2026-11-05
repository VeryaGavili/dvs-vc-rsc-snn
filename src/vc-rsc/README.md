# src/vc_rsc

The library. Each module has one job, and every function takes the `cfg` object instead of reading globals.

| Module | Contents |
|---|---|
| `config.py` | `load_config()` merges network and camera YAML; `validate()` checks sizes, budgets and capacity; derives `dt_ms` |
| `data.py` | `load_sample()` reads event-cube bins; `make_group_input()` builds the pooled sub-step × group array |
| `geometry.py` | Group centres, near-square grid placement, capacity-limited assignment of groups to neurons |
| `kernels.py` | Temporal Gaussian, RSC tuning templates, distance-based source sampling |
| `connectivity.py` | `build_network()`: pixel groups, all layer-to-layer sources, RSC E/I and preferences, neuron parameters |
| `neurons.py` | Group layout, per-neuron parameter sampling, adaptive-LIF update |
| `simulate.py` | `simulate()`: the time loop and the order of updates within one sub-step |
| `stats.py` | Synapse budget, firing rates, per-group summary |
| `plots.py` | Figures 01–11 |
| `sweep.py` | `run_sweep()`: rebuilds and reruns the network for each feedback pair |

Design choices:

- `build_network()` draws connectivity from `cfg.random_seed`. Neuron parameters use `cfg.random_seed + 1`, so changing parameters does not change the wiring.
- The sweep changes only the two feedback shares. Everything else is fixed by the config.
- Input gains are applied as means over each neuron's synapses, so drive depends on the fraction of active inputs, not their count.
