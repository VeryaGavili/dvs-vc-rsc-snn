# scripts

- `run_network.py` runs one recording end to end: load, pool, build, simulate, print statistics, save figures 01–11, and run the sweep.

Common calls:

```bash
python scripts/run_network.py                                  # Gen1 config, default recording path
python scripts/run_network.py --data data/raw/my.h5 --no-sweep # local file, no sweep
python scripts/run_network.py --network configs/networks/vc_rsc_v2.yaml --camera configs/cameras/gen1_240x304.yaml
```

Figures are saved, not shown, unless `--show` is given. Matplotlib uses the `Agg` backend by default, so it works over SSH.
