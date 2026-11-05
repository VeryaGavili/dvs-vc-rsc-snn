# tests

Run all tests from the repository root:

```bash
pytest tests -q
```

- `test_config.py` checks that both camera configs load, that the Gen1 dt is 50/3 ms, and that bad values are rejected: GenX320 with group size 2 (capacity), mismatched layer sizes, and fan-in that does not sum to 75.
- `test_simulate.py` checks the synapse budget per population, that random input runs and stays finite, and that silent input gives no spikes.

These tests do not use a recording. They run without a dataset.
