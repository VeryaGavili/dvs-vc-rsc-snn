# data

Recordings are not stored in the repository.

Put the Gen1 event-cube file in `data/raw/`:

```
data/raw/canrecording3_17-08-28_10-36-49_td_500000_60500000.h5
```

Or pass its location at run time, which overrides the camera config:

```bash
python scripts/run_network.py --data /path/to/recording.h5
```

The file must contain a dataset named `data` with shape (bins, 6, 240, 304), where channels 0/2/4 are ON and 1/3/5 are OFF. Values are uint8.
