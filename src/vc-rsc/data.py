"""Read event-cube recordings and turn them into pooled pixel-group input."""
import h5py
import numpy as np


def load_sample(cfg):
    """Return bins [start_bin, end_bin) of the recording as float32, shape (bins, 6, H, W)."""
    inp = cfg.camera.input
    if inp.format != "event_cube_h5":
        raise NotImplementedError(f"input format '{inp.format}' is not implemented yet")

    with h5py.File(inp.path, "r") as f:
        d = f["data"]
        if not (0 <= inp.start_bin < inp.end_bin <= d.shape[0]):
            raise ValueError(f"bins [{inp.start_bin}, {inp.end_bin}) outside 0..{d.shape[0] - 1}")
        sample = d[inp.start_bin:inp.end_bin].astype(np.float32)

    if inp.normalize_uint8:
        sample /= 255.0
    print("Selected shape:", sample.shape)
    for ch, name in enumerate(inp.channel_names):
        print(f"Channel {ch} ({name}): sum={sample[:, ch].sum():.2f}")
    return sample


def make_group_input(sample, group_size):
    """(bins, 6, H, W) -> (steps, groups), with steps = 3 * bins.

    Channels 0, 2, 4 are ON and 1, 3, 5 are OFF. Each sub-step value is the mean of ON and OFF,
    averaged over each group_size x group_size block.
    """
    nb, _, H, W = sample.shape
    on = sample[:, 0::2]
    off = sample[:, 1::2]
    frames = (0.5 * (on + off)).reshape(nb * 3, H, W)
    gh, gw = H // group_size, W // group_size
    pooled = frames.reshape(nb * 3, gh, group_size, gw, group_size).mean(axis=(2, 4))
    return pooled.reshape(nb * 3, gh * gw).astype(np.float32), gh, gw
