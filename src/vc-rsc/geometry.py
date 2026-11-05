"""Positions on the sensor and capacity-limited assignment of pixel groups."""
import numpy as np


def group_centres(gh, gw, group_size):
    """Centre (x, y) in pixels of every pixel group, row-major."""
    yy, xx = np.mgrid[0:gh, 0:gw]
    off = (group_size - 1) / 2.0
    return np.column_stack([xx.ravel() * group_size + off,
                            yy.ravel() * group_size + off]).astype(np.float32)


def regular_grid_centres(n_neurons, height, width):
    """Near-square grid of n_neurons centres in (x, y) pixel coordinates."""
    pairs = [(r, n_neurons // r) for r in range(1, int(np.sqrt(n_neurons)) + 1) if n_neurons % r == 0]
    if pairs:
        rows, cols = min(pairs, key=lambda p: abs(np.log((height / p[0]) / (width / p[1]))))
    else:
        rows = max(1, int(round(np.sqrt(n_neurons * height / width))))
        cols = int(np.ceil(n_neurons / rows))
    ys = np.linspace(0, height - 1, rows)
    xs = np.linspace(0, width - 1, cols)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])[:n_neurons].astype(np.float32)


def assign_points_capacity(points, centres, capacity, k=12):
    """Give each point to its nearest centre that still has room (at most `capacity` points each).

    Candidates are the k nearest centres. Points that find no room among them go to the
    nearest centre with room. Returns the owner index of every point.
    """
    k = min(k, len(centres))
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
    cand_d = np.take_along_axis(d2, cand, axis=1)
    owner = np.full(len(points), -1, np.int64)
    load = np.zeros(len(centres), np.int64)
    for flat in np.argsort(cand_d, axis=None, kind="stable"):
        p, j = divmod(int(flat), k)
        nb = cand[p, j]
        if owner[p] < 0 and load[nb] < capacity:
            owner[p] = nb
            load[nb] += 1
    for p in np.flatnonzero(owner < 0):
        dd = d2[p].copy()
        dd[load >= capacity] = np.inf
        nb = int(np.argmin(dd))
        owner[p] = nb
        load[nb] += 1
    return owner
