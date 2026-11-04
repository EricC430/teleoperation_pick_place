"""Procedural ground textures for the DR render (EricC 2026-11-05: more sidewalk / asphalt / brick patterns).

Pure numpy + cv2 (no Isaac), so the patterns can be looked at on any machine:
    python3 sim/dr_ground.py <out_dir>      writes one PNG per kind and a contact sheet

Every texture is a seamless tile of `tile_m` metres: dr_render.py gives the floor UVs in metres x (1 / tile_m), so one
texture repeat is tile_m on the ground and the joint widths, brick sizes etc. below are real sizes.
Colours are sRGB albedo, kept in the range of real surfaces (asphalt is dark; no saturated colours).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

KINDS = ("brick_running", "brick_basket", "paver_grid", "sidewalk_slabs", "asphalt", "tile_corridor")
BRICK_COLORS = ([0.40, 0.21, 0.16], [0.36, 0.24, 0.18], [0.46, 0.34, 0.26], [0.42, 0.41, 0.39], [0.50, 0.42, 0.32])
MORTAR = ([0.62, 0.61, 0.58], [0.45, 0.44, 0.42], [0.20, 0.19, 0.18])


def _blur(img, sigma):
    """gaussian blur that wraps around the edges (cv2 has no BORDER_WRAP for it): the texture has to tile."""
    pad = int(sigma * 4) + 1
    big = np.pad(img, pad, mode="wrap")
    return cv2.GaussianBlur(big, (0, 0), sigma)[pad:-pad, pad:-pad]


def _noise(rng, n, scale):
    """seamless low-frequency noise in [-1, 1] at n x n (gaussian-blurred white noise, wrapped)."""
    w = rng.normal(0, 1, (n, n)).astype(np.float32)
    w = _blur(w, scale / 3.0)
    return w / (np.abs(w).max() + 1e-6)


def _grain(rng, n, amp, coarse=0.0):
    g = rng.normal(0, amp, (n, n, 1)).astype(np.float32)
    if coarse:
        g = g + _blur(rng.normal(0, coarse, (n, n)).astype(np.float32), 3.0)[..., None]
    return g


def _grid(n, tile_m):
    c = (np.arange(n, dtype=np.float32) + 0.5) / n * tile_m
    return np.meshgrid(c, c)   # x (columns), y (rows)


def _joint_mask(lx, ly, w_m, h_m, joint_m):
    """True where a point of a w_m x h_m cell (local coords lx, ly) lies in the joint around the cell."""
    j = joint_m / 2.0
    return (lx < j) | (lx > w_m - j) | (ly < j) | (ly > h_m - j)


def make(rng, kind, n=2048):
    """-> (float image [n, n, 3] in 0..1 sRGB, params dict incl. tile_m, roughness)."""
    if kind == "brick_running":
        ncol, nrow = int(rng.choice([4, 5])), int(rng.choice([10, 12, 14]))   # nrow even: the half-brick offset wraps
        tile_m = ncol * float(rng.choice([0.20, 0.22, 0.24])); bw, bh = tile_m / ncol, tile_m / nrow   # ~ 22 x 7 cm bricks
        x, y = _grid(n, tile_m)
        r = np.floor(y / bh).astype(int); xs = (x + (r % 2) * bw / 2) % tile_m
        c = np.floor(xs / bw).astype(int); lx = xs - c * bw; ly = y - r * bh
        base = np.array(BRICK_COLORS[int(rng.integers(len(BRICK_COLORS)))]) * rng.uniform(0.9, 1.1)
        per = np.clip(base * (1 + rng.normal(0, 0.09, (nrow, ncol, 1))) + rng.normal(0, 0.01, (nrow, ncol, 3)), 0.05, 0.9)   # brightness varies, hue barely
        col = per[r % nrow, c % ncol]
        mortar = np.array(MORTAR[int(rng.integers(len(MORTAR)))]); joint = float(rng.uniform(0.008, 0.012))
        col = np.where(_joint_mask(lx, ly, bw, bh, joint)[..., None], mortar, col)
        col = col + _grain(rng, n, 0.025, 0.02) + 0.04 * _noise(rng, n, 60)[..., None]
        return np.clip(col, 0, 1), {"tile_m": tile_m, "roughness": 0.9, "brick_m": [round(bw, 3), round(bh, 3)]}
    if kind == "brick_basket":
        nb = 4; bwid = float(rng.choice([0.07, 0.08, 0.09])); blk = 2 * bwid; tile_m = nb * blk
        x, y = _grid(n, tile_m)
        i, j = np.floor(x / blk).astype(int), np.floor(y / blk).astype(int); lx, ly = x - i * blk, y - j * blk
        horiz = ((i + j) % 2 == 0)
        k = np.where(horiz, np.floor(ly / bwid), np.floor(lx / bwid)).astype(int)
        w_ = np.where(horiz, blk, bwid); h_ = np.where(horiz, bwid, blk)
        sx = np.where(horiz, lx, lx - k * bwid); sy = np.where(horiz, ly - k * bwid, ly)
        base = np.array(BRICK_COLORS[int(rng.integers(len(BRICK_COLORS)))]) * rng.uniform(0.9, 1.1)
        per = np.clip(base * (1 + rng.normal(0, 0.09, (nb, nb, 2, 1))) + rng.normal(0, 0.01, (nb, nb, 2, 3)), 0.05, 0.9)
        col = per[i % nb, j % nb, k % 2]
        mortar = np.array(MORTAR[int(rng.integers(len(MORTAR)))]); joint = float(rng.uniform(0.007, 0.011)); jh = joint / 2
        jm = (sx < jh) | (sx > w_ - jh) | (sy < jh) | (sy > h_ - jh)
        col = np.where(jm[..., None], mortar, col) + _grain(rng, n, 0.025, 0.02) + 0.04 * _noise(rng, n, 60)[..., None]
        return np.clip(col, 0, 1), {"tile_m": tile_m, "roughness": 0.9, "brick_m": [blk, bwid]}
    if kind == "paver_grid":
        npv = 4; p = float(rng.choice([0.3, 0.4, 0.5])); tile_m = npv * p
        x, y = _grid(n, tile_m); i, j = np.floor(x / p).astype(int), np.floor(y / p).astype(int)
        lx, ly = x - i * p, y - j * p
        g = rng.uniform(0.45, 0.68); warm = rng.uniform(-0.03, 0.04)
        per = np.clip((g + np.array([warm, 0, -warm])) * (1 + rng.normal(0, 0.07, (npv, npv, 1))), 0.2, 0.85)
        col = per[i % npv, j % npv]
        joint = float(rng.uniform(0.005, 0.009)); col = np.where(_joint_mask(lx, ly, p, p, joint)[..., None], np.array([0.30, 0.29, 0.28]), col)
        col = col + _grain(rng, n, 0.03, 0.02) + 0.02 * _noise(rng, n, 90)[..., None]
        return np.clip(col, 0, 1), {"tile_m": tile_m, "roughness": 0.88, "paver_m": p}
    if kind == "sidewalk_slabs":
        ns = 2; s_ = float(rng.choice([1.0, 1.2, 1.5])); tile_m = ns * s_
        x, y = _grid(n, tile_m); i, j = np.floor(x / s_).astype(int), np.floor(y / s_).astype(int)
        lx, ly = x - i * s_, y - j * s_
        g = rng.uniform(0.5, 0.7)
        per = np.clip(g * (1 + rng.normal(0, 0.05, (ns, ns, 1))), 0.25, 0.85) * np.ones(3)
        col = per[i % ns, j % ns]
        joint = float(rng.uniform(0.006, 0.011)); col = np.where(_joint_mask(lx, ly, s_, s_, joint)[..., None], np.array([0.30, 0.29, 0.28]), col)
        col = col + _grain(rng, n, 0.03, 0.025) + 0.035 * _noise(rng, n, 150)[..., None] + 0.02 * _noise(rng, n, 30)[..., None]
        return np.clip(col, 0, 1), {"tile_m": tile_m, "roughness": 0.92, "slab_m": s_}
    if kind == "asphalt":
        tile_m = 2.0; g = rng.uniform(0.13, 0.27)
        col = np.full((n, n, 3), g, np.float32) + np.array([0.0, 0.0, 0.01])
        col = col + _grain(rng, n, 0.035, 0.015) + 0.05 * _noise(rng, n, 120)[..., None]
        pts = (rng.random((n, n)) > 0.9965).astype(np.float32)   # light aggregate chips
        pts = _blur(pts, 1.3) * rng.uniform(5.0, 9.0)
        col = col + pts[..., None] * 0.35
        crack = np.zeros((n, n), np.float32)
        for _ in range(int(rng.integers(0, 4))):   # a few hairline cracks (random walks, wrapped)
            y0, x0 = rng.integers(0, n, 2); a = rng.uniform(0, 2 * np.pi)
            for _s in range(int(rng.integers(300, 900))):
                a += rng.normal(0, 0.25); x0 = (x0 + np.cos(a) * 2) % n; y0 = (y0 + np.sin(a) * 2) % n
                crack[int(y0) % n, int(x0) % n] = 1.0
        crack = _blur(crack, 1.0)
        col = col - np.clip(crack * 9.0, 0, 1)[..., None] * (g * 0.8)
        return np.clip(col, 0.02, 1), {"tile_m": tile_m, "roughness": 0.95, "base_grey": round(float(g), 3)}
    if kind == "tile_corridor":
        nt = 4; p = float(rng.choice([0.3, 0.4, 0.6])); tile_m = nt * p
        x, y = _grid(n, tile_m); i, j = np.floor(x / p).astype(int), np.floor(y / p).astype(int)
        lx, ly = x - i * p, y - j * p
        palette = ([0.62, 0.60, 0.56], [0.48, 0.48, 0.48], [0.66, 0.62, 0.54], [0.34, 0.34, 0.36], [0.52, 0.44, 0.38])
        c1 = np.array(palette[int(rng.integers(len(palette)))]); c2 = c1 * rng.uniform(0.78, 0.92)   # a checker is two close tones, not black and white
        checker = rng.random() < 0.4
        per = np.clip(np.where(((np.arange(nt)[:, None] + np.arange(nt)[None]) % 2 == 0)[..., None] & checker, c1, c2 if checker else c1)
                      * (1 + rng.normal(0, 0.05, (nt, nt, 1))), 0.1, 0.9)
        col = per[i % nt, j % nt]
        grout = np.array([0.30, 0.30, 0.29]) if rng.random() < 0.5 else np.array([0.62, 0.61, 0.58])
        joint = float(rng.uniform(0.004, 0.008)); col = np.where(_joint_mask(lx, ly, p, p, joint)[..., None], grout, col)
        col = col + _grain(rng, n, 0.012, 0.01) + 0.025 * _noise(rng, n, 100)[..., None]
        return np.clip(col, 0, 1), {"tile_m": tile_m, "roughness": float(rng.uniform(0.3, 0.6)), "tile_size_m": p}
    raise ValueError(kind)


def write(rng, kind, out_png, n=2048):
    img, params = make(rng, kind, n)
    cv2.imwrite(str(out_png), (img[..., ::-1] * 255).astype(np.uint8))
    return {"kind": kind, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()}}


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in KINDS:
        tiles = []
        for sd in range(3):
            rng = np.random.default_rng(100 + sd); png = out / f"{kind}_{sd}.png"; params = write(rng, kind, png, 1024)
            im = cv2.imread(str(png)); im = np.hstack([im, im]); im = np.vstack([im, im])   # 2x2 repeats: seam check
            tiles.append(cv2.resize(im, (360, 360)))
            print(params)
        row = np.hstack(tiles); cv2.putText(row, kind, (6, 20), 0, 0.6, (0, 255, 255), 2); rows.append(row)
    cv2.imwrite(str(out / "sheet.jpg"), np.vstack(rows))
