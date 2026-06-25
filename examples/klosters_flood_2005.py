"""Reconstruct the August 2005 Landquart flood through Klosters (Prättigau, GR).

In August 2005 the Landquart produced its highest discharge on record — 391 m3/s at the
Felsenbach gauge (614 km2) — and devastated the Prättigau; Klosters was inundated with
water reportedly up to ~3 m deep, ~40 MCHF damage. There is no gauge at Klosters, so the
local peak is ESTIMATED by area-scaling the Felsenbach record:

    Q_klosters ~ 391 * (A_klosters / 614) ** 0.8

With A_klosters ~ 150 km2 (upper Prättigau above the Schlappinbach confluence) this gives
~130 m3/s; we route a ~150 m3/s peak as a plausible upper estimate for the steep, flashy
catchment. This is a scaled scenario, NOT a measured hydrograph.

Terrain: swisstopo SwissALTI3D (2 m -> 10 m), 4 x 3 km around Klosters Platz/Dorf. The
flood is injected at the upstream (south) channel and drains through the NW (west) outlet;
valley walls are solid. Produces a max-depth inundation map.

Prerequisite: data/klosters/dem10.npy + dem10.json (see examples/download_klosters_dem.py).
"""

import json
import os
import numpy as np
from hecras import run_swe2d

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
DEM_NPY = os.path.join(ROOT, "data", "klosters", "dem10.npy")
DEM_META = os.path.join(ROOT, "data", "klosters", "dem10.json")

MANNING_N = 0.05      # steep boulder/gravel mountain river + developed floodplain
Q_BASE = 6.0          # ~ mean flow scaled to the Klosters catchment (m3/s)
Q_PEAK = 150.0        # area-scaled estimate of the Aug 2005 peak at Klosters (m3/s)
T_END = 3600.0        # 60 min of routing (s) — long enough to reach the outlet

# Klosters landmarks (WGS84) for annotation / depth probes
PLATZ = (46.869, 9.879)
DORF = (46.887, 9.857)


def hillshade(z, res, az=315, alt=45):
    dzdy, dzdx = np.gradient(z, res, res)
    azr, altr = np.deg2rad(360 - az + 90), np.deg2rad(alt)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)
    return np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect)


def cell_of(lat, lon, meta):
    from pyproj import Transformer
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(lon, lat)
    return (meta["N0"] - n) / meta["res"], (e - meta["E0"]) / meta["res"]  # (row, col)


def main():
    if not os.path.exists(DEM_NPY):
        raise SystemExit(f"missing DEM: {DEM_NPY} (run examples/download_klosters_dem.py first)")
    dem = np.load(DEM_NPY)
    meta = json.load(open(DEM_META))
    res = meta["res"]
    ny, nx = dem.shape
    cell_area = res * res
    print(f"DEM {ny}x{nx} @ {res:.0f} m ({nx*res/1000:.0f}x{ny*res/1000:.0f} km), "
          f"elev {dem.min():.0f}-{dem.max():.0f} m")

    # Inflow at the south (upstream) edge. Isolate the channel *notch* (the contiguous
    # low band around the thalweg) rather than the whole flat terrace, so the flood
    # routes as a channel flow instead of ponding.
    rows = [ny - 4, ny - 3, ny - 2]
    south_profile = dem[ny - 3, :]
    c0 = int(np.argmin(south_profile))
    thr = south_profile.min() + 1.5
    lo = c0
    while lo > 0 and south_profile[lo - 1] < thr:
        lo -= 1
    hi = c0
    while hi < nx - 1 and south_profile[hi + 1] < thr:
        hi += 1
    chan_cols = list(range(lo, hi + 1))
    source_idx = [int(r) * nx + int(c) for r in rows for c in chan_cols]
    print(f"inflow: {len(source_idx)} cells, south-edge channel cols {lo}-{hi} "
          f"({(hi-lo+1)*res:.0f} m wide, bed ~{south_profile[chan_cols].mean():.0f} m)")

    # 2005-style hydrograph: rise to peak over ~12 min, hold (scenario, not measured).
    src_t = [0.0, 120.0, 720.0, T_END]
    src_q = [Q_BASE, Q_BASE, Q_PEAK, Q_PEAK]

    bed = dem.astype(float).ravel().tolist()
    zero = [0.0] * (nx * ny)

    print(f"routing Aug-2005 scenario: {Q_BASE:.0f} -> {Q_PEAK:.0f} m3/s, n={MANNING_N} ...")
    sim = run_swe2d(
        nx, ny, res, res, bed, zero, zero, zero,
        manning_n=MANNING_N, t_end=T_END, cfl=0.4,
        bc="reflective", bc_xlow="transmissive",   # solid walls; NW (west) outlet open
        max_dt=3.0, source_idx=source_idx, source_t=src_t, source_q=src_q,
        max_steps=4_000_000,
    )

    hmax = np.asarray(sim["h_max"])
    vol = np.asarray(sim["volume"])
    inflow_vol = np.asarray(sim["inflow_vol"])
    stored, injected = vol[-1], inflow_vol[-1]
    outflow = injected - stored
    flooded = int((hmax > 0.10).sum())

    def nbhd_depth(lat, lon, radius_cells=12):
        r, c = cell_of(lat, lon, meta)
        r, c = int(r), int(c)
        win = hmax[max(0, r - radius_cells):r + radius_cells,
                   max(0, c - radius_cells):c + radius_cells]
        return win.max() if win.size else float("nan")

    print(f"\nsteps={sim['steps']}  t_final={sim['t_final']:.0f}s")
    print(f"injected={injected:,.0f} m3  stored={stored:,.0f}  outflow={outflow:,.0f} "
          f"({100*outflow/injected:.0f}% of inflow)")
    print(f"max depth (domain)         = {hmax.max():.2f} m")
    print(f"max depth near Klosters Platz = {nbhd_depth(*PLATZ):.2f} m (within ~120 m)")
    print(f"max depth near Klosters Dorf  = {nbhd_depth(*DORF):.2f} m (within ~120 m)")
    print(f"flooded area (>0.1 m)      = {flooded*cell_area/1e4:.1f} ha")

    assert stored <= injected * 1.0001, "spurious mass created"
    assert hmax.max() < 50.0, "non-physical depth"
    print("\nPASS: mass budget closes; inundation physical")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        hs = hillshade(dem, res)
        depth = np.ma.masked_less(hmax, 0.10)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.imshow(hs, cmap="gray", origin="upper")
        im = ax.imshow(depth, cmap="turbo", origin="upper",
                       norm=Normalize(0, np.percentile(hmax[hmax > 0.1], 98)))
        plt.colorbar(im, ax=ax, label="max flood depth (m)", shrink=0.8)
        for name, (lat, lon) in {"Platz": PLATZ, "Dorf": DORF}.items():
            r, c = cell_of(lat, lon, meta)
            ax.plot(c, r, "w o", ms=7, mec="k")
            ax.annotate(f"Klosters {name}", (c, r), color="k",
                        fontsize=9, fontweight="bold", xytext=(5, 5), textcoords="offset points")
        ax.set_title("Landquart — August 2005 flood reconstruction at Klosters\n"
                     f"SwissALTI3D 10 m + hecras 2D shallow-water, peak ~{Q_PEAK:.0f} m³/s "
                     "(area-scaled estimate)")
        ax.set_xlabel("east →")
        ax.set_ylabel("← north")
        fig.tight_layout()
        out = os.path.join(ROOT, "docs", "klosters_flood_2005.png")
        fig.savefig(out, dpi=120)
        print(f"saved inundation map -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipping map)")


if __name__ == "__main__":
    main()
