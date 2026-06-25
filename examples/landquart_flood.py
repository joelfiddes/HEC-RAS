"""Phase 4 application: 2D flood inundation of the Landquart at Felsenbach (GR).

Real terrain from swisstopo SwissALTI3D (2 m, block-averaged to 10 m) for a 3 x 2 km
reach around the BAFU gauge 2150 (Landquart - Felsenbach; MQ approx 24 m3/s, mean annual
flood approx 189 m3/s). A flood hydrograph is routed across the bare-earth DEM with the
2D shallow-water engine (`run_swe2d`), injecting discharge as a mass source at the
upstream (east) channel cells and letting it drain through the transmissive west edge.

Outputs a max-depth inundation map over a hillshade and reports a mass balance. This is
the methodology intended to transfer to Central Asian mountain catchments.

Prerequisite: run the tile download / mosaic step that produces
data/landquart/dem10.npy + dem10.json (see the project README / data notes).
"""

import json
import os
import numpy as np
from hecras import run_swe2d

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
DEM_NPY = os.path.join(ROOT, "data", "landquart", "dem10.npy")
DEM_META = os.path.join(ROOT, "data", "landquart", "dem10.json")

MANNING_N = 0.04      # natural mountain river / gravel floodplain
Q_BASE = 24.0         # mean discharge MQ (m3/s)
Q_PEAK = 190.0        # ~ mean annual flood HQ (m3/s)
T_END = 1800.0        # 30 min of routing (s)


def hillshade(z, res, az=315, alt=45):
    dzdy, dzdx = np.gradient(z, res, res)
    azr = np.deg2rad(360 - az + 90)
    altr = np.deg2rad(alt)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)
    return np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect)


def main():
    if not os.path.exists(DEM_NPY):
        raise SystemExit(f"missing DEM: {DEM_NPY} (run the SwissALTI3D download/mosaic first)")
    dem = np.load(DEM_NPY)
    meta = json.load(open(DEM_META))
    res = meta["res"]
    ny, nx = dem.shape
    cell_area = res * res
    print(f"DEM {ny}x{nx} @ {res:.0f} m  ({nx*res/1000:.1f} x {ny*res/1000:.1f} km), "
          f"elev {dem.min():.0f}-{dem.max():.0f} m")

    # Inflow: the channel enters from the east edge. Take a 3-cell-wide band a few
    # columns in from the boundary and keep the valley-floor cells (low elevation).
    cols = [nx - 4, nx - 3, nx - 2]
    east_profile = dem[:, nx - 3]
    chan_rows = np.where(east_profile < east_profile.min() + 2.0)[0]
    source_idx = [int(r) * nx + int(c) for c in cols for r in chan_rows]
    print(f"inflow: {len(source_idx)} cells over rows {chan_rows.min()}-{chan_rows.max()} "
          f"at east end (bed ~{east_profile[chan_rows].mean():.0f} m)")

    # Flood hydrograph: rise from base flow to peak over 10 min, then hold.
    src_t = [0.0, 60.0, 600.0, T_END]
    src_q = [Q_BASE, Q_BASE, Q_PEAK, Q_PEAK]

    bed = dem.astype(float).ravel().tolist()
    zero = [0.0] * (nx * ny)

    print(f"routing flood: {Q_BASE:.0f} -> {Q_PEAK:.0f} m3/s, n={MANNING_N}, t_end={T_END:.0f}s ...")
    res_sim = run_swe2d(
        nx, ny, res, res, bed, zero, zero, zero,
        manning_n=MANNING_N, t_end=T_END, cfl=0.4,
        bc="reflective",          # valley walls are solid...
        bc_xlow="transmissive",   # ...except the west edge, where the river drains out
        max_dt=3.0,               # cap dt so the dry-start inflow fills gradually
        source_idx=source_idx, source_t=src_t, source_q=src_q, max_steps=2_000_000,
    )

    hmax = np.asarray(res_sim["h_max"])
    h = np.asarray(res_sim["h"])
    vol = np.asarray(res_sim["volume"])
    inflow_vol = np.asarray(res_sim["inflow_vol"])

    # mass balance: injected = stored + outflow  (dry start => outflow = injected - stored)
    stored = vol[-1]
    injected = inflow_vol[-1]
    outflow = injected - stored
    flooded_cells = int((hmax > 0.10).sum())
    flooded_area = flooded_cells * cell_area
    print(f"\nsteps={res_sim['steps']}  t_final={res_sim['t_final']:.0f}s")
    print(f"injected volume   = {injected:,.0f} m3")
    print(f"stored (final)    = {stored:,.0f} m3")
    print(f"outflow (derived) = {outflow:,.0f} m3  ({100*outflow/injected:.1f}% of inflow)")
    print(f"max depth         = {hmax.max():.2f} m")
    print(f"flooded area (>0.1 m) = {flooded_area/1e4:.1f} ha  ({flooded_cells} cells)")

    # the mass budget must close: injected == stored + outflow by construction; the
    # meaningful check is that storage never exceeds inflow (no spurious water created)
    assert stored <= injected * 1.0001, "spurious mass created"
    assert hmax.max() < 50.0, "non-physical depth — check inflow/BCs"
    print("\nPASS: mass budget closes; inundation field physical")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        hs = hillshade(dem, res)
        depth = np.ma.masked_less(hmax, 0.10)
        fig, ax = plt.subplots(figsize=(11, 7.5))
        ax.imshow(hs, cmap="gray", origin="upper")
        im = ax.imshow(depth, cmap="Blues", origin="upper",
                       norm=Normalize(0, np.percentile(hmax[hmax > 0.1], 98)))
        plt.colorbar(im, ax=ax, label="max flood depth (m)", shrink=0.8)
        # mark the gauge
        gc = (2765384 - meta["E0"]) / res
        gr = (meta["N0"] - 1204914) / res
        ax.plot(gc, gr, "r^", ms=9, label="BAFU gauge 2150")
        ax.legend(loc="upper right")
        ax.set_title(f"Landquart @ Felsenbach — 2D flood inundation (peak {Q_PEAK:.0f} m³/s)\n"
                     f"SwissALTI3D 10 m terrain, hecras 2D shallow-water")
        ax.set_xlabel("east →")
        ax.set_ylabel("← north")
        fig.tight_layout()
        out = os.path.join(ROOT, "docs", "landquart_flood.png")
        fig.savefig(out, dpi=120)
        print(f"saved inundation map -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipping map)")


if __name__ == "__main__":
    main()
