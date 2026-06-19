"""Validate the 2D shallow-water solver against the exact Stoker dam-break solution.

A dam at x = x0 separates still water of depth `hl` (left) from `hr` (right) on a flat,
frictionless bed. At t > 0 the release produces a rarefaction fan moving left and a bore
(shock) moving right, with a constant state in between. Stoker (1957) gives the exact
solution; we run the 2D engine on a long, thin channel (reflective side walls -> the flow
is effectively 1D) and compare the centerline depth to the analytic profile.

A grid-refinement study shows the first-order finite-volume scheme converging to the
exact solution (L1 error decreasing as the mesh is refined).
"""

import math
import numpy as np
from hecras import run_swe2d

G = 9.81
HL = 5.0       # upstream depth (m)
HR = 1.0       # downstream depth (m)
L = 2000.0     # domain length (m)
X0 = 1000.0    # dam position
T_END = 30.0   # simulation time (s)


def stoker(x, t, hl=HL, hr=HR, x0=X0):
    """Exact wet-bed dam-break depth profile (Stoker 1957)."""
    if t <= 0:
        return np.where(x <= x0, hl, hr)

    cl = math.sqrt(G * hl)

    # middle-state depth hm solves the rarefaction/shock matching condition
    def f(hm):
        cm = math.sqrt(G * hm)
        # velocity behind the bore from the shock relations...
        u_shock = (hm - hr) * math.sqrt(0.5 * G * (1.0 / hm + 1.0 / hr))
        # ...must equal the velocity from the rarefaction invariant
        u_rare = 2.0 * (cl - cm)
        return u_shock - u_rare

    # f is increasing in hm: f(hr) < 0, f(hl) > 0
    lo, hi = hr, hl
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    hm = 0.5 * (lo + hi)
    cm = math.sqrt(G * hm)
    um = 2.0 * (cl - cm)
    s_shock = um * hm / (hm - hr)          # bore speed

    xi = (x - x0) / t
    h = np.empty_like(x, dtype=float)
    for k, z in enumerate(xi):
        if z <= -cl:
            h[k] = hl
        elif z <= um - cm:                 # rarefaction fan
            c = (2.0 * cl - z) / 3.0
            h[k] = c * c / G
        elif z <= s_shock:                 # constant middle state
            h[k] = hm
        else:
            h[k] = hr
    return h


def run_case(nx):
    dx = L / nx
    ny = 3
    dy = dx
    xc = (np.arange(nx) + 0.5) * dx
    bed = np.zeros(nx * ny)
    h0 = np.where(xc <= X0, HL, HR)
    h0 = np.tile(h0, ny)
    hu0 = np.zeros(nx * ny)
    hv0 = np.zeros(nx * ny)
    res = run_swe2d(
        nx, ny, dx, dy, list(bed), list(h0), list(hu0), list(hv0),
        manning_n=0.0, t_end=T_END, cfl=0.45, bc="transmissive",
    )
    h = np.asarray(res["h"])           # [ny, nx]
    centerline = h[ny // 2]            # middle row
    exact = stoker(xc, res["t_final"])
    l1 = np.mean(np.abs(centerline - exact))
    return xc, centerline, exact, l1, res


def main():
    print(f"Stoker dam-break: hl={HL} m, hr={HR} m, L={L:.0f} m, t={T_END:.0f} s\n")
    print("  grid (nx)    dx (m)    L1 error (m)    ratio")
    prev = None
    errs = []
    saved = None
    for nx in (250, 500, 1000, 2000):
        xc, cl, exact, l1, res = run_case(nx)
        ratio = f"{prev / l1:5.2f}x" if prev else "   -"
        print(f"  {nx:8d}   {L/nx:6.2f}   {l1:12.5f}    {ratio}")
        errs.append(l1)
        prev = l1
        if nx == 1000:
            saved = (xc, cl, exact)

    assert errs == sorted(errs, reverse=True), f"not converging: {errs}"
    assert errs[-1] < 0.05, f"finest-grid L1 error too large: {errs[-1]:.4f} m"
    print(f"\nPASS: converges to the exact Stoker solution "
          f"(finest L1 = {errs[-1]:.5f} m, shock-capturing 1st-order)")

    try:
        import matplotlib.pyplot as plt

        xc, cl, exact = saved
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xc, exact, "b-", lw=2, label="Stoker exact")
        ax.plot(xc, cl, "r.", ms=3, label="hecras 2D SWE (nx=1000)")
        ax.axvline(X0, color="k", ls=":", lw=1, alpha=0.5, label="dam")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("depth (m)")
        ax.set_title(f"2D dam-break vs Stoker analytical solution (t = {T_END:.0f} s)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = "dam_break_2d.png"
        fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
