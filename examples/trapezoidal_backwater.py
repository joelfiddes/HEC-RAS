"""Validate the standard-step solver against an analytical backwater profile.

A prismatic trapezoidal channel carries a steady discharge on a mild slope. We
impose a downstream control above normal depth, producing an M1 backwater curve.

Two solutions are compared:

  1. The Rust engine (`hecras.steady_profile`), standard-step / energy method.
  2. An independent Python reference that integrates the gradually-varied-flow ODE

         dy/dx = (S0 - Sf) / (1 - Fr^2)

     upstream with RK4 (the "direct-step"-style differential form).

If the engine is correct the two water-surface profiles agree to a few millimetres.
"""

import math
import numpy as np
from hecras import CrossSection, steady_profile

# ---------------------------------------------------------------------------
# Channel definition (SI units)
# ---------------------------------------------------------------------------
B = 10.0        # bottom width (m)
M = 2.0         # side slope (horizontal:vertical = M:1)
N = 0.03        # Manning's n
S0 = 0.001      # bed slope (-)
Q = 50.0        # discharge (m^3/s)
BANK_H = 6.0    # bank height above bed (m)

L_TOTAL = 5000.0
N_SEC = 51
DX = L_TOTAL / (N_SEC - 1)
G = 9.81


# ---------------------------------------------------------------------------
# Analytical trapezoid geometry (depth y above the local bed)
# ---------------------------------------------------------------------------
def area(y):
    return (B + M * y) * y


def top_width(y):
    return B + 2.0 * M * y


def wetted_perimeter(y):
    return B + 2.0 * y * math.sqrt(1.0 + M * M)


def conveyance(y):
    a = area(y)
    r = a / wetted_perimeter(y)
    return (1.0 / N) * a * r ** (2.0 / 3.0)


def friction_slope(y):
    return (Q / conveyance(y)) ** 2


def froude2(y):
    return Q * Q * top_width(y) / (G * area(y) ** 3)


def normal_depth():
    # solve conveyance(y) sqrt(S0) = Q
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if conveyance(mid) * math.sqrt(S0) < Q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_depth():
    # solve froude2(y) = 1
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if froude2(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gvf_profile_rk4(y_down, n_sec, dx):
    """Integrate the GVF profile marching upstream.

    With distance s measured positive upstream, dy/ds = -(dy/dx) so

        dy/ds = (Sf - S0) / (1 - Fr^2)
    """
    def dyds(y):
        return (friction_slope(y) - S0) / (1.0 - froude2(y))

    ys = [y_down]
    y = y_down
    for _ in range(n_sec - 1):
        k1 = dyds(y)
        k2 = dyds(y + 0.5 * dx * k1)
        k3 = dyds(y + 0.5 * dx * k2)
        k4 = dyds(y + dx * k3)
        y = y + dx / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return np.array(ys)


# ---------------------------------------------------------------------------
# Build cross sections for the engine (downstream -> upstream).
# Station/elevation trapezoid with the bed rising by S0*dx going upstream.
# ---------------------------------------------------------------------------
def trapezoid_section(z_bed):
    left_top = (0.0, z_bed + BANK_H)
    left_toe = (M * BANK_H, z_bed)
    right_toe = (M * BANK_H + B, z_bed)
    right_top = (2.0 * M * BANK_H + B, z_bed + BANK_H)
    stations = [left_top[0], left_toe[0], right_toe[0], right_top[0]]
    elevs = [left_top[1], left_toe[1], right_toe[1], right_top[1]]
    return CrossSection(stations, elevs, N)


def run_case(n_sec):
    """Solve one reach with `n_sec` sections; return (x, z_bed, engine_y, ref_y, res)."""
    dx = L_TOTAL / (n_sec - 1)
    y_control = 1.5 * normal_depth()  # 50% above normal depth -> M1 backwater
    x = np.array([i * dx for i in range(n_sec)])
    z_bed = S0 * x
    sections = [trapezoid_section(zb) for zb in z_bed]
    reach_lengths = [dx] * (n_sec - 1)
    # Disable eddy (expansion/contraction) losses: the differential GVF equation
    # accounts only for friction, so the apples-to-apples comparison sets C = 0.
    res = steady_profile(
        sections, reach_lengths, Q, z_bed[0] + y_control,
        alpha=1.0, contraction=0.0, expansion=0.0,
    )
    engine_y = np.asarray(res["depth"])
    ref_y = gvf_profile_rk4(y_control, n_sec, dx)
    return x, z_bed, engine_y, ref_y, res


def main():
    yn = normal_depth()
    yc = critical_depth()
    print(f"normal depth  yn = {yn:.4f} m")
    print(f"critical depth yc = {yc:.4f} m  ({'mild' if yn > yc else 'steep'} slope)")

    # --- Grid-refinement study: standard-step (engine) vs RK4 GVF (reference) -------
    # Both are approximations of the same continuous profile; their difference must
    # shrink as the reach length dx -> 0, confirming the engine converges to it.
    print("\nGrid-refinement convergence (engine standard-step vs analytic RK4):")
    print("   dx (m)    max |diff| (mm)   ratio")
    prev = None
    diffs = []
    for n_sec in (26, 51, 101, 201, 401):
        dx = L_TOTAL / (n_sec - 1)
        _, _, eng, ref, res = run_case(n_sec)
        md = float(np.abs(eng - ref).max()) * 1000.0
        ratio = f"{prev / md:5.2f}x" if prev else "   -"
        print(f"  {dx:7.1f}   {md:14.4f}   {ratio}")
        diffs.append(md)
        prev = md
        assert bool(np.asarray(res["converged"]).all()), "a section failed to converge"

    # The difference must decrease under refinement and vanish at the finest grid.
    assert diffs == sorted(diffs, reverse=True), f"not monotonically converging: {diffs}"
    tol_mm = 1.0
    assert diffs[-1] < tol_mm, f"finest-grid disagreement {diffs[-1]:.4f} mm (> {tol_mm} mm)"
    print(f"\nPASS: engine converges to the analytical GVF profile "
          f"(finest-grid diff {diffs[-1]:.4f} mm < {tol_mm} mm)")

    # --- Detailed profile at a working resolution ----------------------------------
    x, z_bed, engine_depth, ref_depth, res = run_case(N_SEC)
    diff = np.abs(engine_depth - ref_depth)
    print(f"\nProfile at dx={DX:.0f} m  (min Fr {np.nanmin(res['froude']):.3f}, "
          f"max Fr {np.nanmax(res['froude']):.3f}, subcritical M1)")
    print(f"backwater decays toward normal depth: upstream depth = "
          f"{engine_depth[-1]:.4f} m (yn={yn:.4f})")
    print("\n   x (m)   bed (m)   engine y   analytic y   diff (mm)")
    for i in range(0, N_SEC, 5):
        print(f"  {x[i]:7.0f}  {z_bed[i]:7.3f}  {engine_depth[i]:8.4f}   "
              f"{ref_depth[i]:9.4f}  {diff[i]*1000:9.3f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        wse_engine = z_bed + engine_depth
        wse_ref = z_bed + ref_depth
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, z_bed, "k-", lw=1.5, label="bed")
        ax.plot(x, z_bed + yn, "g--", lw=1, label=f"normal depth (yn={yn:.2f} m)")
        ax.plot(x, wse_ref, "b-", lw=2, label="analytic GVF (RK4)")
        ax.plot(x, wse_engine, "r.", ms=5, label="hecras standard-step")
        ax.set_xlabel("distance upstream (m)")
        ax.set_ylabel("elevation (m)")
        ax.set_title(f"M1 backwater, trapezoidal channel — max diff {diff.max()*1000:.2f} mm @ dx={DX:.0f} m")
        ax.legend()
        ax.grid(alpha=0.3)
        out = "trapezoidal_backwater.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
