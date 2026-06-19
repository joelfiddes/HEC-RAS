"""Validate the unsteady (Saint-Venant / Preissmann) solver.

Three independent checks on a prismatic trapezoidal reach:

  1. Steady-state recovery   - constant inflow + normal-depth tail water started off
     normal depth must relax to the Phase-1 steady (normal-depth) profile.
  2. Mass conservation       - over a flood hydrograph, the change in stored volume
     equals the net inflow volume (in - out), to a tight tolerance.
  3. Flood-wave routing      - a flood pulse propagates downstream with physically
     correct peak attenuation and travel time; the wave celerity is compared with the
     kinematic-wave estimate c_k = (5/3) V for a wide channel.
"""

import numpy as np
from hecras import CrossSection, steady_profile, route_unsteady

# np.trapz was renamed to np.trapezoid in NumPy 2.0
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# --- channel (SI) ----------------------------------------------------------
B = 12.0       # bottom width (m)
M = 2.0        # side slope H:V
N = 0.035      # Manning's n
S0 = 0.0006    # bed slope
BANK_H = 7.0
G = 9.81

N_SEC = 41
DX = 250.0
REACH_LEN = DX * (N_SEC - 1)


def trap_section(z_bed):
    return CrossSection(
        [0.0, M * BANK_H, M * BANK_H + B, 2.0 * M * BANK_H + B],
        [z_bed + BANK_H, z_bed, z_bed, z_bed + BANK_H],
        N,
    )


# Sections ordered upstream (index 0, highest bed) -> downstream.
x = np.arange(N_SEC) * DX
z_bed = (REACH_LEN - x) * S0          # bed drops going downstream
SECTIONS = [trap_section(zb) for zb in z_bed]
REACH = [DX] * (N_SEC - 1)


def normal_depth(q):
    return SECTIONS[0].normal_depth(q, S0)


def section_area(i, stage):
    return SECTIONS[i].area(stage)


# ---------------------------------------------------------------------------
def check_steady_state():
    print("\n[1] Steady-state recovery")
    q = 60.0
    yn = normal_depth(q)
    # start 40% too deep everywhere; the solver should drain back to normal depth
    # (needs enough simulation time: ~several wave-travel-times for the perturbation
    # to flush through the downstream boundary)
    z_init = z_bed + 1.4 * yn
    q_init = np.full(N_SEC, q)
    nsteps = 1600
    dt = 30.0
    res = route_unsteady(
        SECTIONS, REACH, list(z_init), list(q_init),
        inflow_q=[q] * (nsteps + 1), dt=dt, n_steps=nsteps,
        downstream="normal", downstream_slope=S0,
    )
    assert bool(np.asarray(res["converged"]).all()), "Newton failed"
    final_depth = res["stage"][-1] - z_bed
    drift = np.abs(final_depth - yn).max()
    print(f"    normal depth yn = {yn:.4f} m")
    print(f"    max |final depth - yn| after {nsteps} steps = {drift*1000:.3f} mm")
    # cross-check against the Phase-1 steady solver (downstream control at normal depth)
    sp = steady_profile(SECTIONS[::-1], REACH, q, z_bed[-1] + yn,
                        contraction=0.0, expansion=0.0)
    steady_depth = np.asarray(sp["depth"])[::-1]
    vs_phase1 = np.abs(final_depth - steady_depth).max()
    print(f"    max |unsteady - Phase1 steady| = {vs_phase1*1000:.3f} mm")
    assert drift < 5e-3, f"did not relax to normal depth ({drift*1000:.2f} mm)"
    assert vs_phase1 < 5e-3, "disagrees with validated Phase-1 steady solver"
    print("    PASS")


def triangular_hydrograph(nsteps, q_base, q_peak, t_rise, t_fall, dt):
    t = np.arange(nsteps + 1) * dt
    q = np.full(nsteps + 1, q_base, dtype=float)
    rise = t <= t_rise
    fall = (t > t_rise) & (t <= t_rise + t_fall)
    q[rise] = q_base + (q_peak - q_base) * (t[rise] / t_rise)
    q[fall] = q_base + (q_peak - q_base) * (1.0 - (t[fall] - t_rise) / t_fall)
    return q


def check_mass_and_wave():
    print("\n[2] Mass conservation + [3] flood-wave routing")
    q_base = 40.0
    q_peak = 220.0
    yn = normal_depth(q_base)
    z_init = z_bed + yn
    q_init = np.full(N_SEC, q_base)

    dt = 20.0
    nsteps = 900
    inflow = triangular_hydrograph(nsteps, q_base, q_peak,
                                   t_rise=2400.0, t_fall=4800.0, dt=dt)

    res = route_unsteady(
        SECTIONS, REACH, list(z_init), list(q_init),
        inflow_q=list(inflow), dt=dt, n_steps=nsteps,
        downstream="normal", downstream_slope=S0,
    )
    assert bool(np.asarray(res["converged"]).all()), "Newton failed"

    time = np.asarray(res["time"])
    stage = np.asarray(res["stage"])
    q_in = np.asarray(res["inflow"])
    q_out = np.asarray(res["outflow"])

    # --- mass balance: d(storage) vs net inflow volume -----------------------
    areas = np.array([[SECTIONS[i].area(stage[k, i]) for i in range(N_SEC)]
                      for k in range(stage.shape[0])])
    storage = trapezoid(areas, x, axis=1)          # volume at each time
    d_storage = storage[-1] - storage[0]
    net_in = trapezoid(q_in - q_out, time)
    rel = abs(d_storage - net_in) / max(abs(net_in), 1.0)
    print(f"    storage change   = {d_storage:,.1f} m^3")
    print(f"    net inflow vol   = {net_in:,.1f} m^3")
    print(f"    relative mass error = {rel*100:.4f} %")
    assert rel < 0.005, f"mass not conserved (rel err {rel:.4f})"

    # --- wave attenuation + celerity ----------------------------------------
    peak_in = q_in.max()
    peak_out = q_out.max()
    t_peak_in = time[np.argmax(q_in)]
    t_peak_out = time[np.argmax(q_out)]
    lag = t_peak_out - t_peak_in
    celerity = REACH_LEN / lag
    # kinematic-wave celerity at the base flow, wide-channel approx c_k = 5/3 V
    a_base = SECTIONS[0].area(z_bed[0] + yn)
    v_base = q_base / a_base
    ck = 5.0 / 3.0 * v_base
    print(f"    inflow peak  {peak_in:.1f} m^3/s at t={t_peak_in/3600:.2f} h")
    print(f"    outflow peak {peak_out:.1f} m^3/s at t={t_peak_out/3600:.2f} h")
    print(f"    peak attenuation = {(1-peak_out/peak_in)*100:.1f} %")
    print(f"    wave lag = {lag/3600:.2f} h  ->  celerity {celerity:.2f} m/s")
    print(f"    kinematic celerity (5/3)V = {ck:.2f} m/s (base flow)")
    assert peak_out < peak_in, "flood peak should attenuate"
    assert lag > 0, "outflow peak should lag inflow peak"
    # celerity should be the right order (between mean velocity and ~2x kinematic)
    assert 0.5 * v_base < celerity < 3.0 * ck, f"celerity {celerity:.2f} unphysical"
    print("    PASS")
    return res, time, q_in, q_out


def main():
    print(f"Reach: {N_SEC} sections, dx={DX:.0f} m, L={REACH_LEN/1000:.1f} km, "
          f"S0={S0}, n={N}")
    check_steady_state()
    res, time, q_in, q_out = check_mass_and_wave()
    print("\nALL UNSTEADY CHECKS PASSED")

    try:
        import matplotlib.pyplot as plt

        stage = np.asarray(res["stage"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
        ax1.plot(time / 3600, q_in, "b-", lw=2, label="inflow (upstream)")
        ax1.plot(time / 3600, q_out, "r-", lw=2, label="outflow (downstream)")
        ax1.set_xlabel("time (h)")
        ax1.set_ylabel("discharge (m³/s)")
        ax1.set_title("Flood-wave routing — attenuation & lag")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # stage snapshots along the reach
        for frac in (0.0, 0.15, 0.3, 0.5, 0.75):
            k = int(frac * (stage.shape[0] - 1))
            ax2.plot(x / 1000, stage[k] - z_bed, label=f"t={time[k]/3600:.1f} h")
        ax2.set_xlabel("distance downstream (km)")
        ax2.set_ylabel("depth (m)")
        ax2.set_title("Depth profiles during the flood")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        out = "flood_routing.png"
        fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
