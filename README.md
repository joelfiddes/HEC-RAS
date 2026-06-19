# HEC-RAS (Rust)

A **from-scratch** 1D/2D hydraulic river-analysis engine in the spirit of the US Army
Corps of Engineers' HEC-RAS, written in Rust with Python bindings. It is *not* a wrapper
around the real HEC-RAS software — the open-channel hydraulics are implemented directly,
so the engine runs natively on macOS/Linux and is scriptable from Python.

Same architecture as [`fsm-rs`](https://github.com/joelfiddes/fsm-rs): a Rust core
(`src/`) exposed to Python via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

## Status

### Phase 1 — 1D steady flow ✅

- Cross-section geometry from station/elevation points (area, wetted perimeter, top
  width, hydraulic radius, Manning conveyance).
- Froude number, critical-depth and normal-depth solvers.
- Subcritical **standard-step** (energy-method) backwater profiles along a reach.
- Validated against an independent analytical gradually-varied-flow integration
  (`examples/trapezoidal_backwater.py`): the engine shows clean **2nd-order
  convergence** to the analytic M1 backwater (error halves 4× per grid halving,
  down to ~0.002 mm).

![M1 backwater validation](docs/trapezoidal_backwater.png)

### Phase 2 — 1D unsteady flow ✅

- Full **Saint-Venant** equations solved with the **Preissmann four-point implicit
  box scheme**; Newton-Raphson per timestep.
- Upstream discharge hydrograph; downstream normal-depth rating or prescribed stage.
- Validated (`examples/flood_routing.py`) on three independent checks: steady-state
  recovery to the Phase-1 profile (< 0.01 mm), **mass conservation** (< 0.1 %), and
  physically-correct flood-wave **attenuation + lag** (celerity ≈ kinematic estimate).

![Flood-wave routing validation](docs/flood_routing.png)

### Roadmap

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | 1D steady gradually-varied flow (standard-step) | ✅ done |
| 2 | 1D unsteady — Saint-Venant via Preissmann implicit scheme | ✅ done |
| 3 | 2D shallow-water finite-volume on a DEM mesh | planned |
| 4 | Application: Swiss reach (SwissALTI3D + BAFU gauge) → transfer to Central Asia | planned |

Units are **SI / metric** throughout.

## Install

```bash
cd ~/src/HEC-RAS
maturin develop --release      # builds the Rust extension into the active env
```

## Quickstart

```python
from hecras import CrossSection, steady_profile

# Trapezoidal section: station/elevation points + Manning's n
xs = CrossSection([0, 12, 22, 34], [6, 0, 0, 6], 0.03)
xs.normal_depth(50.0, 0.001)     # uniform-flow depth for Q, bed slope
xs.critical_depth(50.0)          # critical depth for Q

# Backwater profile along a reach (sections ordered downstream -> upstream)
sections = [xs, xs, xs]
reach_lengths = [100.0, 100.0]   # len(sections) - 1
res = steady_profile(sections, reach_lengths, q=50.0, downstream_wse=2.5)
res["wse"], res["depth"], res["froude"]   # numpy arrays, one value per section
```

### Unsteady flood routing (Phase 2)

```python
from hecras import route_unsteady

# sections ordered upstream -> downstream; inflow hydrograph at the upstream end
res = route_unsteady(
    sections, reach_lengths,
    initial_stage=z0, initial_q=q0,        # state at t=0, length N
    inflow_q=hydrograph,                   # length n_steps + 1
    dt=20.0, n_steps=900,
    downstream="normal", downstream_slope=6e-4,   # or downstream="stage", downstream_stage=...
)
res["time"], res["inflow"], res["outflow"]        # 1D arrays
res["stage"], res["discharge"]                    # 2D arrays [time, node]
```

## Develop / test

```bash
cargo test --no-default-features     # Rust unit tests (geometry, hydraulics, steady)
python examples/trapezoidal_backwater.py
```

`--no-default-features` disables the `extension-module` feature so the test harness can
link libpython; the Python wheel build (maturin) keeps it enabled.

## Theory (Phase 1)

Subcritical profiles march upstream from a downstream control, balancing the energy
equation between adjacent sections:

```
WSE_up + α·V_up²/2g  =  WSE_dn + α·V_dn²/2g  +  h_f  +  h_eddy
```

with friction loss `h_f = S̄_f · L` (average of friction slopes `S_f = (Q/K)²`,
conveyance `K = (1/n)·A·R^(2/3)`) and eddy loss `h_eddy = C·|ΔV²/2g|` using
contraction/expansion coefficients. The upstream water surface is found by bisection,
bracketed above critical depth so the physically-correct subcritical root is selected.

## License

MIT © Joel Fiddes
