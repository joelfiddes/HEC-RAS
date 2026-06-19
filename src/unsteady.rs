//! 1D unsteady flow: the Saint-Venant equations via the Preissmann implicit box scheme.
//!
//! Unknowns at each node are the geodetic stage `z` (water-surface elevation) and the
//! discharge `Q`. The bed slope is carried implicitly by the cross-section thalweg
//! elevations along the reach, so the governing equations are
//!
//!   continuity:  dA/dt + dQ/dx = 0
//!   momentum:    dQ/dt + d(Q^2/A)/dx + g A dz/dx + g A Sf = 0,   Sf = Q|Q| / K^2
//!
//! Each box [i, i+1] is discretised with the four-point Preissmann weighting (time
//! weight `theta`, space weight 1/2), giving two nonlinear equations per box. With an
//! upstream and a downstream boundary condition this closes a 2N system that is solved
//! each timestep by Newton-Raphson (numeric Jacobian + dense LU — chosen for clarity
//! and robustness during validation; a banded/double-sweep solver is a later speedup).

use crate::geometry::CrossSection;
use crate::hydraulics::G;

/// Downstream boundary condition.
pub enum DownstreamBc {
    /// Prescribed stage at each time level (length nsteps + 1).
    Stage(Vec<f64>),
    /// Normal-depth rating Q = K(z) * sqrt(slope).
    Normal(f64),
}

/// Result of an unsteady run; `stage` and `discharge` are row-major [time][node],
/// with (nsteps + 1) rows and N columns.
pub struct UnsteadyResult {
    pub n: usize,
    pub nsteps: usize,
    pub stage: Vec<f64>,
    pub discharge: Vec<f64>,
    pub max_residual: Vec<f64>,
    pub converged: Vec<bool>,
}

/// Solve a dense linear system A x = b by Gaussian elimination with partial pivoting.
/// `a` is row-major n*n and is consumed; returns None if singular.
fn solve_dense(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let f = a[r * n + col] / diag;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    // back-substitution
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in (r + 1)..n {
            s -= a[r * n + c] * x[c];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

#[inline]
fn area(xs: &CrossSection, z: f64) -> f64 {
    xs.props(z).area
}

/// Momentum flux Q^2 / A (guarded against a dry section).
#[inline]
fn flux(xs: &CrossSection, z: f64, q: f64) -> f64 {
    let a = area(xs, z);
    if a > 1e-9 {
        q * q / a
    } else {
        0.0
    }
}

/// A * Sf = A * Q|Q| / K^2  (the gravity-weighted friction term, signed by Q).
#[inline]
fn a_sf(xs: &CrossSection, z: f64, q: f64) -> f64 {
    let a = area(xs, z);
    let k = xs.conveyance(z);
    if k > 1e-9 {
        a * q * q.abs() / (k * k)
    } else {
        0.0
    }
}

/// Assemble the 2N residual vector for the unknown vector `u` = [z0, q0, z1, q1, ...],
/// given the previous time level `u0`, the upstream discharge at the new time level
/// `inflow_next`, and the downstream BC value (stage or normal slope) at the new level.
#[allow(clippy::too_many_arguments)]
fn residual(
    u: &[f64],
    u0: &[f64],
    sections: &[CrossSection],
    dx: &[f64],
    theta: f64,
    dt: f64,
    inflow_next: f64,
    downstream: &DownstreamBc,
    step_next: usize,
) -> Vec<f64> {
    let n = sections.len();
    let m = 2 * n;
    let mut r = vec![0.0; m];
    let z = |i: usize| u[2 * i];
    let q = |i: usize| u[2 * i + 1];
    let z0 = |i: usize| u0[2 * i];
    let q0 = |i: usize| u0[2 * i + 1];

    // Row 0: upstream boundary — prescribed discharge.
    r[0] = q(0) - inflow_next;

    // Interior box equations -> rows 1 .. 2N-2.
    for i in 0..n - 1 {
        let l = dx[i];
        let xl = &sections[i];
        let xr = &sections[i + 1];

        // areas at both time levels
        let al_n = area(xl, z(i));
        let ar_n = area(xr, z(i + 1));
        let al_o = area(xl, z0(i));
        let ar_o = area(xr, z0(i + 1));

        // --- continuity ---
        let da_dt = ((al_n - al_o) + (ar_n - ar_o)) / (2.0 * dt);
        let dq_dx = (theta * (q(i + 1) - q(i)) + (1.0 - theta) * (q0(i + 1) - q0(i))) / l;
        r[1 + 2 * i] = da_dt + dq_dx;

        // --- momentum ---
        let dq_dt = ((q(i) - q0(i)) + (q(i + 1) - q0(i + 1))) / (2.0 * dt);

        let fl_n = flux(xl, z(i), q(i));
        let fr_n = flux(xr, z(i + 1), q(i + 1));
        let fl_o = flux(xl, z0(i), q0(i));
        let fr_o = flux(xr, z0(i + 1), q0(i + 1));
        let dflux_dx = (theta * (fr_n - fl_n) + (1.0 - theta) * (fr_o - fl_o)) / l;

        let abar = theta * 0.5 * (al_n + ar_n) + (1.0 - theta) * 0.5 * (al_o + ar_o);
        let dz_dx = (theta * (z(i + 1) - z(i)) + (1.0 - theta) * (z0(i + 1) - z0(i))) / l;

        let asf_ln = a_sf(xl, z(i), q(i));
        let asf_rn = a_sf(xr, z(i + 1), q(i + 1));
        let asf_lo = a_sf(xl, z0(i), q0(i));
        let asf_ro = a_sf(xr, z0(i + 1), q0(i + 1));
        let asf_bar = theta * 0.5 * (asf_ln + asf_rn) + (1.0 - theta) * 0.5 * (asf_lo + asf_ro);

        r[2 + 2 * i] = dq_dt + dflux_dx + G * abar * dz_dx + G * asf_bar;
    }

    // Last row: downstream boundary.
    let last = n - 1;
    r[m - 1] = match downstream {
        DownstreamBc::Stage(series) => z(last) - series[step_next],
        DownstreamBc::Normal(slope) => q(last) - sections[last].conveyance(z(last)) * slope.sqrt(),
    };
    r
}

/// Route an unsteady flow through the reach.
///
/// `sections` ordered upstream..downstream is *not* required; index 0 is the upstream
/// boundary (where `inflow_q` is applied) and the last index is the downstream boundary.
/// `dx` has length N-1. `z0`/`q0` are the initial stage/discharge (length N).
/// `inflow_q` has length nsteps+1 (discharge at each time level, including t=0).
#[allow(clippy::too_many_arguments)]
pub fn route_unsteady(
    sections: &[CrossSection],
    dx: &[f64],
    z0: &[f64],
    q0: &[f64],
    inflow_q: &[f64],
    dt: f64,
    nsteps: usize,
    theta: f64,
    downstream: &DownstreamBc,
    newton_tol: f64,
    max_newton: usize,
) -> UnsteadyResult {
    let n = sections.len();
    let m = 2 * n;

    let mut u = vec![0.0; m];
    for i in 0..n {
        u[2 * i] = z0[i];
        u[2 * i + 1] = q0[i];
    }

    let mut stage = vec![0.0; (nsteps + 1) * n];
    let mut discharge = vec![0.0; (nsteps + 1) * n];
    let mut max_residual = vec![0.0; nsteps];
    let mut converged = vec![false; nsteps];
    // store initial state
    for i in 0..n {
        stage[i] = u[2 * i];
        discharge[i] = u[2 * i + 1];
    }

    for step in 0..nsteps {
        let u_prev = u.clone(); // previous time level (known)
        let inflow_next = inflow_q[step + 1];
        let step_next = step + 1;

        // Newton iterations: solve residual(u) = 0 for the new time level.
        let mut last_max = f64::INFINITY;
        let mut conv = false;
        for _it in 0..max_newton {
            let r = residual(
                &u, &u_prev, sections, dx, theta, dt, inflow_next, downstream, step_next,
            );
            last_max = r.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            if last_max < newton_tol {
                conv = true;
                break;
            }

            // numeric Jacobian (full; cheap enough for validation grids)
            let mut jac = vec![0.0; m * m];
            for j in 0..m {
                let save = u[j];
                let h = 1e-6 * (1.0 + save.abs());
                u[j] = save + h;
                let rj = residual(
                    &u, &u_prev, sections, dx, theta, dt, inflow_next, downstream, step_next,
                );
                u[j] = save;
                for i in 0..m {
                    jac[i * m + j] = (rj[i] - r[i]) / h;
                }
            }
            let rhs: Vec<f64> = r.iter().map(|v| -v).collect();
            match solve_dense(jac, rhs, m) {
                Some(du) => {
                    for j in 0..m {
                        u[j] += du[j];
                    }
                }
                None => break, // singular: bail this step, keep last iterate
            }
        }

        max_residual[step] = last_max;
        converged[step] = conv;
        let base = (step + 1) * n;
        for i in 0..n {
            stage[base + i] = u[2 * i];
            discharge[base + i] = u[2 * i + 1];
        }
    }

    UnsteadyResult {
        n,
        nsteps,
        stage,
        discharge,
        max_residual,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CrossSection;
    use crate::hydraulics::normal_wse;

    fn trap_section(z_bed: f64) -> CrossSection {
        // trapezoid: bottom width 10, side slope 2:1, banks 6 m
        let b = 10.0;
        let m = 2.0;
        let h = 6.0;
        CrossSection::new(
            vec![0.0, m * h, m * h + b, 2.0 * m * h + b],
            vec![z_bed + h, z_bed, z_bed, z_bed + h],
            0.03,
        )
    }

    /// With constant inflow and a normal-depth tail water, a reach started at normal
    /// depth must *stay* at normal depth (steady state is a fixed point of the scheme).
    #[test]
    fn steady_state_is_a_fixed_point() {
        let q = 50.0;
        let slope = 0.001;
        let dx_val = 200.0;
        let n = 11;
        let sections: Vec<_> = (0..n).map(|i| trap_section(slope * dx_val * (n - 1 - i) as f64)).collect();
        // index 0 = upstream (highest bed). bed drops downstream.
        let dx = vec![dx_val; n - 1];
        let yn = normal_wse(&sections[0], q, slope).unwrap() - sections[0].min_elevation();
        let z0: Vec<f64> = sections.iter().map(|s| s.min_elevation() + yn).collect();
        let q0 = vec![q; n];
        let inflow = vec![q; 6];
        let res = route_unsteady(
            &sections,
            &dx,
            &z0,
            &q0,
            &inflow,
            10.0,
            5,
            0.6,
            &DownstreamBc::Normal(slope),
            1e-8,
            50,
        );
        assert!(res.converged.iter().all(|&c| c), "newton failed to converge");
        // final stage should equal initial (normal-depth) stage
        let base = res.nsteps * n;
        for i in 0..n {
            let drift = (res.stage[base + i] - z0[i]).abs();
            assert!(drift < 1e-3, "node {} drifted {} m from steady state", i, drift);
        }
    }

    /// Mass conservation: storage change equals net inflow over the run.
    #[test]
    fn mass_is_conserved() {
        let slope = 0.0008;
        let dx_val = 200.0;
        let n = 11;
        let sections: Vec<_> = (0..n).map(|i| trap_section(slope * dx_val * (n - 1 - i) as f64)).collect();
        let dx = vec![dx_val; n - 1];
        let q_base = 40.0;
        let yn = normal_wse(&sections[0], q_base, slope).unwrap() - sections[0].min_elevation();
        let z0: Vec<f64> = sections.iter().map(|s| s.min_elevation() + yn).collect();
        let q0 = vec![q_base; n];

        let dt = 20.0;
        let nsteps = 120;
        // triangular flood pulse on top of the base flow
        let mut inflow = vec![q_base; nsteps + 1];
        for (k, v) in inflow.iter_mut().enumerate() {
            let peak = 30;
            let extra = if k <= peak {
                60.0 * (k as f64 / peak as f64)
            } else if k <= 2 * peak {
                60.0 * (2.0 - k as f64 / peak as f64)
            } else {
                0.0
            };
            *v += extra;
        }

        let res = route_unsteady(
            &sections,
            &dx,
            &z0,
            &q0,
            &inflow,
            dt,
            nsteps,
            0.6,
            &DownstreamBc::Normal(slope),
            1e-7,
            50,
        );
        assert!(res.converged.iter().all(|&c| c));

        // storage volume from cross-sectional areas (trapezoidal in x)
        let storage = |row: usize| -> f64 {
            let mut v = 0.0;
            for i in 0..n - 1 {
                let ai = area(&sections[i], res.stage[row * n + i]);
                let aip = area(&sections[i + 1], res.stage[row * n + i + 1]);
                v += 0.5 * (ai + aip) * dx[i];
            }
            v
        };
        let d_storage = storage(nsteps) - storage(0);

        // net inflow volume (trapezoidal in time) using boundary discharges
        let mut net_in = 0.0;
        for k in 0..nsteps {
            let qin0 = res.discharge[k * n];
            let qin1 = res.discharge[(k + 1) * n];
            let qout0 = res.discharge[k * n + (n - 1)];
            let qout1 = res.discharge[(k + 1) * n + (n - 1)];
            net_in += 0.5 * ((qin0 - qout0) + (qin1 - qout1)) * dt;
        }

        let err = (d_storage - net_in).abs();
        let rel = err / net_in.abs().max(1.0);
        assert!(rel < 0.02, "mass error {:.4} m^3 (rel {:.4})", err, rel);
    }
}
