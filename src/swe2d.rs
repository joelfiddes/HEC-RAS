//! 2D shallow-water flow: explicit finite-volume Godunov scheme on a Cartesian grid.
//!
//! Solves the conservative shallow-water equations
//!
//!   d(h)/dt   + d(hu)/dx     + d(hv)/dy     = 0
//!   d(hu)/dt  + d(hu^2+gh^2/2)/dx + d(huv)/dy = -g h dz/dx - g h Sfx
//!   d(hv)/dt  + d(huv)/dx + d(hv^2+gh^2/2)/dy = -g h dz/dy - g h Sfy
//!
//! with an HLL approximate Riemann solver at cell interfaces and the Audusse et al.
//! (2004) *hydrostatic reconstruction* of the bed source term, which makes the scheme
//! well-balanced: a lake at rest over an arbitrary bed stays exactly at rest. Manning
//! friction is applied point-implicitly for stability in shallow water, and wetting /
//! drying is handled through a dry-depth tolerance. First order in space and time —
//! the natural baseline that converges under grid refinement.

use crate::hydraulics::G;

const DRY: f64 = 1.0e-6;

/// Boundary condition applied to all four domain edges.
#[derive(Clone, Copy)]
pub enum Bc {
    /// Zero-gradient (open) — waves leave the domain.
    Transmissive,
    /// Solid wall — normal velocity reflected.
    Reflective,
}

pub struct Swe2dResult {
    pub nx: usize,
    pub ny: usize,
    pub h: Vec<f64>,  // final depth, row-major [j*nx + i]
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    pub times: Vec<f64>,
    pub volumes: Vec<f64>,
    pub steps: usize,
    pub t_final: f64,
}

#[inline]
fn vel(h: f64, q: f64) -> f64 {
    if h > DRY {
        q / h
    } else {
        0.0
    }
}

/// Physical flux in the normal direction for state (h, un, ut):
/// returns [mass, normal-momentum, transverse-momentum].
#[inline]
fn phys_flux(h: f64, un: f64, ut: f64) -> [f64; 3] {
    [h * un, h * un * un + 0.5 * G * h * h, h * un * ut]
}

/// HLL flux for the normal direction given left/right depths and velocities.
/// Components: [mass, normal momentum, transverse momentum].
fn hll(hl: f64, unl: f64, utl: f64, hr: f64, unr: f64, utr: f64) -> [f64; 3] {
    if hl <= DRY && hr <= DRY {
        return [0.0, 0.0, 0.0];
    }
    let cl = (G * hl.max(0.0)).sqrt();
    let cr = (G * hr.max(0.0)).sqrt();
    let fl = phys_flux(hl, unl, utl);
    let fr = phys_flux(hr, unr, utr);

    let (sl, sr) = if hl <= DRY {
        (unr - 2.0 * cr, unr + cr)
    } else if hr <= DRY {
        (unl - cl, unl + 2.0 * cl)
    } else {
        let us = 0.5 * (unl + unr) + cl - cr;
        let cs = 0.5 * (cl + cr) + 0.25 * (unl - unr);
        ((unl - cl).min(us - cs), (unr + cr).max(us + cs))
    };

    if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        // conserved states U = [h, h*un, h*ut]
        let ul = [hl, hl * unl, hl * utl];
        let ur = [hr, hr * unr, hr * utr];
        let mut f = [0.0; 3];
        for k in 0..3 {
            f[k] = (sr * fl[k] - sl * fr[k] + sl * sr * (ur[k] - ul[k])) / (sr - sl);
        }
        f
    }
}

/// Run the 2D shallow-water model to `t_end`.
///
/// `zb`, `h0`, `hu0`, `hv0` are interior fields of length nx*ny, row-major [j*nx + i].
#[allow(clippy::too_many_arguments)]
pub fn run_swe2d(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    zb: &[f64],
    h0: &[f64],
    hu0: &[f64],
    hv0: &[f64],
    manning_n: f64,
    t_end: f64,
    cfl: f64,
    bc: Bc,
    max_steps: usize,
) -> Swe2dResult {
    // Padded arrays with one ghost layer on each side.
    let w = nx + 2;
    let ht = ny + 2;
    let idx = |i: usize, j: usize| j * w + i;
    let mut h = vec![0.0; w * ht];
    let mut hu = vec![0.0; w * ht];
    let mut hv = vec![0.0; w * ht];
    let mut bed = vec![0.0; w * ht];
    for j in 0..ny {
        for i in 0..nx {
            let s = j * nx + i;
            let d = idx(i + 1, j + 1);
            h[d] = h0[s];
            hu[d] = hu0[s];
            hv[d] = hv0[s];
            bed[d] = zb[s];
        }
    }

    let fill_ghosts = |h: &mut [f64], hu: &mut [f64], hv: &mut [f64], bed: &mut [f64]| {
        let refl = matches!(bc, Bc::Reflective);
        // left (i=0) / right (i=nx+1) columns
        for j in 1..=ny {
            let (li, ri) = (idx(1, j), idx(0, j));
            h[ri] = h[li];
            bed[ri] = bed[li];
            hv[ri] = hv[li];
            hu[ri] = if refl { -hu[li] } else { hu[li] };
            let (li2, ri2) = (idx(nx, j), idx(nx + 1, j));
            h[ri2] = h[li2];
            bed[ri2] = bed[li2];
            hv[ri2] = hv[li2];
            hu[ri2] = if refl { -hu[li2] } else { hu[li2] };
        }
        // bottom (j=0) / top (j=ny+1) rows
        for i in 1..=nx {
            let (bi, gi) = (idx(i, 1), idx(i, 0));
            h[gi] = h[bi];
            bed[gi] = bed[bi];
            hu[gi] = hu[bi];
            hv[gi] = if refl { -hv[bi] } else { hv[bi] };
            let (bi2, gi2) = (idx(i, ny), idx(i, ny + 1));
            h[gi2] = h[bi2];
            bed[gi2] = bed[bi2];
            hu[gi2] = hu[bi2];
            hv[gi2] = if refl { -hv[bi2] } else { hv[bi2] };
        }
    };

    let volume = |h: &[f64]| -> f64 {
        let mut v = 0.0;
        for j in 1..=ny {
            for i in 1..=nx {
                v += h[idx(i, j)];
            }
        }
        v * dx * dy
    };

    let mut times = vec![0.0];
    let mut volumes = vec![volume(&h)];

    let mut t = 0.0;
    let mut step = 0;
    while t < t_end && step < max_steps {
        fill_ghosts(&mut h, &mut hu, &mut hv, &mut bed);

        // timestep from the CFL condition
        let mut inv_dt = 1.0e-12;
        for j in 1..=ny {
            for i in 1..=nx {
                let c = idx(i, j);
                if h[c] <= DRY {
                    continue;
                }
                let u = vel(h[c], hu[c]);
                let v = vel(h[c], hv[c]);
                let cc = (G * h[c]).sqrt();
                let s = (u.abs() + cc) / dx + (v.abs() + cc) / dy;
                if s > inv_dt {
                    inv_dt = s;
                }
            }
        }
        let mut dt = cfl / inv_dt;
        if t + dt > t_end {
            dt = t_end - t;
        }

        // flux accumulation
        let mut rh = vec![0.0; w * ht];
        let mut rhu = vec![0.0; w * ht];
        let mut rhv = vec![0.0; w * ht];

        // x-direction interfaces (between cell i and i+1, for j in 1..=ny)
        for j in 1..=ny {
            for i in 0..=nx {
                let l = idx(i, j);
                let r = idx(i + 1, j);
                let etal = h[l] + bed[l];
                let etar = h[r] + bed[r];
                let zface = bed[l].max(bed[r]);
                let hls = (etal - zface).max(0.0);
                let hrs = (etar - zface).max(0.0);
                let ul = vel(h[l], hu[l]);
                let ur = vel(h[r], hu[r]);
                let vl = vel(h[l], hv[l]);
                let vr = vel(h[r], hv[r]);
                let f = hll(hls, ul, vl, hrs, ur, vr); // normal=x, transverse=y
                // pressure corrections (well-balanced source) on normal momentum
                let fl1 = f[1] + 0.5 * G * (h[l] * h[l] - hls * hls);
                let fr1 = f[1] + 0.5 * G * (h[r] * h[r] - hrs * hrs);
                // accumulate: left cell -F/dx, right cell +F/dx
                if i >= 1 {
                    rh[l] -= f[0] / dx;
                    rhu[l] -= fl1 / dx;
                    rhv[l] -= f[2] / dx;
                }
                if i + 1 <= nx {
                    rh[r] += f[0] / dx;
                    rhu[r] += fr1 / dx;
                    rhv[r] += f[2] / dx;
                }
            }
        }

        // y-direction interfaces (between cell j and j+1, for i in 1..=nx)
        for i in 1..=nx {
            for j in 0..=ny {
                let l = idx(i, j);
                let r = idx(i, j + 1);
                let etal = h[l] + bed[l];
                let etar = h[r] + bed[r];
                let zface = bed[l].max(bed[r]);
                let hls = (etal - zface).max(0.0);
                let hrs = (etar - zface).max(0.0);
                let vl = vel(h[l], hv[l]);
                let vr = vel(h[r], hv[r]);
                let ul = vel(h[l], hu[l]);
                let ur = vel(h[r], hu[r]);
                let f = hll(hls, vl, ul, hrs, vr, ur); // normal=y, transverse=x
                let fl1 = f[1] + 0.5 * G * (h[l] * h[l] - hls * hls);
                let fr1 = f[1] + 0.5 * G * (h[r] * h[r] - hrs * hrs);
                if j >= 1 {
                    rh[l] -= f[0] / dy;
                    rhv[l] -= fl1 / dy; // normal momentum is hv
                    rhu[l] -= f[2] / dy; // transverse momentum is hu
                }
                if j + 1 <= ny {
                    rh[r] += f[0] / dy;
                    rhv[r] += fr1 / dy;
                    rhu[r] += f[2] / dy;
                }
            }
        }

        // update interior cells + point-implicit friction
        let n2 = manning_n * manning_n;
        for j in 1..=ny {
            for i in 1..=nx {
                let c = idx(i, j);
                h[c] += dt * rh[c];
                hu[c] += dt * rhu[c];
                hv[c] += dt * rhv[c];
                if h[c] <= DRY {
                    h[c] = h[c].max(0.0);
                    hu[c] = 0.0;
                    hv[c] = 0.0;
                    continue;
                }
                if n2 > 0.0 {
                    let u = hu[c] / h[c];
                    let v = hv[c] / h[c];
                    let speed = (u * u + v * v).sqrt();
                    if speed > 0.0 {
                        let cf = G * n2 * speed / h[c].powf(4.0 / 3.0);
                        let denom = 1.0 + dt * cf;
                        hu[c] /= denom;
                        hv[c] /= denom;
                    }
                }
            }
        }

        t += dt;
        step += 1;
        if step % 20 == 0 {
            times.push(t);
            volumes.push(volume(&h));
        }
    }

    times.push(t);
    volumes.push(volume(&h));

    // unpad interior fields
    let mut ho = vec![0.0; nx * ny];
    let mut huo = vec![0.0; nx * ny];
    let mut hvo = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let s = j * nx + i;
            let d = idx(i + 1, j + 1);
            ho[s] = h[d];
            huo[s] = hu[d];
            hvo[s] = hv[d];
        }
    }

    Swe2dResult {
        nx,
        ny,
        h: ho,
        hu: huo,
        hv: hvo,
        times,
        volumes,
        steps: step,
        t_final: t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audusse C-property: a flat water surface over a bumpy bed must remain at rest.
    #[test]
    fn lake_at_rest_is_well_balanced() {
        let nx = 40;
        let ny = 20;
        let (dx, dy) = (5.0, 5.0);
        let eta = 2.0; // flat water surface
        let mut zb = vec![0.0; nx * ny];
        let mut h0 = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                // a gaussian bump in the bed, partly submerged
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                let b = 1.5 * (-(((x - 100.0).powi(2) + (y - 50.0).powi(2)) / 800.0)).exp();
                zb[j * nx + i] = b;
                h0[j * nx + i] = (eta - b).max(0.0);
            }
        }
        let hu0 = vec![0.0; nx * ny];
        let hv0 = vec![0.0; nx * ny];
        let res = run_swe2d(
            nx, ny, dx, dy, &zb, &h0, &hu0, &hv0, 0.0, 50.0, 0.45, Bc::Reflective, 100000,
        );
        // velocities must stay ~0 and the surface flat
        let mut max_vel = 0.0_f64;
        let mut max_deta = 0.0_f64;
        for s in 0..nx * ny {
            let u = vel(res.h[s], res.hu[s]);
            let v = vel(res.h[s], res.hv[s]);
            max_vel = max_vel.max(u.hypot(v));
            if res.h[s] > DRY {
                max_deta = max_deta.max((res.h[s] + zb[s] - eta).abs());
            }
        }
        assert!(max_vel < 1e-10, "spurious velocity {:e}", max_vel);
        assert!(max_deta < 1e-10, "surface moved {:e}", max_deta);
    }

    /// Closed basin: total water volume is conserved as a wave sloshes around.
    #[test]
    fn mass_conserved_in_closed_basin() {
        let nx = 60;
        let ny = 40;
        let (dx, dy) = (2.0, 2.0);
        let zb = vec![0.0; nx * ny];
        let mut h0 = vec![1.0; nx * ny];
        // a raised blob of water that will collapse and slosh
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                if (x - 40.0).powi(2) + (y - 40.0).powi(2) < 200.0 {
                    h0[j * nx + i] = 3.0;
                }
            }
        }
        let hu0 = vec![0.0; nx * ny];
        let hv0 = vec![0.0; nx * ny];
        let res = run_swe2d(
            nx, ny, dx, dy, &zb, &h0, &hu0, &hv0, 0.02, 30.0, 0.45, Bc::Reflective, 100000,
        );
        let v0 = res.volumes[0];
        let vf = *res.volumes.last().unwrap();
        let rel = (vf - v0).abs() / v0;
        assert!(rel < 1e-12, "volume drifted by {:e} (reflective walls)", rel);
    }
}
