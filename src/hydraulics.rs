//! Open-channel hydraulics: Froude number, critical depth and normal depth solvers.

use crate::geometry::CrossSection;

/// Acceleration due to gravity (m/s^2).
pub const G: f64 = 9.81;

/// Froude number Fr = V / sqrt(g * D), with hydraulic depth D = A / T.
/// Fr < 1 subcritical, Fr = 1 critical, Fr > 1 supercritical.
pub fn froude(xs: &CrossSection, q: f64, wse: f64) -> f64 {
    let p = xs.props(wse);
    if p.area <= 0.0 || p.top_width <= 0.0 {
        return f64::NAN;
    }
    let hyd_depth = p.area / p.top_width;
    (q / p.area) / (G * hyd_depth).sqrt()
}

/// Bisection on a monotone function; assumes `f(a)` and `f(b)` straddle zero.
fn bisect<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> f64 {
    let mut fa = f(a);
    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if !fm.is_finite() {
            // shallow / invalid side — pull the bracket up toward deeper water
            a = m;
            fa = f(a);
            continue;
        }
        if (b - a).abs() < tol {
            return m;
        }
        if (fa <= 0.0) == (fm <= 0.0) {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// Critical water-surface elevation: the wse where Fr = 1, i.e. Q^2 * T / (g * A^3) = 1.
/// The function decreases monotonically with depth, so we bracket and bisect.
pub fn critical_wse(xs: &CrossSection, q: f64) -> Option<f64> {
    let zmin = xs.min_elevation();
    let f = |wse: f64| -> f64 {
        let p = xs.props(wse);
        if p.area <= 0.0 || p.top_width <= 0.0 {
            return 1.0e12; // very shallow => Fr^2 >> 1
        }
        q * q * p.top_width / (G * p.area.powi(3)) - 1.0
    };
    let a = zmin + 1.0e-6;
    let mut b = zmin + 0.1;
    for _ in 0..200 {
        if f(b) < 0.0 {
            return Some(bisect(f, a, b, 1.0e-8, 300));
        }
        b = zmin + (b - zmin) * 2.0;
        if b - zmin > 1.0e6 {
            break;
        }
    }
    None
}

/// Normal (uniform-flow) water-surface elevation: solve K(wse) * sqrt(slope) = Q.
/// Conveyance increases monotonically with depth.
pub fn normal_wse(xs: &CrossSection, q: f64, slope: f64) -> Option<f64> {
    if slope <= 0.0 {
        return None;
    }
    let zmin = xs.min_elevation();
    let sqrt_s = slope.sqrt();
    let f = |wse: f64| -> f64 { xs.conveyance(wse) * sqrt_s - q };
    let a = zmin + 1.0e-6;
    let mut b = zmin + 0.1;
    for _ in 0..200 {
        if f(b) > 0.0 {
            return Some(bisect(f, a, b, 1.0e-8, 300));
        }
        b = zmin + (b - zmin) * 2.0;
        if b - zmin > 1.0e6 {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CrossSection;

    fn wide_rect() -> CrossSection {
        // 20 m wide, 10 m tall walls
        CrossSection::new(vec![0.0, 0.0, 20.0, 20.0], vec![10.0, 0.0, 0.0, 10.0], 0.03)
    }

    #[test]
    fn critical_depth_rectangle() {
        // For a rectangle, yc = (q^2 / g)^(1/3) with q = Q/width (unit discharge).
        let xs = wide_rect();
        let q = 40.0;
        let width = 20.0;
        let unit_q = q / width;
        let yc_analytic = (unit_q * unit_q / G).powf(1.0 / 3.0);
        let yc = critical_wse(&xs, q).unwrap() - xs.min_elevation();
        assert!((yc - yc_analytic).abs() < 1e-4, "yc {} vs {}", yc, yc_analytic);
        // sanity: Froude at critical depth ~ 1
        assert!((froude(&xs, q, xs.min_elevation() + yc) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn normal_depth_matches_manning() {
        let xs = wide_rect();
        let q = 40.0;
        let slope = 0.001;
        let yn = normal_wse(&xs, q, slope).unwrap() - xs.min_elevation();
        // back-substitute into Manning: Q ?= K(yn) sqrt(S)
        let q_check = xs.conveyance(xs.min_elevation() + yn) * slope.sqrt();
        assert!((q_check - q).abs() < 1e-4, "Q {} vs {}", q_check, q);
    }
}
