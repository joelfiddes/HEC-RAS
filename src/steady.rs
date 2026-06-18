//! Steady gradually-varied flow: the standard-step method.
//!
//! Subcritical flow is computed by marching upstream from a known downstream
//! water-surface elevation, balancing the energy equation between sections:
//!
//!   WSE_up + a*V_up^2/2g = WSE_dn + a*V_dn^2/2g + h_f + h_eddy
//!
//! with friction loss h_f = Sf_avg * L (average of the friction slopes Sf = (Q/K)^2)
//! and the eddy (contraction/expansion) loss h_eddy = C * |Vh_up - Vh_dn|.

use crate::geometry::CrossSection;
use crate::hydraulics::{critical_wse, froude, G};

/// Computed hydraulic state at one cross section.
#[derive(Clone, Copy, Debug)]
pub struct SectionState {
    pub wse: f64,
    pub depth: f64,
    pub velocity: f64,
    pub froude: f64,
    pub energy_grade: f64,
    pub area: f64,
    pub top_width: f64,
    pub friction_slope: f64,
    pub converged: bool,
}

fn make_state(xs: &CrossSection, wse: f64, q: f64, alpha: f64, converged: bool) -> SectionState {
    let p = xs.props(wse);
    let area = p.area;
    let v = if area > 0.0 { q / area } else { f64::NAN };
    let k = xs.conveyance(wse);
    let sf = if k > 0.0 { (q / k).powi(2) } else { f64::NAN };
    SectionState {
        wse,
        depth: wse - xs.min_elevation(),
        velocity: v,
        froude: froude(xs, q, wse),
        energy_grade: wse + alpha * v * v / (2.0 * G),
        area,
        top_width: p.top_width,
        friction_slope: sf,
        converged,
    }
}

/// Solve the energy equation for the upstream water-surface elevation given the
/// downstream one. Returns (wse_up, converged).
#[allow(clippy::too_many_arguments)]
pub fn standard_step_upstream(
    xs_dn: &CrossSection,
    wse_dn: f64,
    xs_up: &CrossSection,
    q: f64,
    reach_len: f64,
    alpha: f64,
    cc: f64,
    ce: f64,
) -> (f64, bool) {
    let p_dn = xs_dn.props(wse_dn);
    let v_dn = q / p_dn.area;
    let vh_dn = alpha * v_dn * v_dn / (2.0 * G);
    let eg_dn = wse_dn + vh_dn;
    let k_dn = xs_dn.conveyance(wse_dn);
    let sf_dn = (q / k_dn).powi(2);

    // Energy-balance residual as a function of the unknown upstream wse.
    let resid = |wse_up: f64| -> f64 {
        let p = xs_up.props(wse_up);
        if p.area <= 1.0e-9 {
            return f64::NAN;
        }
        let v = q / p.area;
        let vh = alpha * v * v / (2.0 * G);
        let k = xs_up.conveyance(wse_up);
        if k <= 0.0 {
            return f64::NAN;
        }
        let sf = (q / k).powi(2);
        let sf_avg = 0.5 * (sf + sf_dn);
        let hf = sf_avg * reach_len;
        // contraction when flow accelerates downstream (vh_dn > vh_up), else expansion
        let c = if vh_dn > vh { cc } else { ce };
        let h_eddy = c * (vh - vh_dn).abs();
        (wse_up + vh) - (eg_dn + hf + h_eddy)
    };

    // The subcritical root lies above critical depth (specific energy has its
    // minimum at critical depth, so we bracket from just above it upward).
    let zmin = xs_up.min_elevation();
    let wc = critical_wse(xs_up, q).unwrap_or(zmin);
    let lo0 = wc + 1.0e-4;
    let mut hi = wse_dn.max(lo0) + 0.5;
    let mut found = false;
    for _ in 0..200 {
        let r = resid(hi);
        if r.is_finite() && r > 0.0 {
            found = true;
            break;
        }
        hi += (hi - lo0).max(0.5);
        if hi - lo0 > 1.0e5 {
            break;
        }
    }
    if !found {
        return (hi, false);
    }

    // Bisection: resid is increasing through the subcritical root.
    let mut lo = lo0;
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        let fm = resid(mid);
        if !fm.is_finite() {
            lo = mid; // shallow side invalid, move up
            continue;
        }
        if fm > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1.0e-8 {
            break;
        }
    }
    (0.5 * (lo + hi), true)
}

/// Compute a subcritical water-surface profile along a reach.
///
/// `sections` are ordered from downstream (index 0, the control) to upstream.
/// `reach_lengths[i]` is the channel distance between section `i` and section `i+1`
/// (length `sections.len() - 1`). `downstream_wse` is the boundary control.
#[allow(clippy::too_many_arguments)]
pub fn steady_profile(
    sections: &[CrossSection],
    reach_lengths: &[f64],
    q: f64,
    downstream_wse: f64,
    alpha: f64,
    cc: f64,
    ce: f64,
) -> Vec<SectionState> {
    let n = sections.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut wse_prev = downstream_wse;
    out.push(make_state(&sections[0], wse_prev, q, alpha, true));
    for i in 1..n {
        let l = reach_lengths.get(i - 1).copied().unwrap_or(0.0);
        let (wse, conv) =
            standard_step_upstream(&sections[i - 1], wse_prev, &sections[i], q, l, alpha, cc, ce);
        out.push(make_state(&sections[i], wse, q, alpha, conv));
        wse_prev = wse;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CrossSection;
    use crate::hydraulics::normal_wse;

    fn wide_rect(zbed: f64) -> CrossSection {
        CrossSection::new(
            vec![0.0, 0.0, 50.0, 50.0],
            vec![zbed + 10.0, zbed, zbed, zbed + 10.0],
            0.03,
        )
    }

    #[test]
    fn flat_pool_stays_level_at_zero_flow_limit() {
        // At a downstream wse equal to normal depth on a prismatic reach, the
        // profile should stay close to normal depth (uniform flow).
        let q = 60.0;
        let slope = 0.0008;
        let dx = 100.0;
        let n_sec = 20;
        let sections: Vec<_> = (0..n_sec).map(|i| wide_rect(slope * dx * i as f64)).collect();
        let lengths = vec![dx; n_sec - 1];
        let yn = normal_wse(&sections[0], q, slope).unwrap() - sections[0].min_elevation();
        let down_wse = sections[0].min_elevation() + yn;
        let prof = steady_profile(&sections, &lengths, q, down_wse, 1.0, 0.1, 0.3);
        for s in &prof {
            assert!(s.converged);
            assert!((s.depth - yn).abs() < 1e-2, "depth {} vs yn {}", s.depth, yn);
            assert!(s.froude < 1.0, "should be subcritical, Fr={}", s.froude);
        }
    }
}
