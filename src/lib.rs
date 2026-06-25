//! HEC-RAS — a from-scratch hydraulic river-analysis engine in Rust.
//!
//! Phase 1 implements 1D steady gradually-varied flow (standard-step backwater).
//! Later phases add 1D unsteady (Saint-Venant) and 2D shallow-water solvers.

pub mod geometry;
pub mod hydraulics;
pub mod steady;
pub mod swe2d;
pub mod unsteady;

// The Python bindings are gated behind the `extension-module` feature so that
// `cargo test --no-default-features` compiles the pure-Rust core without linking
// libpython. maturin builds with the feature on.
#[cfg(feature = "extension-module")]
mod python_bindings {
use crate::geometry::CrossSection as RsCrossSection;
use crate::swe2d::{self, Bc};
use crate::unsteady::{self, DownstreamBc};
use crate::{hydraulics, steady};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A river cross section defined by station/elevation points and a Manning's n.
#[pyclass(name = "CrossSection")]
#[derive(Clone)]
struct PyCrossSection {
    inner: RsCrossSection,
}

#[pymethods]
impl PyCrossSection {
    #[new]
    fn new(stations: Vec<f64>, elevations: Vec<f64>, manning_n: f64) -> PyResult<Self> {
        if stations.len() != elevations.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stations and elevations must have equal length",
            ));
        }
        if stations.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "need at least two points",
            ));
        }
        if manning_n <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "manning_n must be positive",
            ));
        }
        Ok(Self {
            inner: RsCrossSection::new(stations, elevations, manning_n),
        })
    }

    /// Wetted area (m^2) at a water-surface elevation.
    fn area(&self, wse: f64) -> f64 {
        self.inner.props(wse).area
    }

    /// Wetted perimeter (m).
    fn wetted_perimeter(&self, wse: f64) -> f64 {
        self.inner.props(wse).wetted_perimeter
    }

    /// Water-surface top width (m).
    fn top_width(&self, wse: f64) -> f64 {
        self.inner.props(wse).top_width
    }

    /// Hydraulic radius R = A / P (m).
    fn hydraulic_radius(&self, wse: f64) -> f64 {
        self.inner.hydraulic_radius(wse)
    }

    /// Manning conveyance K, so that Q = K * sqrt(Sf).
    fn conveyance(&self, wse: f64) -> f64 {
        self.inner.conveyance(wse)
    }

    /// Froude number for discharge `q` at water-surface `wse`.
    fn froude(&self, q: f64, wse: f64) -> f64 {
        hydraulics::froude(&self.inner, q, wse)
    }

    /// Critical depth (m above thalweg) for discharge `q`.
    fn critical_depth(&self, q: f64) -> Option<f64> {
        hydraulics::critical_wse(&self.inner, q).map(|w| w - self.inner.min_elevation())
    }

    /// Critical water-surface elevation (m) for discharge `q`.
    fn critical_wse(&self, q: f64) -> Option<f64> {
        hydraulics::critical_wse(&self.inner, q)
    }

    /// Normal (uniform-flow) depth (m above thalweg) for discharge `q` and bed slope.
    fn normal_depth(&self, q: f64, slope: f64) -> Option<f64> {
        hydraulics::normal_wse(&self.inner, q, slope).map(|w| w - self.inner.min_elevation())
    }

    /// Normal water-surface elevation (m).
    fn normal_wse(&self, q: f64, slope: f64) -> Option<f64> {
        hydraulics::normal_wse(&self.inner, q, slope)
    }

    /// Lowest bed elevation (thalweg, m).
    #[getter]
    fn min_elevation(&self) -> f64 {
        self.inner.min_elevation()
    }

    fn __repr__(&self) -> String {
        format!(
            "CrossSection(npoints={}, manning_n={})",
            self.inner.stations.len(),
            self.inner.manning_n
        )
    }
}

/// Compute a subcritical steady-flow water-surface profile by the standard-step method.
///
/// `sections` are ordered downstream -> upstream; `sections[0]` carries the boundary
/// control `downstream_wse`. `reach_lengths` has length len(sections) - 1, giving the
/// channel distance between successive sections. Returns a dict of numpy arrays
/// (one value per section): wse, depth, velocity, froude, energy_grade, area,
/// top_width, friction_slope, converged.
#[pyfunction]
#[pyo3(signature = (sections, reach_lengths, q, downstream_wse, alpha=1.0, contraction=0.1, expansion=0.3))]
#[allow(clippy::too_many_arguments)]
fn steady_profile<'py>(
    py: Python<'py>,
    sections: Vec<PyCrossSection>,
    reach_lengths: Vec<f64>,
    q: f64,
    downstream_wse: f64,
    alpha: f64,
    contraction: f64,
    expansion: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if sections.len() >= 2 && reach_lengths.len() != sections.len() - 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "reach_lengths must have length len(sections)-1 = {}, got {}",
            sections.len() - 1,
            reach_lengths.len()
        )));
    }
    let xss: Vec<RsCrossSection> = sections.into_iter().map(|s| s.inner).collect();
    let states = steady::steady_profile(
        &xss,
        &reach_lengths,
        q,
        downstream_wse,
        alpha,
        contraction,
        expansion,
    );

    let wse: Vec<f64> = states.iter().map(|s| s.wse).collect();
    let depth: Vec<f64> = states.iter().map(|s| s.depth).collect();
    let velocity: Vec<f64> = states.iter().map(|s| s.velocity).collect();
    let froude: Vec<f64> = states.iter().map(|s| s.froude).collect();
    let energy: Vec<f64> = states.iter().map(|s| s.energy_grade).collect();
    let area: Vec<f64> = states.iter().map(|s| s.area).collect();
    let top_width: Vec<f64> = states.iter().map(|s| s.top_width).collect();
    let friction_slope: Vec<f64> = states.iter().map(|s| s.friction_slope).collect();
    let converged: Vec<bool> = states.iter().map(|s| s.converged).collect();

    let d = PyDict::new_bound(py);
    d.set_item("wse", PyArray1::from_vec_bound(py, wse))?;
    d.set_item("depth", PyArray1::from_vec_bound(py, depth))?;
    d.set_item("velocity", PyArray1::from_vec_bound(py, velocity))?;
    d.set_item("froude", PyArray1::from_vec_bound(py, froude))?;
    d.set_item("energy_grade", PyArray1::from_vec_bound(py, energy))?;
    d.set_item("area", PyArray1::from_vec_bound(py, area))?;
    d.set_item("top_width", PyArray1::from_vec_bound(py, top_width))?;
    d.set_item("friction_slope", PyArray1::from_vec_bound(py, friction_slope))?;
    d.set_item("converged", converged)?;
    Ok(d)
}

/// Route 1D unsteady flow through a reach with the Preissmann implicit scheme.
///
/// `sections` are ordered upstream -> downstream; the upstream discharge hydrograph
/// `inflow_q` (length n_steps + 1) is applied at index 0. `reach_lengths` has length
/// len(sections) - 1. `initial_stage` / `initial_q` are the state at t = 0 (length N).
///
/// The downstream boundary is either `"normal"` (normal-depth rating, requires
/// `downstream_slope`) or `"stage"` (prescribed water-surface, `downstream_stage` may
/// be a scalar or a length n_steps+1 series).
///
/// Returns a dict of numpy arrays: `time` (n_steps+1), `stage` and `discharge`
/// (2D, [time, node]), `inflow`, `outflow`, `max_residual`, `converged`.
#[pyfunction]
#[pyo3(signature = (sections, reach_lengths, initial_stage, initial_q, inflow_q, dt,
    n_steps, theta=0.6, downstream="normal", downstream_slope=None, downstream_stage=None,
    newton_tol=1e-7, max_newton=50))]
#[allow(clippy::too_many_arguments)]
fn route_unsteady<'py>(
    py: Python<'py>,
    sections: Vec<PyCrossSection>,
    reach_lengths: Vec<f64>,
    initial_stage: Vec<f64>,
    initial_q: Vec<f64>,
    inflow_q: Vec<f64>,
    dt: f64,
    n_steps: usize,
    theta: f64,
    downstream: &str,
    downstream_slope: Option<f64>,
    downstream_stage: Option<Vec<f64>>,
    newton_tol: f64,
    max_newton: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = sections.len();
    let err = |s: String| pyo3::exceptions::PyValueError::new_err(s);
    if n < 2 {
        return Err(err("need at least two sections".into()));
    }
    if reach_lengths.len() != n - 1 {
        return Err(err(format!(
            "reach_lengths must have length {}, got {}",
            n - 1,
            reach_lengths.len()
        )));
    }
    if initial_stage.len() != n || initial_q.len() != n {
        return Err(err("initial_stage and initial_q must have length len(sections)".into()));
    }
    if inflow_q.len() != n_steps + 1 {
        return Err(err(format!(
            "inflow_q must have length n_steps+1 = {}, got {}",
            n_steps + 1,
            inflow_q.len()
        )));
    }

    let bc = match downstream {
        "normal" => {
            let s = downstream_slope
                .ok_or_else(|| err("downstream='normal' requires downstream_slope".into()))?;
            DownstreamBc::Normal(s)
        }
        "stage" => {
            let raw = downstream_stage
                .ok_or_else(|| err("downstream='stage' requires downstream_stage".into()))?;
            let series = if raw.len() == 1 {
                vec![raw[0]; n_steps + 1]
            } else if raw.len() == n_steps + 1 {
                raw
            } else {
                return Err(err(format!(
                    "downstream_stage must be scalar or length n_steps+1 = {}",
                    n_steps + 1
                )));
            };
            DownstreamBc::Stage(series)
        }
        other => return Err(err(format!("unknown downstream BC '{}'", other))),
    };

    let xss: Vec<RsCrossSection> = sections.into_iter().map(|s| s.inner).collect();
    let res = unsteady::route_unsteady(
        &xss,
        &reach_lengths,
        &initial_stage,
        &initial_q,
        &inflow_q,
        dt,
        n_steps,
        theta,
        &bc,
        newton_tol,
        max_newton,
    );

    let rows = res.nsteps + 1;
    let stage = Array2::from_shape_vec((rows, res.n), res.stage)
        .map_err(|e| err(e.to_string()))?;
    let discharge = Array2::from_shape_vec((rows, res.n), res.discharge)
        .map_err(|e| err(e.to_string()))?;
    let time: Vec<f64> = (0..rows).map(|k| k as f64 * dt).collect();
    let inflow: Vec<f64> = (0..rows).map(|k| discharge[[k, 0]]).collect();
    let outflow: Vec<f64> = (0..rows).map(|k| discharge[[k, res.n - 1]]).collect();

    let d = PyDict::new_bound(py);
    d.set_item("time", PyArray1::from_vec_bound(py, time))?;
    d.set_item("stage", PyArray2::from_owned_array_bound(py, stage))?;
    d.set_item("discharge", PyArray2::from_owned_array_bound(py, discharge))?;
    d.set_item("inflow", PyArray1::from_vec_bound(py, inflow))?;
    d.set_item("outflow", PyArray1::from_vec_bound(py, outflow))?;
    d.set_item("max_residual", PyArray1::from_vec_bound(py, res.max_residual))?;
    d.set_item("converged", res.converged)?;
    Ok(d)
}

/// Run the 2D shallow-water model on a Cartesian grid.
///
/// `bed`, `h0`, `hu0`, `hv0` are flat row-major fields of length nx*ny (index j*nx + i).
/// `bc` is `"transmissive"` (open) or `"reflective"` (wall). Returns a dict with the
/// final `h`, `hu`, `hv` as 2D arrays [ny, nx], plus `time` and `volume` diagnostic
/// series, `steps` and `t_final`.
#[pyfunction]
#[pyo3(signature = (nx, ny, dx, dy, bed, h0, hu0, hv0, manning_n=0.0, t_end=10.0,
    cfl=0.45, bc="transmissive", max_steps=1_000_000, max_dt=f64::INFINITY,
    bc_xlow=None, bc_xhigh=None, bc_ylow=None, bc_yhigh=None,
    source_idx=vec![], source_t=vec![], source_q=vec![]))]
#[allow(clippy::too_many_arguments)]
fn run_swe2d<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    bed: Vec<f64>,
    h0: Vec<f64>,
    hu0: Vec<f64>,
    hv0: Vec<f64>,
    manning_n: f64,
    t_end: f64,
    cfl: f64,
    bc: &str,
    max_steps: usize,
    max_dt: f64,
    bc_xlow: Option<&str>,
    bc_xhigh: Option<&str>,
    bc_ylow: Option<&str>,
    bc_yhigh: Option<&str>,
    source_idx: Vec<usize>,
    source_t: Vec<f64>,
    source_q: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let err = |s: String| pyo3::exceptions::PyValueError::new_err(s);
    let need = nx * ny;
    for (name, v) in [("bed", &bed), ("h0", &h0), ("hu0", &hu0), ("hv0", &hv0)] {
        if v.len() != need {
            return Err(err(format!("{} must have length nx*ny = {}, got {}", name, need, v.len())));
        }
    }
    let parse_bc = |s: &str| -> PyResult<Bc> {
        match s {
            "transmissive" | "open" => Ok(Bc::Transmissive),
            "reflective" | "wall" => Ok(Bc::Reflective),
            other => Err(err(format!("unknown bc '{}'", other))),
        }
    };
    // per-edge BCs, each defaulting to the general `bc`
    let bc = [
        parse_bc(bc_xlow.unwrap_or(bc))?,
        parse_bc(bc_xhigh.unwrap_or(bc))?,
        parse_bc(bc_ylow.unwrap_or(bc))?,
        parse_bc(bc_yhigh.unwrap_or(bc))?,
    ];

    for &s in &source_idx {
        if s >= need {
            return Err(err(format!("source_idx {} out of range (nx*ny={})", s, need)));
        }
    }

    let res = swe2d::run_swe2d(
        nx, ny, dx, dy, &bed, &h0, &hu0, &hv0, manning_n, t_end, cfl, bc, max_steps, max_dt,
        &source_idx, &source_t, &source_q,
    );

    let to2d = |v: Vec<f64>| -> PyResult<_> {
        Array2::from_shape_vec((ny, nx), v).map_err(|e| err(e.to_string()))
    };
    let d = PyDict::new_bound(py);
    d.set_item("h", PyArray2::from_owned_array_bound(py, to2d(res.h)?))?;
    d.set_item("hu", PyArray2::from_owned_array_bound(py, to2d(res.hu)?))?;
    d.set_item("hv", PyArray2::from_owned_array_bound(py, to2d(res.hv)?))?;
    d.set_item("h_max", PyArray2::from_owned_array_bound(py, to2d(res.h_max)?))?;
    d.set_item("time", PyArray1::from_vec_bound(py, res.times))?;
    d.set_item("volume", PyArray1::from_vec_bound(py, res.volumes))?;
    d.set_item("inflow_vol", PyArray1::from_vec_bound(py, res.inflow_vol))?;
    d.set_item("steps", res.steps)?;
    d.set_item("t_final", res.t_final)?;
    Ok(d)
}

#[pymodule]
fn _hecras(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrossSection>()?;
    m.add_function(wrap_pyfunction!(steady_profile, m)?)?;
    m.add_function(wrap_pyfunction!(route_unsteady, m)?)?;
    m.add_function(wrap_pyfunction!(run_swe2d, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
} // mod python_bindings
