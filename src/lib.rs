//! HEC-RAS — a from-scratch hydraulic river-analysis engine in Rust.
//!
//! Phase 1 implements 1D steady gradually-varied flow (standard-step backwater).
//! Later phases add 1D unsteady (Saint-Venant) and 2D shallow-water solvers.

pub mod geometry;
pub mod hydraulics;
pub mod steady;

// The Python bindings are gated behind the `extension-module` feature so that
// `cargo test --no-default-features` compiles the pure-Rust core without linking
// libpython. maturin builds with the feature on.
#[cfg(feature = "extension-module")]
mod python_bindings {
use crate::geometry::CrossSection as RsCrossSection;
use crate::{hydraulics, steady};
use numpy::PyArray1;
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

#[pymodule]
fn _hecras(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrossSection>()?;
    m.add_function(wrap_pyfunction!(steady_profile, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
} // mod python_bindings
