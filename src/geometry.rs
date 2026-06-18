//! Cross-section geometry: hydraulic properties from station-elevation data.
//!
//! A cross section is a sequence of (station, elevation) points looking downstream,
//! `station` is the lateral distance across the section and `elevation` is the bed
//! level (geodetic, metres). A single Manning's roughness applies to the whole section.

/// Wetted hydraulic properties of a section at a given water-surface elevation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HydraulicProps {
    pub area: f64,
    pub wetted_perimeter: f64,
    pub top_width: f64,
}

/// A river cross section defined by station/elevation points.
#[derive(Clone, Debug)]
pub struct CrossSection {
    pub stations: Vec<f64>,
    pub elevations: Vec<f64>,
    pub manning_n: f64,
}

impl CrossSection {
    pub fn new(stations: Vec<f64>, elevations: Vec<f64>, manning_n: f64) -> Self {
        assert_eq!(
            stations.len(),
            elevations.len(),
            "stations and elevations must have equal length"
        );
        assert!(stations.len() >= 2, "need at least two points");
        assert!(manning_n > 0.0, "manning_n must be positive");
        CrossSection {
            stations,
            elevations,
            manning_n,
        }
    }

    /// Lowest bed elevation in the section (thalweg).
    pub fn min_elevation(&self) -> f64 {
        self.elevations.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Highest point in the section (top of banks).
    pub fn max_elevation(&self) -> f64 {
        self.elevations
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Wetted area, perimeter and top width at water-surface elevation `wse`.
    ///
    /// Each panel between consecutive points is clipped at the water line; a panel
    /// can be fully dry, fully wet, or partially wet (one end above the water).
    pub fn props(&self, wse: f64) -> HydraulicProps {
        let mut p = HydraulicProps::default();
        for i in 0..self.stations.len() - 1 {
            let (x1, z1) = (self.stations[i], self.elevations[i]);
            let (x2, z2) = (self.stations[i + 1], self.elevations[i + 1]);
            let w = x2 - x1;
            if w < 0.0 {
                continue; // ignore overhangs (non-increasing stations)
            }
            if w == 0.0 {
                // vertical wall: no area or top width, but the submerged height
                // counts toward the wetted perimeter.
                let zlo = z1.min(z2);
                let zhi = z1.max(z2);
                p.wetted_perimeter += (wse.min(zhi) - zlo).max(0.0);
                continue;
            }
            let d1 = wse - z1; // water depth at left point (negative => dry)
            let d2 = wse - z2; // water depth at right point
            match (d1 > 0.0, d2 > 0.0) {
                (false, false) => {} // dry panel
                (true, true) => {
                    // fully submerged trapezoid
                    p.area += 0.5 * (d1 + d2) * w;
                    p.wetted_perimeter += (w * w + (z2 - z1).powi(2)).sqrt();
                    p.top_width += w;
                }
                (true, false) => {
                    // wet on the left, dry on the right; water line crosses the panel
                    let t = d1 / (d1 - d2); // fraction of width that is wet, in (0,1]
                    let ww = t * w;
                    p.area += 0.5 * d1 * ww;
                    p.wetted_perimeter += (ww * ww + (t * (z2 - z1)).powi(2)).sqrt();
                    p.top_width += ww;
                }
                (false, true) => {
                    // dry on the left, wet on the right
                    let t = d2 / (d2 - d1);
                    let ww = t * w;
                    p.area += 0.5 * d2 * ww;
                    p.wetted_perimeter += (ww * ww + (t * (z1 - z2)).powi(2)).sqrt();
                    p.top_width += ww;
                }
            }
        }
        p
    }

    /// Hydraulic radius R = A / P (0 when dry).
    pub fn hydraulic_radius(&self, wse: f64) -> f64 {
        let p = self.props(wse);
        if p.wetted_perimeter > 0.0 {
            p.area / p.wetted_perimeter
        } else {
            0.0
        }
    }

    /// Manning conveyance K = (1/n) * A * R^(2/3) (SI units), so that Q = K * sqrt(Sf).
    /// Whole-section (single subdivision) — exact for a simple prismatic channel.
    pub fn conveyance(&self, wse: f64) -> f64 {
        let p = self.props(wse);
        if p.area <= 0.0 || p.wetted_perimeter <= 0.0 {
            return 0.0;
        }
        let r = p.area / p.wetted_perimeter;
        (1.0 / self.manning_n) * p.area * r.powf(2.0 / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10 m wide rectangular channel with 5 m tall vertical walls.
    fn rect() -> CrossSection {
        CrossSection::new(
            vec![0.0, 0.0, 10.0, 10.0],
            vec![5.0, 0.0, 0.0, 5.0],
            0.03,
        )
    }

    #[test]
    fn rectangular_props_at_2m() {
        let xs = rect();
        let p = xs.props(2.0);
        assert!((p.area - 20.0).abs() < 1e-9, "area {}", p.area);
        // wetted perimeter = 2 walls (2 m each) + bed (10 m) = 14 m
        assert!(
            (p.wetted_perimeter - 14.0).abs() < 1e-9,
            "P {}",
            p.wetted_perimeter
        );
        assert!((p.top_width - 10.0).abs() < 1e-9, "T {}", p.top_width);
        assert!((xs.hydraulic_radius(2.0) - 20.0 / 14.0).abs() < 1e-9);
    }

    #[test]
    fn dry_section_is_zero() {
        let xs = rect();
        let p = xs.props(-1.0);
        assert_eq!(p.area, 0.0);
        assert_eq!(p.wetted_perimeter, 0.0);
        assert_eq!(xs.conveyance(-1.0), 0.0);
    }

    #[test]
    fn conveyance_matches_manning_rectangle() {
        let xs = rect();
        let wse = 2.0;
        let a = 20.0;
        let r: f64 = 20.0 / 14.0;
        let expected = (1.0 / 0.03) * a * r.powf(2.0 / 3.0);
        assert!((xs.conveyance(wse) - expected).abs() < 1e-9);
    }

    #[test]
    fn partial_panel_triangle() {
        // single V-notch: (0,2)-(2,0)-(4,2); at wse=1 each side is half-wet
        let xs = CrossSection::new(vec![0.0, 2.0, 4.0], vec![2.0, 0.0, 2.0], 0.03);
        let p = xs.props(1.0);
        // water surface meets each sloping bed at station 1 and station 3 => top width 2
        assert!((p.top_width - 2.0).abs() < 1e-9, "T {}", p.top_width);
        // area is a triangle of width 2 and depth 1 => 1.0
        assert!((p.area - 1.0).abs() < 1e-9, "area {}", p.area);
    }
}
