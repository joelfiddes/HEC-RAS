"""HEC-RAS: a from-scratch hydraulic river-analysis engine (Rust core, Python API).

Phase 1 provides 1D steady gradually-varied flow:

    >>> from hecras import CrossSection, steady_profile
    >>> xs = CrossSection([0, 0, 10, 10], [5, 0, 0, 5], 0.03)
    >>> xs.normal_depth(50.0, 0.001)
"""

from ._hecras import (
    CrossSection,
    steady_profile,
    route_unsteady,
    run_swe2d,
    __version__,
)

__all__ = [
    "CrossSection",
    "steady_profile",
    "route_unsteady",
    "run_swe2d",
    "__version__",
]
