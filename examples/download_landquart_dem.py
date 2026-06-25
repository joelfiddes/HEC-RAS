"""Download + mosaic the SwissALTI3D terrain for the Landquart (Felsenbach) reach.

Pulls the 2 m swisstopo SwissALTI3D tiles covering a 3 x 2 km block around the BAFU
gauge 2150 (LV95 2765384 / 1204914), mosaics them, block-averages to a 10 m grid, and
writes data/landquart/dem10.npy + dem10.json for examples/landquart_flood.py.

Public swisstopo STAC API; no authentication required. Requires rasterio.
"""

import json
import os
import re
import urllib.request

import numpy as np
import rasterio
from rasterio.merge import merge

HERE = os.path.dirname(__file__)
OUT = os.path.join(os.path.dirname(HERE), "data", "landquart")
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"

# LV95 km tiles covering the reach (easting 2764-2766, northing 1204-1205)
TILES = [(e, n) for e in (2764, 2765, 2766) for n in (1204, 1205)]
DOWNSAMPLE = 5  # 2 m -> 10 m


def tile_hrefs():
    """Map (E,N) tile -> 2 m GeoTIFF URL via two STAC bbox queries."""
    hrefs = {}
    for bbox in ("9.60,46.972,9.66,47.000", "9.60,46.980,9.66,47.012"):
        url = f"{STAC}?bbox={bbox}&limit=100"
        for f in json.load(urllib.request.urlopen(url, timeout=60))["features"]:
            m = re.search(r"_(\d{4})-(\d{4})", f["id"])
            key = (int(m.group(1)), int(m.group(2))) if m else None
            if key not in TILES:
                continue
            for k, a in f["assets"].items():
                if k.endswith("_2_2056_5728.tif"):
                    hrefs[key] = a["href"]
    return hrefs


def main():
    os.makedirs(OUT, exist_ok=True)
    hrefs = tile_hrefs()
    missing = set(TILES) - set(hrefs)
    if missing:
        raise SystemExit(f"STAC did not return tiles: {sorted(missing)}")

    paths = []
    for key, href in sorted(hrefs.items()):
        p = os.path.join(OUT, f"{key[0]}-{key[1]}.tif")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            print("downloading", os.path.basename(p))
            urllib.request.urlretrieve(href, p)
        paths.append(p)

    mosaic, tr = merge([rasterio.open(p) for p in paths])
    dem = mosaic[0].astype(float)
    f = DOWNSAMPLE
    ny = (dem.shape[0] // f) * f
    nx = (dem.shape[1] // f) * f
    dem10 = dem[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))

    np.save(os.path.join(OUT, "dem10.npy"), dem10)
    meta = dict(E0=float(tr.c), N0=float(tr.f), res=float(tr.a * f),
                nrows=int(dem10.shape[0]), ncols=int(dem10.shape[1]))
    json.dump(meta, open(os.path.join(OUT, "dem10.json"), "w"))
    print(f"wrote dem10.npy {dem10.shape} @ {meta['res']:.0f} m, "
          f"elev {dem10.min():.0f}-{dem10.max():.0f} m")


if __name__ == "__main__":
    main()
