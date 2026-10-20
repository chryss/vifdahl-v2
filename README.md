# vifdahl

VIIRS I-band Fire Detection Algorithm for High Latitudes (VIFDAHL), modernized
(2026). Successor to the 2016–2019 code at github.com/chryss/VIFDAHL.

Inputs: VIIRS I4/I5 swath data, either NASA L1B NetCDF (VNP02IMG/VNP03IMG and
NOAA-20/21 equivalents, via `earthaccess`) or SDR HDF5 from GINA direct
broadcast (planned). Output: one GeoPackage per local solar date with layers
`points`, `footprints` and `fused` (H and L areas, L − H; day/night per date).

## Layout

- `src/vifdahl/algorithm.py` – threshold classification (format-agnostic)
- `src/vifdahl/swath.py` – scan-overlap duplicates, day/night, pixel footprints
- `src/vifdahl/detect.py` – run on one granule, tabulate detections
- `src/vifdahl/periods.py` – local solar date, day/night period
- `src/vifdahl/aggregate.py` – fuse footprints into H/L areas (EPSG:6931)
- `src/vifdahl/output.py` – layers, daily GeoPackage writer
- `src/vifdahl/readers/`: common swath schema; SDR and L1B readers
- `data/reference/`: fire perimeters, 2015 detections, 2019 Shovel Creek products
- `legacy/` (not checked in - rsynched on dev machines): old code, local only, not in git (see `legacy/README.md`)

## Setup

    pixi install
    pixi run test

Earthdata credentials: `~/.netrc` with an entry for `urs.earthdata.nasa.gov`.
