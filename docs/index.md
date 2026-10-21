---
title: VIFDAHL fire detections – data files
---

# VIFDAHL fire detections: data files

VIFDAHL (VIIRS I-band Fire Detection Algorithm for High Latitudes) detects
active fire in VIIRS 375 m I-band data (I4, 3.7 µm; I5, 11 µm), with
thresholds tuned for boreal and Arctic fires. This page describes the
GeoPackage files from the current version, run on NASA L1B swath data
(VNP02IMG/VNP03IMG and the NOAA-20/21 equivalents).

**Status: preliminary.** These files are for evaluation. Formats and
thresholds may still change before a versioned release.

## Downloads

| Archive | Local dates | AOIs | Daily files | Size |
|---|---|---|---|---|
| [`VIFDAHL_eastsiberia_2020.zip`](https://drive.google.com/file/d/150FDD8du2dkVXZO3Qq34DyqkGyt_1mJ6/view?usp=drive_link) | 2020-06-01 – 2020-09-30 | Bering_tundra, Northeast_Siberian_coastal_tundra, Northeast_Siberian_taiga | 201 | 171 MB |
| [`VIFDAHL_AK_boreal_2022.zip`](https://drive.google.com/file/d/1zf_B2qsoQjmQwLhne90WlPsbbWnr5VV-/view?usp=drive_link) | 2022-04-26 – 2022-09-25 | Boreal_AK_and_adjacent_CA | 129 | 56 MB |
| [`VIFDAHL_mukluk_fullrun.zip`](https://drive.google.com/file/d/1uCvF6K8tN6eyf5-VDgHdkRHFUEQf3_gS/view?usp=drive_link) | 2026-08-17 – 2026-09-10 | mukluk | 4 | 0.4 MB |

| Run | Platforms | AOI |
|---|---|---|
| eastsiberia_2020 | S-NPP | 2020 large-fire perimeters (TaLuCi), unioned per ecoregion |
| AK_boreal_2022 | S-NPP | Boreal Alaska and adjacent Canada |
| mukluk_fullrun | S-NPP, NOAA-20, NOAA-21 | 2026 Mukluk Fire, Alaska (box around the fire) |

## Archive layout

```
<run>/summary.csv
<run>/<aoi_id>/VIFDAHL_<aoi_id>_<YYYYMMDD>.gpkg
```

There is one GeoPackage per AOI and **local solar date**, and only for dates
with at least one detection. `summary.csv` lists every date, including dates
without fire.

## Conventions

- **Local solar date and period:** local mean solar time = UTC + longitude/15 h,
  taken at each pixel. `day` is 06–18 h. `night` is 18–06 h and is labeled
  with the date on which it starts.
- **Fire types:**
  - `H`: high-intensity fire.
  - `L`: low-intensity fire.
  - `A`: anomalous. I4 is saturated or folded, so TB4 is unreliable, but the
    pixel is very likely hot.

  Each pixel has exactly one type.
- **Coordinates:** all layers are in EPSG:4326. Pixels duplicated by the
  VIIRS scan overlap are removed.

## Layers

### `points` and `footprints`

Both layers have one feature per detected pixel and the same attributes.
`points` holds the pixel center, `footprints` the pixel outline.

| Field | Meaning |
|---|---|
| `Platform` | `NPP`, `N20`, `N21` |
| `Sensor`, `Bands` | `VIIRS`, `I-Bands` |
| `Orbit` | Orbit number |
| `Granule` | Granule start time, UTC (ISO 8601) |
| `DATE`, `GMT` | Granule start date and time (HHMM), UTC |
| `Type` | `H`, `L` or `A` |
| `AscNode` | `Y` for an ascending pass, `N` for descending |
| `LAT`, `LON` | Pixel center (degrees) |
| `TB4`, `TB5` | I4 and I5 brightness temperature (K) |
| `SolZenithAng`, `SatZenithAng` | Solar and satellite zenith angle (degrees) |
| `LocalDate`, `Period` | Local solar date (YYYY-MM-DD) and `day`/`night` |
| `row`, `col` | Pixel position in the granule |

### `fused`

Contiguous fire areas per local date and period:

1. Each footprint is buffered by +200 m and then −120 m in EPSG:6931, which
   rounds its corners.
2. The buffered footprints are merged per class: **H** contains H and A
   pixels, **L** contains L pixels minus the H area.

| Field | Meaning |
|---|---|
| `LocalDate`, `Period` | As above |
| `Platform` | `ALL` (all platforms fused together) |
| `Class` | `H` or `L` |
| `n_pixels`, `n_overpasses` | Member pixels and distinct overpasses |
| `platforms` | Contributing platforms |
| `first_utc`, `last_utc` | First and last granule start time |
| `TB4_max` | Hottest member TB4 (K) |
| `area_km2` | Area of the fused polygon (km²) |

## `summary.csv`

One row per AOI × local date × period × platform, plus a `Platform = ALL` row.

| Field | Meaning |
|---|---|
| `n_overpasses` | Overpasses covering the AOI. Counts are empty when there was no overpass, so "not observed" and "no fire" stay distinct. |
| `overpass_source` | `orbit`, or `time_gap` if overpasses were grouped by time |
| `n_failed` | Granules that failed processing |
| `n_A`, `n_H`, `n_L`, `n_detections` | Pixel counts |
| `area_H_km2`, `area_L_km2` | Fused area (`ALL` rows only) |
| `TB4_max` | Hottest TB4 (K) |

## Processing settings

| Setting | Value |
|---|---|
| I5 threshold for the special warm test (`min_i5_special`) | 305 K |
| Day/night selection |  `nadir` |
| Anomalous I4 rule (L1B) | I4 Saturation flag, or Out_of_Range flag with TB5 > 300 K |
| Code version | commit eee1fe0 |

Day/night threshold selection options are

- `scene` : a scene is rejected if it has pixels with sun zenith angle SZA > 90° and SZA ≤ 90°. Every scene is entirely either day or night 
- `pixel`: each pixel is  individually classified by SZA
- `nadir`: classification goes by nadir, pixels at the edge that belong to the other time of day are rejected 

## Reference and contact

The algorithm is described in:

- Waigl, C. F., Stuefer, M., Prakash, A., and Ichoku, C. (2017). Detecting
  high and low-intensity fires in Alaska using VIIRS I-band data: An
  improved operational approach for high latitudes. *Remote Sensing of
  Environment*, 199, 389–400.

Contact: Chris Waigl, International Arctic Research Center, University of 
Alaska Fairbanks (cwaigl@alaska.edu).
