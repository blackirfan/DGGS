# DGGS hash-joins for flood exposure in Bangladesh: how much speed, how much error?

Aggregating flooded pixels to administrative units (Bangladesh has ~4,500 unions) is a
point-in-polygon problem I have repeatedly hit in flood-risk and union-level indicator work.
Discrete Global Grid Systems promise to replace geometric predicates with a simple
cell-ID equality join. This repo checks that promise on real data and asks the second
question: **what does the speed cost in boundary accuracy, at each H3 resolution?**

Inspired by L. Juhász, *Accelerating Point-in-Polygon Predicates via Algebraic Hash-Joins
and Discrete Global Grids at Scale* (ACM SIGSPATIAL 2026, arXiv:2607.08956).

## Data
- **Flood extent:** Sentinel-1 SAR flood map, Sylhet, June 2022 (my GEE workflow) — `[file name, pixel size]`
- **Units:** union boundaries (ADM4) — `[source, e.g. geoBoundaries BGD ADM4]`

## Method
1. Baseline: exact point-in-polygon with an STRtree index (GeoPandas `sjoin`).
2. H3: encode flood-pixel centroids and union polygons (centre containment) to H3 at
   resolutions 6–10, then run a pure equality join in DuckDB.
3. Report time for encoding and join separately (pre-indexed vs on-the-fly), plus:
   share of pixels assigned to the same union as the exact method, unassigned pixels,
   and per-union flood-count error (median and 95th percentile).

```bash
pip install -r requirements.txt
python dggs_bench.py --polygons unions.gpkg --id-col shapeID \
    --raster sylhet_flood_2022.tif --resolutions 6 7 8 9 10 --out results
```
`python dggs_bench.py --synthetic` runs on generated data, with no downloads.

## Results
`[paste results table + results.png]`

## What I learned
`[2–4 honest sentences in your own words: which resolution is "good enough" for union-level
flood counts, where encoding overhead dominates, and which unions (small, elongated, riverine
char areas?) suffer most at boundaries.]`

## Limitations
- Point encoding here is a Python loop over `h3-py`; the DuckDB `h3` extension or a
  vectorised encoder would shrink the encoding cost considerably.
- Polygons use H3 centre containment, so slivers smaller than a cell can vanish.
- Single event, single region.
