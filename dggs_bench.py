"""
DGGS (H3) hash-join vs exact point-in-polygon: speed AND accuracy.

Question: when flood pixels are aggregated to administrative units (e.g. Bangladesh
unions), how much faster is an H3 cell-ID equality join than an exact
point-in-polygon join -- and how much assignment error does it introduce at
polygon boundaries, at each H3 resolution?

Usage
  # synthetic test (no data needed)
  python dggs_bench.py --synthetic --n-points 1000000

  # real data: union polygons + a binary flood GeoTIFF (1 = flooded)
  python dggs_bench.py --polygons unions.gpkg --id-col shapeID \
                       --raster sylhet_flood_2022.tif --resolutions 7 8 9 10
"""
import argparse
import time

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

WGS84 = "EPSG:4326"


# ---------------------------------------------------------------- inputs
def synthetic_inputs(n_points, n_polys, seed=42):
    """Voronoi 'unions' over the Sylhet region + random 'flood pixels'."""
    rng = np.random.default_rng(seed)
    bbox = box(91.0, 24.2, 92.5, 25.2)
    seeds = shapely.multipoints(
        np.column_stack([rng.uniform(91.0, 92.5, n_polys),
                         rng.uniform(24.2, 25.2, n_polys)]))
    cells = shapely.get_parts(shapely.voronoi_polygons(seeds, extend_to=bbox))
    cells = shapely.intersection(cells, bbox)
    polys = gpd.GeoDataFrame({"poly_id": np.arange(len(cells))},
                             geometry=cells, crs=WGS84)
    pts = pd.DataFrame({"lon": rng.uniform(91.0, 92.5, n_points),
                        "lat": rng.uniform(24.2, 25.2, n_points)})
    return polys, pts


def raster_to_points(path, flood_value=1, max_points=None, seed=42):
    """Centroids of flooded pixels, in WGS84."""
    import rasterio
    from rasterio.warp import transform as warp
    with rasterio.open(path) as src:
        arr = src.read(1)
        rows, cols = np.nonzero(arr == flood_value)
        if max_points and len(rows) > max_points:
            idx = np.random.default_rng(seed).choice(len(rows), max_points, replace=False)
            rows, cols = rows[idx], cols[idx]
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        if src.crs and src.crs.to_string() != WGS84:
            xs, ys = warp(src.crs, WGS84, xs, ys)
    return pd.DataFrame({"lon": np.asarray(xs), "lat": np.asarray(ys)})


# ---------------------------------------------------------------- methods
def exact_pip(polys, pts):
    """Baseline: exact point-in-polygon with an STRtree spatial index."""
    t0 = time.perf_counter()
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=WGS84)
    j = gpd.sjoin(g, polys[["poly_id", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]          # a point on a shared edge can match twice
    return j["poly_id"].to_numpy(), time.perf_counter() - t0


def encode_points(pts, res):
    return np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, res))
                        for la, lo in zip(pts.lat.to_numpy(), pts.lon.to_numpy())),
                       dtype=np.uint64, count=len(pts))


def encode_polys(polys, res):
    """Polygon -> H3 cells whose centre falls inside (H3 v4 'center' containment)."""
    rows = []
    for pid, geom in zip(polys.poly_id, polys.geometry):
        for part in getattr(geom, "geoms", [geom]):
            for c in h3.geo_to_cells(part.__geo_interface__, res):
                rows.append((h3.str_to_int(c), pid))
    df = pd.DataFrame(rows, columns=["cell", "poly_id"])
    df["cell"] = df["cell"].astype("uint64")
    return df.drop_duplicates("cell")    # a cell belongs to one polygon only


def h3_join(pt_cells, poly_cells):
    """Pure equality hash-join in DuckDB -- no geometry involved."""
    con = duckdb.connect()
    con.register("p", pd.DataFrame({"i": np.arange(len(pt_cells)), "cell": pt_cells}))
    con.register("c", poly_cells)
    t0 = time.perf_counter()
    out = con.execute("SELECT p.i, c.poly_id FROM p LEFT JOIN c USING (cell) ORDER BY p.i").df()
    return out["poly_id"].to_numpy(dtype=float), time.perf_counter() - t0


# ---------------------------------------------------------------- experiment
def run(polys, pts, resolutions):
    print(f"{len(pts):,} points, {len(polys):,} polygons")
    truth, t_exact = exact_pip(polys, pts)
    truth = truth.astype(float)
    print(f"exact point-in-polygon: {t_exact:.2f}s")

    true_counts = pd.Series(truth).value_counts()
    results = []
    for res in resolutions:
        t0 = time.perf_counter(); pc = encode_points(pts, res); t_ep = time.perf_counter() - t0
        t0 = time.perf_counter(); gc = encode_polys(polys, res); t_eg = time.perf_counter() - t0
        pred, t_join = h3_join(pc, gc)

        inside = ~np.isnan(truth)
        agree = np.mean(pred[inside] == truth[inside])
        unassigned = np.mean(np.isnan(pred[inside]))
        h3_counts = pd.Series(pred).value_counts().reindex(true_counts.index, fill_value=0)
        ape = (h3_counts - true_counts).abs() / true_counts
        results.append(dict(
            h3_res=res,
            edge_km=round(h3.average_hexagon_edge_length(res, "km"), 3),
            encode_points_s=round(t_ep, 3), encode_polys_s=round(t_eg, 3),
            join_s=round(t_join, 3),
            total_s=round(t_ep + t_eg + t_join, 3),
            speedup_preindexed=round(t_exact / t_join, 1),
            point_agreement_pct=round(float(100 * agree), 3),
            unassigned_pct=round(float(100 * unassigned), 3),
            median_unit_count_err_pct=round(float(100 * ape.median()), 3),
            p95_unit_count_err_pct=round(float(100 * ape.quantile(0.95)), 3),
        ))
        print(results[-1])
    df = pd.DataFrame(results)
    df.insert(0, "exact_pip_s", round(t_exact, 3))
    return df


def plot(df, path):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df.h3_res, df.join_s, "o-", label="H3 join (pre-indexed)")
    ax1.plot(df.h3_res, df.total_s, "s--", label="H3 incl. encoding")
    ax1.axhline(df.exact_pip_s.iloc[0], color="grey", ls=":", label="exact point-in-polygon")
    ax1.set_yscale("log"); ax1.set_xlabel("H3 resolution"); ax1.set_ylabel("seconds (log)")
    ax2 = ax1.twinx()
    ax2.plot(df.h3_res, 100 - df.point_agreement_pct, "^-", color="tab:red", label="point mis-assignment %")
    ax2.set_ylabel("mis-assigned points (%)", color="tab:red")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax1.set_title("Speed vs boundary error: H3 hash-join vs exact PIP")
    fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-points", type=int, default=1_000_000)
    ap.add_argument("--n-polys", type=int, default=300)
    ap.add_argument("--polygons"); ap.add_argument("--id-col", default="shapeID")
    ap.add_argument("--raster"); ap.add_argument("--flood-value", type=int, default=1)
    ap.add_argument("--max-points", type=int)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.synthetic:
        polys, pts = synthetic_inputs(a.n_points, a.n_polys)
    else:
        polys = gpd.read_file(a.polygons).to_crs(WGS84)
        polys = polys.rename(columns={a.id_col: "poly_id"})[["poly_id", "geometry"]]
        polys["poly_id"] = pd.factorize(polys["poly_id"])[0]
        pts = raster_to_points(a.raster, a.flood_value, a.max_points)
        xmin, ymin, xmax, ymax = pts.lon.min(), pts.lat.min(), pts.lon.max(), pts.lat.max()
        polys = polys.cx[xmin:xmax, ymin:ymax]          # only units the flood touches

    df = run(polys, pts, a.resolutions)
    df.to_csv(f"{a.out}.csv", index=False)
    plot(df, f"{a.out}.png")
    print(df.to_string(index=False))
