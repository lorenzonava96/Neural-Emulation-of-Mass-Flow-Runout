import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError:
    gpd = None

import json
from matplotlib.colors import LightSource

def _extract_coords_from_kmz(kmz_path: str) -> Tuple[float, float]:
    if not os.path.exists(kmz_path):
        raise FileNotFoundError(kmz_path)
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_name = next((f for f in kmz.namelist() if f.lower().endswith(".kml")), None)
        if not kml_name:
            raise FileNotFoundError("No KML file found in KMZ.")
        root = ET.fromstring(kmz.read(kml_name))
        ns = {"kml": root.tag.split("}")[0].strip("{")}
        node = root.find(".//kml:coordinates", namespaces=ns)
        if node is None or not (node.text or "").strip():
            raise ValueError("No <coordinates> found in KML.")
        lon, lat, *_ = map(float, node.text.strip().split(","))
        return float(lat), float(lon)


def _extract_coords_from_shapefile(shapefile_path: str) -> Tuple[float, float]:
    if gpd is None:
        raise ImportError("geopandas is required for shapefile support")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(shapefile_path)
    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        raise ValueError("Shapefile is empty.")
    if not all(gdf.geometry.geom_type == "Point"):
        raise ValueError("Shapefile must contain only Point geometries.")
    gdf = gdf.to_crs("EPSG:4326")
    pt = gdf.geometry.iloc[0]
    return float(pt.y), float(pt.x)  # (lat, lon)


def _sample_chip(dem_path: str, center_lat: float, center_lon: float, size: int = 256):
    with rasterio.open(dem_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x_proj, y_proj = transformer.transform(center_lon, center_lat)
        row, col = src.index(x_proj, y_proj)

        half = size // 2
        window = Window(col - half, row - half, size, size).intersection(
            Window(0, 0, src.width, src.height)
        )

        chip = src.read(1, window=window)
        transform = src.window_transform(window)

        # Pad if near edges
        pad_h = size - chip.shape[0]
        pad_w = size - chip.shape[1]
        if pad_h > 0 or pad_w > 0:
            chip = np.pad(chip, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        profile = src.profile.copy()
        profile.update(height=size, width=size, transform=transform, count=1, driver="GTiff")

    return chip, transform, profile

def prepare_case_chip(
    case: str,
    root: str = ".",
    size: int = 256,
    kmz_path: Optional[str] = None,
    shp_path: Optional[str] = None,
    dem_path: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    save: bool = True,
    plot: bool = False,
):

    folder = os.path.join(root, case)
    os.makedirs(folder, exist_ok=True)

    # Resolve DEM path
    if dem_path is None:
        default_dem = os.path.join(folder, f"{case}.tif")
        if os.path.exists(default_dem):
            dem_path = default_dem
        else:
            # try auto-detect a single .tif in folder
            tifs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")]
            if len(tifs) == 1:
                dem_path = tifs[0]
            else:
                raise FileNotFoundError(
                    f"DEM not found. Expected {default_dem} or exactly one .tif in {folder}."
                )

    # Resolve coordinates
    if lat is not None and lon is not None:
        pass
    elif kmz_path is not None or os.path.exists(os.path.join(folder, f"{case}.kmz")):
        kmz_use = kmz_path or os.path.join(folder, f"{case}.kmz")
        lat, lon = _extract_coords_from_kmz(kmz_use)
    elif shp_path is not None or os.path.exists(os.path.join(folder, f"{case}.shp")):
        shp_use = shp_path or os.path.join(folder, f"{case}.shp")
        lat, lon = _extract_coords_from_shapefile(shp_use)
    else:
        raise ValueError("Coordinates not provided: supply lat/lon OR a KMZ/SHP path.")

    chip, transform, profile = _sample_chip(dem_path, float(lat), float(lon), size=size)

    paths = {
        "folder": folder,
        "dem_input": dem_path,
        "chip_tif": os.path.join(folder, "dem.tif"),
        "chip_npy": os.path.join(folder, "dem.npy"),
        "kmz_used": kmz_path,
        "shp_used": shp_path,
        "lat": float(lat),
        "lon": float(lon),
    }

    if save:
        with rasterio.open(paths["chip_tif"], "w", **profile) as dst:
            dst.write(chip, 1)
        np.save(paths["chip_npy"], chip)
        print(f"✅ Saved chip: {paths['chip_tif']} and {paths['chip_npy']}")

    if plot:
        plt.imshow(chip, cmap="terrain")
        plt.title(f"{case} DEM chip ({size}×{size})")
        plt.axis("off")
        plt.show()

    return chip, transform, profile, paths

def save_probability_and_plot(
    case: str,
    output_dir: str = None,
    dem_path: str = None,
    threshold: float = 0.1,
    prob_tif_name: str = "runout_probability.tif",
    overlay_png_name: str = "runout_probability_overlay.png",
    alpha: float = 0.6,
    cmap: str = "viridis",
    match_shape: str = "strict",  # "strict" | "crop"
):

    # default paths
    if output_dir is None:
        output_dir = f"{case}/output_nn"
    if dem_path is None:
        dem_path = f"{case}/dem.tif"
    os.makedirs(output_dir, exist_ok=True)

    # 1) raccogli i .npy
    run_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npy")]
    if not run_files:
        raise FileNotFoundError(f"Nessun .npy trovato in {output_dir}")

    # 2) accumula in prob_map
    prob_map = None
    n_runs = 0
    for path in run_files:
        arr = np.load(path)
        binary = (arr > float(threshold)).astype(np.uint8)
        if prob_map is None:
            prob_map = np.zeros_like(binary, dtype=np.float32)
        if prob_map.shape != binary.shape:
            if match_shape == "crop":
                h = min(prob_map.shape[0], binary.shape[0])
                w = min(prob_map.shape[1], binary.shape[1])
                prob_map = prob_map[:h, :w]
                binary = binary[:h, :w]
            else:
                raise ValueError(f"Shape diversa tra runouts: {prob_map.shape} vs {binary.shape}. "
                                 f"Usa match_shape='crop' se vuoi tagliare.")
        prob_map += binary
        n_runs += 1

    prob_map = prob_map / max(n_runs, 1)

    # 3) apri DEM e verifica shape
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        meta = src.meta.copy()

    dem = gaussian_filter(dem, sigma=1)

    if dem.shape != prob_map.shape:
        if match_shape == "crop":
            h = min(dem.shape[0], prob_map.shape[0])
            w = min(dem.shape[1], prob_map.shape[1])
            dem = dem[:h, :w]
            prob_map = prob_map[:h, :w]
            # Nota: il transform resta corretto per l'angolo in alto a sinistra,
            # ma la dimensione del raster cambia (ok perché aggiorniamo height/width).
        else:
            raise ValueError(f"Shape DEM {dem.shape} diversa da prob_map {prob_map.shape}. "
                             f"Imposta match_shape='crop' per tagliare al minimo comune.")

    # 4) salva GeoTIFF probabilità
    prob_out_path = os.path.join(output_dir, prob_tif_name)
    meta.update(dtype=rasterio.float32, count=1, height=prob_map.shape[0], width=prob_map.shape[1])
    with rasterio.open(prob_out_path, "w", **meta) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # 5) hillshade e overlay
    ls = LightSource(azdeg=315, altdeg=45)
    # Normalizza DEM per hillshade più leggibile (evita range enorme)
    dem_min, dem_max = np.nanpercentile(dem, [2, 98])
    dem_clip = np.clip(dem, dem_min, dem_max)
    hillshade = ls.hillshade(dem_clip, vert_exag=1, dx=1, dy=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(hillshade, cmap="gray")
    im = ax.imshow(prob_map, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(f"{case} – Runout probability (n={n_runs}, thr={threshold})")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    overlay_path = os.path.join(output_dir, overlay_png_name)
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved probability GeoTIFF: {prob_out_path}")
    print(f"🖼️ Saved overlay PNG:        {overlay_path}")

    return {
        "prob_tif": prob_out_path,
        "overlay_png": overlay_path,
        "n_runs": n_runs,
        "threshold": threshold,
        "shape": prob_map.shape,
    }

def save_v_probability_and_plot(
    case: str,
    output_dir: str = None,
    dem_path: str = None,
    threshold: float = 0.1,
    prob_tif_name: str = "v_runout_probability.tif",
    overlay_png_name: str = "v_runout_probability_overlay.png",
    alpha: float = 0.6,
    cmap: str = "viridis",
    match_shape: str = "strict",  # "strict" | "crop"
):

    # default pat769-frhs
    if output_dir is None:
        output_dir = f"{case}/output_v_nn"
    if dem_path is None:
        dem_path = f"{case}/dem.tif"
    os.makedirs(output_dir, exist_ok=True)

    # 1) raccogli i .npy
    run_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npy")]
    if not run_files:
        raise FileNotFoundError(f"Nessun .npy trovato in {output_dir}")

    # 2) accumula in prob_map
    prob_map = None
    n_runs = 0
    for path in run_files:
        arr = np.load(path)
        binary = (arr > float(threshold)).astype(np.uint8)
        if prob_map is None:
            prob_map = np.zeros_like(binary, dtype=np.float32)
        if prob_map.shape != binary.shape:
            if match_shape == "crop":
                h = min(prob_map.shape[0], binary.shape[0])
                w = min(prob_map.shape[1], binary.shape[1])
                prob_map = prob_map[:h, :w]
                binary = binary[:h, :w]
            else:
                raise ValueError(f"Shape diversa tra runouts: {prob_map.shape} vs {binary.shape}. "
                                 f"Usa match_shape='crop' se vuoi tagliare.")
        prob_map += binary
        n_runs += 1

    prob_map = prob_map / max(n_runs, 1)

    # 3) apri DEM e verifica shape
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        meta = src.meta.copy()

    dem = gaussian_filter(dem, sigma=1)

    if dem.shape != prob_map.shape:
        if match_shape == "crop":
            h = min(dem.shape[0], prob_map.shape[0])
            w = min(dem.shape[1], prob_map.shape[1])
            dem = dem[:h, :w]
            prob_map = prob_map[:h, :w]
            # Nota: il transform resta corretto per l'angolo in alto a sinistra,
            # ma la dimensione del raster cambia (ok perché aggiorniamo height/width).
        else:
            raise ValueError(f"Shape DEM {dem.shape} diversa da prob_map {prob_map.shape}. "
                             f"Imposta match_shape='crop' per tagliare al minimo comune.")

    # 4) salva GeoTIFF probabilità
    prob_out_path = os.path.join(output_dir, prob_tif_name)
    meta.update(dtype=rasterio.float32, count=1, height=prob_map.shape[0], width=prob_map.shape[1])
    with rasterio.open(prob_out_path, "w", **meta) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # 5) hillshade e overlay
    ls = LightSource(azdeg=315, altdeg=45)
    # Normalizza DEM per hillshade più leggibile (evita range enorme)
    dem_min, dem_max = np.nanpercentile(dem, [2, 98])
    dem_clip = np.clip(dem, dem_min, dem_max)
    hillshade = ls.hillshade(dem_clip, vert_exag=1, dx=1, dy=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(hillshade, cmap="gray")
    im = ax.imshow(prob_map, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(f"{case} – Runout probability (n={n_runs}, thr={threshold})")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    overlay_path = os.path.join(output_dir, overlay_png_name)
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved probability GeoTIFF: {prob_out_path}")
    print(f"🖼️ Saved overlay PNG:        {overlay_path}")

    return {
        "prob_tif": prob_out_path,
        "overlay_png": overlay_path,
        "n_runs": n_runs,
        "threshold": threshold,
        "shape": prob_map.shape,
    }

def load_runout_stack(output_dir: str, match_shape: str = "strict") -> np.ndarray:

    files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"Nessun .npy in {output_dir}")
    stack = None
    for p in files:
        arr = np.load(p)
        if stack is None:
            stack = arr[None, ...].astype(np.float32)
        else:
            if arr.shape != stack.shape[1:]:
                if match_shape == "crop":
                    h = min(arr.shape[0], stack.shape[1]); w = min(arr.shape[1], stack.shape[2])
                    arr = arr[:h,:w]; stack = stack[:, :h, :w]
                else:
                    raise ValueError(f"Shape mismatch: {arr.shape} vs {stack.shape[1:]}")
            stack = np.concatenate([stack, arr[None, ...].astype(np.float32)], axis=0)
    return stack  # (N,H,W)


def exceedance_probability(stack: np.ndarray, threshold: float) -> np.ndarray:

    return (stack > float(threshold)).mean(axis=0).astype(np.float32)


def percentile_maps(stack: np.ndarray, qs: List[float]) -> List[np.ndarray]:

    qs = np.asarray(qs, dtype=float)
    return [np.quantile(stack, q, axis=0).astype(np.float32) for q in qs]


def save_geotiff(arr: np.ndarray, ref_dem_path: str, out_path: str, dtype=rasterio.float32):
    with rasterio.open(ref_dem_path) as src:
        meta = src.meta.copy()
    meta.update(dtype=dtype, count=1, height=arr.shape[0], width=arr.shape[1])
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


def overlay_on_hillshade(arr: np.ndarray, dem_path: str, title: str, out_png: Optional[str] = None,
                         vmin: float = 0.0, vmax: float = 1.0, alpha: float = 0.6, cmap: str = "magma"):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)

    # hillshade robusto
    ls = LightSource(azdeg=315, altdeg=45)
    lo, hi = np.nanpercentile(dem, [2, 98])
    dem_clip = np.clip(dem, lo, hi)
    dem_clip = gaussian_filter(dem_clip, sigma=1)
    hill = ls.hillshade(dem_clip, vert_exag=1, dx=1, dy=1)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(hill, cmap="gray")
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
    ax.set_title(title); ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Value")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()


def aoi_mask_from_vector(aoi_path: str, dem_path: str) -> np.ndarray:

    if gpd is None:
        raise ImportError("geopandas richiesto per AOI")
    import rasterio.features as rfeatures

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError("AOI vuota")
    # proietta all'epsg del dem
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
    gdf = gdf.to_crs(dem_crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rfeatures.rasterize(shapes=shapes, out_shape=shape, transform=transform, fill=0, all_touched=False).astype(bool)
    return mask


def aoi_exceedance_curve(stack: np.ndarray, aoi_mask: np.ndarray, thresholds: np.ndarray) -> np.ndarray:

    area_aoi = aoi_mask.sum()
    if area_aoi == 0:
        raise ValueError("AOI senza pixel")
    fracs = []
    for t in thresholds:
        p = (stack > float(t)).mean(axis=0)   # prob pixel-wise
        fracs.append((p[aoi_mask] > 0.5).mean())  # % area con P>0.5 (puoi cambiare criterio)
    return np.asarray(fracs, dtype=float)


def save_metadata_json(out_path: str, meta: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

