from __future__ import annotations
import os
import warnings
from typing import Iterable, List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter, distance_transform_edt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: richdem for flow accumulation
try:
    import richdem as rd
    _HAS_RICHDEM = True
except Exception:
    _HAS_RICHDEM = False

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def pick_device(pref: str = "cuda") -> str:
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def safe_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
    return (x - mn) / denom

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path, allow_pickle=False).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Skipping {path}: {e}")
        return None

def compute_slope(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(dem, cell_size, cell_size)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    return slope.astype(np.float32)

def compute_curvature(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(dem, cell_size)
    gyy, _ = np.gradient(gy, cell_size)
    _, gxx = np.gradient(gx, cell_size)
    return (gxx + gyy).astype(np.float32)

def compute_flow_accumulation(dem_array: np.ndarray) -> np.ndarray:
    """
    Flow accumulation via richdem (D8). Falls back to a smoothed DEM proxy if richdem is missing.
    """
    if not _HAS_RICHDEM:
        # Fallback: smoothed DEM as a weak proxy, then scale [0,1]
        return safe_minmax(gaussian_filter(dem_array.astype(np.float32), 3))

    warnings.filterwarnings("ignore", message="No geotransform defined", module="richdem")
    no_data_val = -9999.0
    dem_clean = np.where(np.isnan(dem_array), no_data_val, dem_array).astype(np.float32)
    dem_rd = rd.rdarray(dem_clean, no_data=no_data_val)
    dem_rd.projection = "UNKNOWN"
    rd.FillDepressions(dem_rd, in_place=True)
    flow_acc = rd.FlowAccumulation(dem_rd, method="D8")
    acc = np.log1p(flow_acc)
    return safe_minmax(acc).astype(np.float32)

def compute_dist_to_source(h0_like: np.ndarray) -> np.ndarray:
    """
    Accepts an h0 array (thickness) OR a binary mask (True where source present).
    """
    source_mask = (h0_like > 0)
    return distance_transform_edt(~source_mask).astype(np.float32)

def compute_ns_coord(H: int, W: int) -> np.ndarray:
    return np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, axis=1)

def compute_we_coord(H: int, W: int) -> np.ndarray:
    return np.linspace(0, 1, W, dtype=np.float32)[None, :].repeat(H, axis=0)

def normalize_dem(dem: np.ndarray) -> np.ndarray:
    return safe_minmax(dem)

def normalize_params(cohesion: float, rho: float, volume: float) -> np.ndarray:
    COH = (5000, 50000)
    RHO = (917, 2650)
    VLO = (1e4, 1e7)
    norm_coh = (cohesion - COH[0]) / (COH[1] - COH[0])
    norm_rho = (rho - RHO[0]) / (RHO[1] - RHO[0])
    norm_logv = (np.log10(volume) - np.log10(VLO[0])) / (np.log10(VLO[1]) - np.log10(VLO[0]))
    return np.array([norm_coh, norm_rho, norm_logv], dtype=np.float32)

def create_single_h0_adaptive(
    output_path: Optional[str] = None,
    grid_size: int = 256,
    pixel_size: float = 30,
    volume: float = 1e7,
    target_thickness: float = 25.0,
    min_kernel: int = 7,
    max_kernel: Optional[int] = 21,
) -> np.ndarray:
    """Gaussian h0 scaled to a target volume; optionally saved."""
    PIXEL_AREA = pixel_size**2
    cx = cy = grid_size // 2

    source_area_m2 = volume / target_thickness
    source_radius_m = np.sqrt(source_area_m2 / np.pi)
    source_radius_px = int(source_radius_m / pixel_size)

    k = max(min_kernel, 2 * source_radius_px + 1)
    if max_kernel is not None:
        k = min(k, max_kernel)
    if k % 2 == 0:
        k += 1

    sigma = k / 5.0
    ax = np.linspace(-(k // 2), k // 2, k)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    ker /= ker.sum()

    h0 = np.zeros((grid_size, grid_size), dtype=np.float32)
    h = k // 2
    h0[cy - h: cy + h + 1, cx - h: cx + h + 1] = ker

    current_sum = h0.sum() * PIXEL_AREA
    h0 *= volume / current_sum

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, h0)
    return h0

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, in_features: int, num_channels: int):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(in_features, num_channels), nn.ReLU(), nn.Linear(num_channels, num_channels))
        self.beta  = nn.Sequential(nn.Linear(in_features, num_channels), nn.ReLU(), nn.Linear(num_channels, num_channels))
    def forward(self, x, conditioning):
        gamma = self.gamma(conditioning).unsqueeze(2).unsqueeze(3)
        beta  = self.beta(conditioning).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=p)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        idt = self.skip(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        return self.relu(out + idt)

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.GroupNorm(4, F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.GroupNorm(4, F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.GroupNorm(1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class UNetFiLMPlus(nn.Module):
    def __init__(self, in_channels: int = 8, film_params: int = 3, base_ch: int = 32):
        super().__init__()
        # enc
        self.enc1 = ResidualConvBlock(in_channels, base_ch);      self.film1 = FiLM(film_params, base_ch);      self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualConvBlock(base_ch, base_ch*2);        self.film2 = FiLM(film_params, base_ch*2);    self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualConvBlock(base_ch*2, base_ch*4);      self.film3 = FiLM(film_params, base_ch*4);    self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualConvBlock(base_ch*4, base_ch*8);      self.film4 = FiLM(film_params, base_ch*8);    self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualConvBlock(base_ch*8, base_ch*16)
        # dec
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.att4 = AttentionGate(base_ch*8, base_ch*8, base_ch*4)
        self.dec4 = ResidualConvBlock(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.att3 = AttentionGate(base_ch*4, base_ch*4, base_ch*2)
        self.dec3 = ResidualConvBlock(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.att2 = AttentionGate(base_ch*2, base_ch*2, base_ch)
        self.dec2 = ResidualConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.att1 = AttentionGate(base_ch, base_ch, base_ch // 2)
        self.dec1 = ResidualConvBlock(base_ch*2, base_ch)

        self.out_seg = nn.Conv2d(base_ch, 1, 1)
        self.out_thick = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, params):
        e1 = self.film1(self.enc1(x), params)
        e2 = self.film2(self.enc2(self.pool1(e1)), params)
        e3 = self.film3(self.enc3(self.pool2(e2)), params)
        e4 = self.film4(self.enc4(self.pool3(e3)), params)
        b  = self.bottleneck(self.pool4(e4))

        u4 = self.up4(b);  d4 = self.dec4(torch.cat([u4, self.att4(e4, u4)], dim=1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, self.att3(e3, u3)], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, self.att2(e2, u2)], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, self.att1(e1, u1)], dim=1))

        seg = self.out_seg(d1)
        thick = F.relu(self.out_thick(d1))
        return seg, thick

def load_model(
    checkpoint_path: str,
    in_channels: int = 8,
    film_params: int = 3,
    base_ch: int = 32,
    device: str = "cuda",
):
    device = pick_device(device)
    model = UNetFiLMPlus(in_channels=in_channels, film_params=film_params, base_ch=base_ch)
    try:
        ckpt = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("⚠️ load_state_dict:", "missing:", missing, "unexpected:", unexpected)
    model.to(device).eval()
    return model, device

def run_inference(model: nn.Module, input_stack: np.ndarray, param_tensor: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(input_stack[None], dtype=torch.float32, device=device)  # (1, C, H, W)
    p = torch.tensor(param_tensor[None], dtype=torch.float32, device=device) # (1, 3)
    with torch.no_grad():
        logits, thick = model(x, p)
        mask = torch.sigmoid(logits).squeeze().cpu().numpy()
        thick = thick.squeeze().cpu().numpy()
    return mask, thick

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_landslide_batch(
    landslide: str = "oso",
    image_size: Tuple[int, int] = (256, 256),
    cell_size: float = 30.0,
    cohesions: Iterable[float] = (19000,),
    rhos: Iterable[float] = (1950,),
    volumes: Iterable[float] = (9e6,),
    model_path: str = "weights/attunetplus.pth",
    base_ch: int = 32,
    device: str = "cuda",
    smooth_sigma: float = 1.0,
    save_npz: bool = True,
    save_npy: bool = True,
    plot: bool = False,
    use_dem_npy_first: bool = True,
    *,
    combination_mode: str = "grid",   # "grid" (default) or "elementwise"
    model: Any = None,                 # optional preloaded model; if None, will be loaded
    # NEW:
    reuse_dist: bool = True,
    source_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare inputs → run inference.

    combination_mode:
      - "grid": Cartesian product over cohesions × rhos × volumes (original behavior).
      - "elementwise": treat (cohesions, rhos, volumes) as parallel arrays of equal length,
                       running one triplet per index. Ideal for Sobol/LHS samples.

    If 'model' is provided, it's used directly (no reloading). Otherwise a model is loaded
    from 'model_path'. Shared DEM-derived features are computed once per call.

    Performance tweaks:
      - Preallocates the input buffer to avoid per-iteration stacking.
      - Caches distance-to-source if 'reuse_dist=True' (uses 'source_mask' if provided,
        else caches from the first h0).
    """
    H, W = image_size

    # Paths
    dem_path = f"{landslide}/dem.tif"
    dem_npy_path = f"{landslide}/dem.npy"
    prepared_dir = f"{landslide}/prepared"
    output_dir = f"{landslide}/output_nn"
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(dem_path))[0]

    # DEM
    if use_dem_npy_first and os.path.exists(dem_npy_path):
        dem = np.load(dem_npy_path).astype(np.float32)
    else:
        try:
            import rasterio
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load DEM from '{dem_npy_path}' or '{dem_path}'."
            ) from e

    dem = dem[:H, :W]
    if smooth_sigma and smooth_sigma > 0:
        dem = gaussian_filter(dem, sigma=smooth_sigma)

    # Shared features (compute once)
    dem_norm = normalize_dem(dem)
    slope = (compute_slope(dem, cell_size) / 90.0).astype(np.float32)
    curv = compute_curvature(dem, cell_size)
    curv = safe_minmax(np.clip(curv, -10, 10))
    flow_acc = compute_flow_accumulation(dem)
    ns_coord = compute_ns_coord(H, W)
    we_coord = compute_we_coord(H, W)

    # Preallocate input buffer: fill 6 static channels once; last 2 vary (dist, h0)
    base_stack = np.stack([dem_norm, slope, curv, ns_coord, we_coord, flow_acc], axis=0).astype(np.float32)
    input_stack = np.empty((8, H, W), dtype=np.float32)
    input_stack[:6] = base_stack

    # Distance caching
    dist_cached = None
    if reuse_dist and source_mask is not None:
        dist_cached = compute_dist_to_source(source_mask)
        dist_cached = dist_cached / (np.max(dist_cached) + 1e-6)

    # Model (load once if not provided)
    if model is None:
        model, device = load_model(model_path, in_channels=8, base_ch=base_ch, device=device)
    if hasattr(model, "eval"):
        model.eval()

    results: List[Dict[str, Any]] = []

    # Build iterator of triplets according to mode
    if combination_mode == "grid":
        triplets = (
            (float(c), float(r), float(v))
            for c in cohesions
            for r in rhos
            for v in volumes
        )
        n_expected = None  # unknown without materializing
    elif combination_mode == "elementwise":
        C = list(cohesions); R = list(rhos); V = list(volumes)
        if not (len(C) == len(R) == len(V)):
            raise ValueError(
                f"elementwise mode requires equal lengths: "
                f"len(cohesions)={len(C)}, len(rhos)={len(R)}, len(volumes)={len(V)}"
            )
        triplets = ((float(C[i]), float(R[i]), float(V[i])) for i in range(len(C)))
        n_expected = len(C)
    else:
        raise ValueError("combination_mode must be 'grid' or 'elementwise'")

    # Sweep
    for idx, (cohesion, rho, volume) in enumerate(triplets, start=1):
        # h0 for this volume
        h0 = create_single_h0_adaptive(
            output_path=None,
            grid_size=H,
            pixel_size=cell_size,
            volume=float(volume),
        ).astype(np.float32)

        # distance: reuse if possible
        if reuse_dist:
            if dist_cached is None:
                dist_cached = compute_dist_to_source(h0)
                dist_cached = dist_cached / (np.max(dist_cached) + 1e-6)
            dist = dist_cached
        else:
            dist = compute_dist_to_source(h0)
            dist = dist / (np.max(dist) + 1e-6)

        # fill buffer (no per-iter stacking)
        input_stack[6] = dist
        input_stack[7] = h0

        params = normalize_params(cohesion, rho, volume)

        # Stable, informative stub (use sequential sim index)
        vol_M = int(round(volume / 1e6))
        sim_idx = f"{idx:04d}" if n_expected else "001"
        out_stub = f"{base_name}_sim{sim_idx}_c{int(cohesion)}_r{int(rho)}_v{vol_M}M"

        npz_path = None
        if save_npz:
            npz_path = os.path.join(prepared_dir, f"{out_stub}.npz")
            # input_stack is already contiguous float32
            np.savez_compressed(npz_path, input=input_stack, params=params)

        # Inference (ensure no grad if PyTorch)
        try:
            with torch.no_grad():
                pred_mask, pred_thick = run_inference(model, input_stack, params, device)
        except Exception:
            pred_mask, pred_thick = run_inference(model, input_stack, params, device)

        if plot:
            ls = LightSource(azdeg=315, altdeg=45)
            hs = ls.shade(dem, cmap=plt.cm.gray, blend_mode="overlay", vert_exag=1)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(hs, aspect="equal")
            axs[0].imshow(pred_mask, cmap="Reds", alpha=0.35)
            axs[0].set_title("Predicted Mask")
            axs[0].axis("off")
            axs[1].imshow(hs, aspect="equal")
            thick_masked = pred_thick * pred_mask
            im = axs[1].imshow(thick_masked, cmap="magma", alpha=0.6, vmin=0)
            axs[1].set_title("Predicted Thickness (h)")
            axs[1].axis("off")
            cbar = plt.colorbar(im, ax=axs[1], shrink=0.8)
            cbar.set_label("Thickness (m)")
            plt.tight_layout(); plt.show()

        # Save final thickness (masked)
        final_thick = (pred_thick * pred_mask).astype(np.float32)
        final_npy_path = None
        if save_npy:
            final_npy_path = os.path.join(output_dir, f"{out_stub}.npy")
            np.save(final_npy_path, final_thick)

        results.append({
            "cohesion": float(cohesion),
            "rho": float(rho),
            "volume": float(volume),
            "npz_path": npz_path,
            "final_npy_path": final_npy_path,
        })

    return results

# --- Sobol sampling (log-space for volume) ---
import math
import pandas as pd
from scipy.stats import qmc  # pip/conda: scipy>=1.7

def sobol_params(
    n: int,
    volume_range=(1e4, 1e7),
    cohesion_range=(5e3, 5e4),
    rho_range=(917, 2650),
    seed: int = 123,
    log_volume: bool = True,
) -> pd.DataFrame:
    """
    Draw n Sobol samples across (volume, cohesion, rho).
    - volume in log10-space if log_volume=True.
    Returns DataFrame with columns ['volume','cohesion','rho'].
    """
    v_lo, v_hi = map(float, volume_range)
    c_lo, c_hi = map(float, cohesion_range)
    r_lo, r_hi = map(float, rho_range)

    eng = qmc.Sobol(d=3, scramble=True, seed=seed)
    U = eng.random(n)  # [0,1)^3

    if log_volume:
        lo_log, hi_log = math.log10(v_lo), math.log10(v_hi)
        vols = 10 ** (lo_log + U[:, 0] * (hi_log - lo_log))
    else:
        vols = v_lo + U[:, 0] * (v_hi - v_lo)

    cohs = c_lo + U[:, 1] * (c_hi - c_lo)
    rhos = r_lo + U[:, 2] * (r_hi - r_lo)

    return pd.DataFrame({"volume": vols, "cohesion": cohs, "rho": rhos})

# --- Run emulator over Sobol samples in ONE call (elementwise) ---
def run_sobol_samples(
    samples: pd.DataFrame,
    case: str,
    image_size=(256, 256),
    cell_size=30.0,
    model_path="weights/final/best_model.pth",
    device="cpu",
    *,
    reuse_dist: bool = True,
    source_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Calls emulator once for all Sobol triplets using combination_mode='elementwise'.
    Writes 'sobol_samples.csv' in {case}/output_nn and returns that directory.
    """
    out_dir = f"{case}/output_nn"
    os.makedirs(out_dir, exist_ok=True)
    samples = samples.reset_index(drop=True)
    samples.to_csv(os.path.join(out_dir, "sobol_samples.csv"), index=False)

    C = samples["cohesion"].to_numpy(float)
    R = samples["rho"].to_numpy(float)
    V = samples["volume"].to_numpy(float)

    run_landslide_batch(
        landslide=case,
        image_size=image_size,
        cell_size=float(cell_size),
        cohesions=C,
        rhos=R,
        volumes=V,
        model_path=model_path,
        device=device,
        plot=False,
        combination_mode="elementwise",
        reuse_dist=reuse_dist,
        source_mask=source_mask,
    )
    return out_dir
