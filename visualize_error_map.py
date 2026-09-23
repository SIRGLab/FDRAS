"""Render FDRAS error maps as a 3D airway-surface heatmap (paper Fig. 3 style).

The input airway mask is surfaced with marching cubes and each vertex is coloured by the
predicted error map sampled in a small neighbourhood (errors live in a thin band around
the boundary, so a plain nearest-voxel lookup under-reports them). Warmer = larger error.

Example:
    python -m visualize_error_map \
        --mask pred.nii.gz --err out/sdf_err.nii.gz --refined out/mask.nii.gz \
        --out fig_sdf_err.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Dark orange body -> yellow hotspots, as in Fig. 3.
WARM = LinearSegmentedColormap.from_list("fdras_warm", ["#a33a00", "#e86a00", "#ff9f1a", "#ffd21f", "#fff27a"])


def load_canonical(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load as RAS+ so the anterior camera is the same for every case."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj).astype(np.float32), np.array(img.header.get_zooms()[:3], float), img.affine


def surface(mask: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> pv.PolyData:
    grid = pv.ImageData(dimensions=mask.shape, spacing=spacing, origin=origin)
    grid.point_data["m"] = ndi.gaussian_filter((mask > 0.5).astype(np.float32), 0.7).ravel(order="F")
    return grid.contour([0.5], scalars="m").smooth_taubin(n_iter=30, pass_band=0.1)


def sample_error(mesh: pv.PolyData, err: np.ndarray, spacing: np.ndarray, origin: np.ndarray, band_vox: int) -> np.ndarray:
    err_band = ndi.maximum_filter(err, size=2 * band_vox + 1) if band_vox > 0 else err
    ijk = np.rint((mesh.points - origin) / spacing).astype(int)
    ijk = np.clip(ijk, 0, np.array(err.shape) - 1)
    return err_band[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def auto_zoom_centre(mesh: pv.PolyData, values: np.ndarray, radius_mm: float) -> np.ndarray:
    """Centre of the surface patch with the highest mean error."""
    tree = cKDTree(mesh.points)
    step = max(1, len(values) // 20000)
    idx = np.arange(0, len(values), step)
    scores = [values[tree.query_ball_point(mesh.points[i], radius_mm)].mean() for i in idx]
    return mesh.points[idx[int(np.argmax(scores))]]


def render(mesh, centre, half_extent, window, scalars=None, clim=None, color="white", markers=None):
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("black")
    kw = dict(smooth_shading=True, ambient=0.25, diffuse=0.8, specular=0.25, specular_power=20)
    if scalars is None:
        pl.add_mesh(mesh, color=color, **kw)
    else:
        pl.add_mesh(mesh, scalars=scalars, cmap=WARM, clim=clim, show_scalar_bar=False, **kw)
    if markers is not None and len(markers):
        pl.add_mesh(pv.PolyData(markers), color="cyan", point_size=14, render_points_as_spheres=True)
    # Anterior view in RAS+: camera on +y looking back, head up.
    pl.camera.position = (centre[0], centre[1] + 1000.0, centre[2])
    pl.camera.focal_point = tuple(centre)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = half_extent
    pl.camera.clipping_range = (1.0, 5000.0)
    img = pl.screenshot(return_img=True)
    return img, pl


def world_to_pixel(pl: pv.Plotter, pts: np.ndarray, height: int) -> np.ndarray:
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    out = []
    for p in pts:
        coord.SetValue(*map(float, p))
        x, y = coord.GetComputedDoubleDisplayValue(pl.renderer)
        out.append((x, height - y))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", required=True, help="Input (backbone) airway mask; this surface is painted")
    ap.add_argument("--err", required=True, help="Error map in [0,1], e.g. sdf_err.nii.gz from predict.py --out_err_dir")
    ap.add_argument("--refined", default=None, help="Optional FDRAS refined mask for a third panel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--band_vox", type=int, default=1, help="Neighbourhood radius for sampling the error band")
    ap.add_argument("--clim", type=float, nargs=2, default=None, help="Fixed colour limits; default = percentiles")
    ap.add_argument("--pct", type=float, nargs=2, default=[2.0, 99.5], help="Percentiles for auto colour limits")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Display contrast (>1 keeps most of the tree dark and only the top errors bright)")
    ap.add_argument("--no_colorbar", action="store_true", help="Hide the colorbar with raw error values")
    ap.add_argument("--names", nargs=3, default=["Backbone", "Predicted SDF Error Map", "Backbone + FDRAS"],
                    help="Panel titles: input, error map, repaired")
    ap.add_argument("--zoom_centre", type=float, nargs=3, default=None, help="Zoom centre in RAS voxel ijk")
    ap.add_argument("--zoom_mm", type=float, default=40.0)
    ap.add_argument("--markers_json", default=None, help="JSON with 'gap_centres_vox' (original ijk) to overlay")
    ap.add_argument("--markers_ref", default=None, help="NIfTI the marker ijk refer to (default: --mask)")
    a = ap.parse_args()

    mask, spacing, affine = load_canonical(a.mask)
    err, _, _ = load_canonical(a.err)
    if err.shape != mask.shape:
        raise ValueError(f"error map {err.shape} and mask {mask.shape} differ")
    refined = load_canonical(a.refined)[0] if a.refined else None
    # Crop to the union bbox; mesh coordinates stay in full-volume voxel*spacing space via origin.
    union = (mask > 0.5) if refined is None else (mask > 0.5) | (refined > 0.5)
    nz = np.argwhere(union)
    lo_v = np.maximum(nz.min(0) - 4, 0)
    hi_v = np.minimum(nz.max(0) + 5, np.array(mask.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo_v, hi_v))
    mask, err = mask[sl], err[sl]
    refined = refined[sl] if refined is not None else None
    origin = lo_v * spacing

    mesh = surface(mask, spacing, origin)
    vals = sample_error(mesh, err, spacing, origin, a.band_vox)
    clim = tuple(a.clim) if a.clim else tuple(np.percentile(vals, a.pct))
    norm = PowerNorm(a.gamma, vmin=clim[0], vmax=clim[1], clip=True)
    mesh["err"] = np.asarray(norm(vals))  # 0..1 after contrast; the colorbar maps back to raw values

    markers = None
    if a.markers_json:
        ref = nib.load(a.markers_ref or a.mask)
        ijk = np.array(json.load(open(a.markers_json))["gap_centres_vox"], float)
        world = nib.affines.apply_affine(ref.affine, ijk)
        markers = nib.affines.apply_affine(np.linalg.inv(affine), world) * spacing

    lo, hi = mesh.bounds[::2], mesh.bounds[1::2]
    centre = (np.array(lo) + np.array(hi)) / 2
    half = 0.53 * max(hi[2] - lo[2], (hi[0] - lo[0]) * 0.75)
    zc = np.array(a.zoom_centre) * spacing if a.zoom_centre else auto_zoom_centre(mesh, vals, a.zoom_mm / 4)

    panels = [(a.names[0], None, "white", None)]
    panels.append((a.names[1], "err", None, markers))
    refined_mesh = surface(refined, spacing, origin) if refined is not None else None
    if refined_mesh is not None:
        panels.append((a.names[2], None, "white", None))

    win = (900, 1200)
    fig, axes = plt.subplots(1, 2 * len(panels), figsize=(4.2 * len(panels), 4.2),
                             gridspec_kw={"width_ratios": [3, 1] * len(panels)}, facecolor="black")
    zwin = (400, 1200)
    # Box must match the inset's aspect: parallel_scale sets the half-height, width follows the window.
    hw = a.zoom_mm / 2 * zwin[0] / zwin[1]
    box = np.array([zc + [-hw, 0, -a.zoom_mm / 2], zc + [hw, 0, a.zoom_mm / 2]])
    for k, (title, scal, col, mk) in enumerate(panels):
        m = refined_mesh if k == 2 else mesh
        img, pl = render(m, centre, half, win, scalars=scal, clim=(0.0, 1.0), color=col, markers=mk)
        px = world_to_pixel(pl, box, img.shape[0])
        pl.close()
        zimg, zpl = render(m, zc, a.zoom_mm / 2, zwin, scalars=scal, clim=(0.0, 1.0), color=col, markers=mk)
        zpl.close()
        ax, zax = axes[2 * k], axes[2 * k + 1]
        ax.imshow(img)
        x0, x1 = sorted(px[:, 0])
        y0, y1 = sorted(px[:, 1])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.5))
        zax.imshow(zimg)
        for s in zax.spines.values():
            s.set_edgecolor("red")
            s.set_linewidth(2)
        ax.set_title(title, color="white", fontsize=10)
        for x in (ax, zax):
            x.set_xticks([])
            x.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if scal is not None and not a.no_colorbar:
            cax = ax.inset_axes([0.04, 0.06, 0.03, 0.3])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=WARM), cax=cax)
            cb.set_ticks([clim[0], clim[1]])
            cb.ax.set_yticklabels([f"{clim[0]:.4f}", f"{clim[1]:.4f}"])
            cb.ax.tick_params(colors="white", labelsize=5, length=2)
            cb.outline.set_edgecolor("white")
            cb.set_label("predicted error (raw)", color="white", fontsize=5)
    fig.tight_layout(pad=0.3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="black")
    meta = {"clim": [float(c) for c in clim], "pct": a.pct, "gamma": a.gamma, "band_vox": a.band_vox,
            "zoom_centre_mm": [float(v) for v in zc], "err_vertex_stats": {
                "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max())}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out}  clim={meta['clim']}")


if __name__ == "__main__":
    main()
