import numpy as np
import slicer
import vtk
from scipy.spatial import ConvexHull, distance

# =======================
# PARAMETERS
# =======================
HEAD_FRACTION          = 0.15
DISTAL_FRACTION_77     = 0.12
DISTAL_FRACTION_84_85  = 0.10   # increased from 0.06 to capture full anterior condyle surface
RANSAC_ITERS           = 1200
INLIER_THRESH_MM       = 1.2
MIN_INLIERS            = 300

TROCH_SCAN_START       = 0.06
TROCH_SCAN_END         = 0.45
N_TROCH_SLICES         = 150
TROCH_SMOOTH_WIN       = 7
TROCH_BAND_HALF_FRAC   = 0.007
SUB_OFFSET_FRAC        = 0.10

# =================================================
# GET MODEL NODE
# =================================================
MODEL_NAME = None

if MODEL_NAME:
    modelNode = slicer.util.getNode(MODEL_NAME)
else:
    modelNode = None
    for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode"):
        pd = node.GetPolyData()
        if pd is not None and pd.GetNumberOfPoints() > 100:
            modelNode = node
            break
    if modelNode is None:
        raise RuntimeError(
            "No suitable model node found. "
            "Set MODEL_NAME to the exact node name."
        )

print(f"[INFO] Using model node: '{modelNode.GetName()}'")

polydata = modelNode.GetPolyData()
points   = polydata.GetPoints()
if points is None or points.GetNumberOfPoints() == 0:
    raise RuntimeError("Model has no points.")

pts = np.array(
    [points.GetPoint(i) for i in range(points.GetNumberOfPoints())],
    dtype=float
)

# =================================================
# PCA → LONG AXIS
# =================================================
pts_mean     = pts.mean(axis=0)
pts_centered = pts - pts_mean
_, _, Vt     = np.linalg.svd(pts_centered, full_matrices=False)
axis         = Vt[0]

proj         = pts_centered @ axis
min_p        = proj.min()
max_p        = proj.max()
total_length = max_p - min_p

# =================================================
# LOCAL FRAME  (x, y perpendicular to long axis)
# =================================================
z = axis
x = np.cross(z, [1, 0, 0])
if np.linalg.norm(x) < 1e-6:
    x = np.cross(z, [0, 1, 0])
x /= np.linalg.norm(x)
y  = np.cross(z, x)

# =================================================
# MARKUP HELPERS
# =================================================
fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Auto landmarks")

def add_point(p, label):
    fid.AddControlPoint(p.tolist() if isinstance(p, np.ndarray) else p)
    fid.SetNthControlPointLabel(fid.GetNumberOfControlPoints() - 1, label)

def make_line(name, p1, p2):
    line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
    line.AddControlPoint(p1.tolist() if isinstance(p1, np.ndarray) else p1)
    line.AddControlPoint(p2.tolist() if isinstance(p2, np.ndarray) else p2)
    return line

def dist3d(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

# =================================================
# SPHERE HELPERS
# =================================================
def fit_sphere_least_squares(P):
    A        = np.column_stack([P, np.ones(len(P))])
    b        = -np.sum(P ** 2, axis=1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c, d = coef
    center   = np.array([-a / 2, -b_ / 2, -c / 2], dtype=float)
    r2       = np.sum(center ** 2) - d
    return center, float(np.sqrt(max(r2, 0.0)))

def sphere_from_4pts(p1, p2, p3, p4):
    P   = np.stack([p1, p2, p3, p4])
    M   = np.stack([P[1]-P[0], P[2]-P[0], P[3]-P[0]])
    rhs = 0.5 * (np.sum(P[1:] ** 2, axis=1) - np.sum(P[0] ** 2))
    if abs(np.linalg.det(M)) < 1e-10:
        return None, None
    c = np.linalg.solve(M, rhs)
    return c, float(np.linalg.norm(P[0] - c))

def _try_ransac(P, n_iters, thresh, min_inl, rng_):
    best = None
    for _ in range(n_iters):
        idx = rng_.choice(len(P), size=4, replace=False)
        c, r = sphere_from_4pts(P[idx[0]], P[idx[1]], P[idx[2]], P[idx[3]])
        if c is None or not np.isfinite(r):
            continue
        inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    cnt = 0 if best is None else int(best.sum())
    return (best if cnt >= min_inl else None), cnt

# =================================================
# DETECT PROXIMAL / DISTAL END
# =================================================
mask_low  = proj < (min_p + HEAD_FRACTION * total_length)
mask_high = proj > (max_p - HEAD_FRACTION * total_length)

rng      = np.random.default_rng(0)
pts_low  = pts[mask_low]
pts_high = pts[mask_high]

inl_low,  cnt_low  = _try_ransac(pts_low,  RANSAC_ITERS, INLIER_THRESH_MM, MIN_INLIERS, rng)
inl_high, cnt_high = _try_ransac(pts_high, RANSAC_ITERS, INLIER_THRESH_MM, MIN_INLIERS, rng)

score_low  = cnt_low  / max(len(pts_low),  1)
score_high = cnt_high / max(len(pts_high), 1)

print(f"[INFO] Sphere score LOW : {score_low:.4f}  ({cnt_low}/{len(pts_low)})")
print(f"[INFO] Sphere score HIGH: {score_high:.4f}  ({cnt_high}/{len(pts_high)})")

if score_low >= score_high:
    distal_is_low = False
    proximal_pts  = pts_low
    best_inliers  = inl_low
else:
    distal_is_low = True
    proximal_pts  = pts_high
    best_inliers  = inl_high

if best_inliers is None:
    raise RuntimeError(
        "Femoral head sphere fit failed on both ends. "
        "Try increasing HEAD_FRACTION or INLIER_THRESH_MM."
    )

print(f"[INFO] Head at {'LOW' if not distal_is_low else 'HIGH'} end | distal_is_low={distal_is_low}")

distal_tip_proj = min_p if distal_is_low else max_p

if distal_is_low:
    distal_pts_77    = pts[proj < (min_p + DISTAL_FRACTION_77    * total_length)]
    distal_pts_84_85 = pts[proj < (min_p + DISTAL_FRACTION_84_85 * total_length)]
else:
    distal_pts_77    = pts[proj > (max_p - DISTAL_FRACTION_77    * total_length)]
    distal_pts_84_85 = pts[proj > (max_p - DISTAL_FRACTION_84_85 * total_length)]

prox_end = max_p if distal_is_low else min_p
prox_dir = -1.0  if distal_is_low else +1.0

# =================================================
# 75. MAXIMUM FEMUR LENGTH
# =================================================
i_prox  = np.argmin(proj)
i_dist  = np.argmax(proj)
prox_pt = pts[i_prox]
dist_pt = pts[i_dist]

add_point(prox_pt, "75_MaxLen_Prox")
add_point(dist_pt, "75_MaxLen_Dist")
make_line("75_Maximum_femur_length", prox_pt, dist_pt)
max_femur_length = dist3d(prox_pt, dist_pt)

# =================================================
# 77. EPICONDYLAR BREADTH  (temp — corrected after #78)
# =================================================
distal_2d = np.column_stack([
    (distal_pts_77 - pts_mean) @ x,
    (distal_pts_77 - pts_mean) @ y
])
_, _, Vt2 = np.linalg.svd(distal_2d - distal_2d.mean(axis=0), full_matrices=False)
ml_dir    = Vt2[0]
proj_ml   = distal_2d @ ml_dir
epi_A     = distal_pts_77[np.argmin(proj_ml)]
epi_B     = distal_pts_77[np.argmax(proj_ml)]

# =================================================
# 78. MAXIMUM FEMORAL HEAD DIAMETER
# =================================================
center78, radius78 = fit_sphere_least_squares(proximal_pts[best_inliers])
max_head_diameter  = 2.0 * radius78

d_qc = np.cross(axis, [1, 0, 0])
if np.linalg.norm(d_qc) < 1e-6:
    d_qc = np.cross(axis, [0, 1, 0])
d_qc /= np.linalg.norm(d_qc)

pH1 = center78 + radius78 * d_qc
pH2 = center78 - radius78 * d_qc
add_point(center78, "FemurHead_Center")
add_point(pH1,      "FemurHead_Max_A")
add_point(pH2,      "FemurHead_Max_B")
make_line("78_Max_femoral_head_diameter", pH1, pH2)

# =================================================
# ANATOMICAL DIRECTIONS
# =================================================
shaft_center = pts_mean
v_med        = center78 - shaft_center
v_med       -= np.dot(v_med, axis) * axis
if np.linalg.norm(v_med) < 1e-8:
    raise RuntimeError("Failed to compute medial direction from femoral head.")
v_med /= np.linalg.norm(v_med)
v_lat  = -v_med

v_ap = np.cross(axis, v_med)
if np.linalg.norm(v_ap) < 1e-8:
    raise RuntimeError("Failed to compute AP direction.")
v_ap /= np.linalg.norm(v_ap)

mid_band_frac = 0.08
mid_proj      = 0.5 * (min_p + max_p)
mid_mask      = np.abs(proj - mid_proj) < (total_length * mid_band_frac)
mid_band_pts  = pts[mid_mask]
if len(mid_band_pts) > 20:
    pos_ap_ext = float(((mid_band_pts - pts_mean) @  v_ap).max())
    neg_ap_ext = float(((mid_band_pts - pts_mean) @ -v_ap).max())
    if pos_ap_ext > neg_ap_ext:
        v_ap = -v_ap

# =================================================
# CORRECT EPICONDYLE LABELING FOR LEFT / RIGHT
# =================================================
# epi_med: argmax on v_med — known to work correctly
epi_med = distal_pts_77[np.argmax((distal_pts_77 - pts_mean) @ v_med)]

# epi_lat: axial zone 6-18% from distal tip + anterior 60% of AP range
# to skip both the condyle articular surface and the posterolateral edge
_d77_from_tip = np.abs((distal_pts_77 - pts_mean) @ axis - distal_tip_proj)
_axial_mask   = (_d77_from_tip > total_length * 0.06) & \
                (_d77_from_tip < total_length * 0.18)
_d77_ap       = (distal_pts_77 - pts_mean) @ v_ap
_ap_threshold = _d77_ap.min() + 0.60 * (_d77_ap.max() - _d77_ap.min())
_ap_mask      = _d77_ap < _ap_threshold
_epi_lat_mask = _axial_mask & _ap_mask
_epi_lat_pool = distal_pts_77[_epi_lat_mask]
if len(_epi_lat_pool) < 10:
    _epi_lat_pool = distal_pts_77[_axial_mask] if _axial_mask.sum() > 10 else distal_pts_77
    print("[WARN] epi_lat AP constraint relaxed")
epi_lat = _epi_lat_pool[np.argmax((_epi_lat_pool - pts_mean) @ v_lat)]
print(f"[INFO] epi_lat zone: {_epi_lat_mask.sum()} pts (axial 6-18% + anterior 60%)")

add_point(epi_med, "Epicondyle_Medial")
add_point(epi_lat, "Epicondyle_Lateral")
make_line("77_Epicondylar_breadth", epi_med, epi_lat)
epicondylar_breadth = dist3d(epi_med, epi_lat)

# =================================================
# 76. BICONDYLAR LENGTH
# =================================================
d_to_med_76 = np.linalg.norm(distal_pts_84_85 - epi_med, axis=1)
d_to_lat_76 = np.linalg.norm(distal_pts_84_85 - epi_lat, axis=1)
med_pts_76  = distal_pts_84_85[d_to_med_76 <  d_to_lat_76]
lat_pts_76  = distal_pts_84_85[d_to_lat_76 <= d_to_med_76]

def most_distal_proj(cpts):
    p = (cpts - pts_mean) @ axis
    return p.min() if distal_is_low else p.max()

med_distal_proj     = most_distal_proj(med_pts_76)
lat_distal_proj     = most_distal_proj(lat_pts_76)
condylar_plane_proj = (min(med_distal_proj, lat_distal_proj) if distal_is_low
                       else max(med_distal_proj, lat_distal_proj))

head_prox_proj    = (prox_pt - pts_mean) @ axis
bicondylar_length = abs(head_prox_proj - condylar_plane_proj)

p76_prox = pts_mean + head_prox_proj      * axis
p76_dist = pts_mean + condylar_plane_proj * axis

add_point(p76_prox, "76_Head_Prox_Axis")
add_point(p76_dist, "76_Condylar_Plane")
make_line("76_Bicondylar_length", p76_prox, p76_dist)

# =================================================
# TROCHANTER DETECTION
# =================================================
BAND_HALF = total_length * TROCH_BAND_HALF_FRAC
N         = N_TROCH_SLICES

scan_levels = np.linspace(
    prox_end + prox_dir * TROCH_SCAN_START * total_length,
    prox_end + prox_dir * TROCH_SCAN_END   * total_length,
    N
)

lateral_ext = np.full(N, np.nan)
medial_ext  = np.full(N, np.nan)

for k, lev in enumerate(scan_levels):
    mask = np.abs(proj - lev) < BAND_HALF
    if mask.sum() < 15:
        continue
    band = pts[mask]
    lateral_ext[k] = float(((band - pts_mean) @ v_lat).max())
    medial_ext[k]  = float(((band - pts_mean) @ v_med).max())

def moving_avg(a, w):
    valid = ~np.isnan(a)
    if valid.sum() < w:
        return a.copy()
    idx    = np.arange(len(a))
    filled = np.interp(idx, idx[valid], a[valid])
    return np.convolve(filled, np.ones(w) / w, mode='same')

sm_lateral = moving_avg(lateral_ext, TROCH_SMOOTH_WIN)
sm_medial  = moving_avg(medial_ext,  TROCH_SMOOTH_WIN)

gt_search_end = int(N * 0.60)
gt_peak_idx   = int(np.argmax(sm_lateral[:gt_search_end]))
gt_level      = scan_levels[gt_peak_idx]
add_point(pts_mean + gt_level * axis, "GreaterTrochanter_Peak")

lt_search_start = gt_peak_idx + TROCH_SMOOTH_WIN
if lt_search_start >= N:
    raise RuntimeError("Greater trochanter found too close to scan end. Try increasing TROCH_SCAN_END.")

lt_peak_idx = int(np.argmax(sm_medial[lt_search_start:])) + lt_search_start
lt_level    = scan_levels[lt_peak_idx]
add_point(pts_mean + lt_level * axis, "LesserTrochanter_Peak")

sub_level  = lt_level + prox_dir * SUB_OFFSET_FRAC * total_length
sub_origin = pts_mean + sub_level * axis
add_point(sub_origin, "Subtrochanteric_Level")

sub_plane  = vtk.vtkPlane()
sub_plane.SetOrigin(sub_origin)
sub_plane.SetNormal(axis)
sub_cutter = vtk.vtkCutter()
sub_cutter.SetCutFunction(sub_plane)
sub_cutter.SetInputData(polydata)
sub_cutter.Update()

sub_poly = sub_cutter.GetOutput()
if sub_poly.GetNumberOfPoints() < 20:
    raise RuntimeError("Subtrochanteric cross-section has too few points.")

sub_pts = np.array(
    [sub_poly.GetPoint(i) for i in range(sub_poly.GetNumberOfPoints())],
    dtype=float
)

# 79 — Transverse subtrochanteric (ML, v_med direction)
proj79 = (sub_pts - sub_origin) @ v_med
p79_A  = sub_pts[np.argmin(proj79)]
p79_B  = sub_pts[np.argmax(proj79)]
add_point(p79_A, "79_Subtro_Lateral")
add_point(p79_B, "79_Subtro_Medial")
make_line("79_Transverse_subtrochanteric", p79_A, p79_B)
len79 = dist3d(p79_A, p79_B)

# 80 — Sagittal subtrochanteric (AP, v_ap direction)
proj80 = (sub_pts - sub_origin) @ v_ap
p80_A  = sub_pts[np.argmin(proj80)]
p80_B  = sub_pts[np.argmax(proj80)]
add_point(p80_A, "80_Subtro_Anterior")
add_point(p80_B, "80_Subtro_Posterior")
make_line("80_AP_subtrochanteric", p80_A, p80_B)
len80 = dist3d(p80_A, p80_B)

gt_frac         = abs(gt_level  - prox_end) / total_length
lt_frac         = abs(lt_level  - prox_end) / total_length
sub_frac        = abs(sub_level - prox_end) / total_length
sub_mm_below_lt = abs(sub_level - lt_level)

# =================================================
# 81 & 82. MIDSHAFT DIAMETERS  +  83. CIRCUMFERENCE
# =================================================
mid_p     = 0.5 * (min_p + max_p)
mid_point = pts_mean + mid_p * axis

mid_plane  = vtk.vtkPlane()
mid_plane.SetOrigin(mid_point)
mid_plane.SetNormal(axis)
mid_cutter = vtk.vtkCutter()
mid_cutter.SetCutFunction(mid_plane)
mid_cutter.SetInputData(polydata)
mid_cutter.Update()

cut_poly = mid_cutter.GetOutput()
if cut_poly.GetNumberOfPoints() < 30:
    raise RuntimeError("Midshaft cross-section failed — too few points.")

cut_pts = np.array(
    [cut_poly.GetPoint(i) for i in range(cut_poly.GetNumberOfPoints())],
    dtype=float
)

pts_2d   = np.column_stack([
    (cut_pts - mid_point) @ x,
    (cut_pts - mid_point) @ y
])
hull     = ConvexHull(pts_2d)
hull_pts = pts_2d[hull.vertices]

# 81. Maximum midshaft diameter
D    = distance.squareform(distance.pdist(hull_pts))
i, j = np.unravel_index(np.argmax(D), D.shape)
pA   = mid_point + hull_pts[i, 0] * x + hull_pts[i, 1] * y
pB   = mid_point + hull_pts[j, 0] * x + hull_pts[j, 1] * y
add_point(pA, "Midshaft_Max_A")
add_point(pB, "Midshaft_Max_B")
make_line("81_Max_midshaft_diameter", pA, pB)
max_midshaft_diameter = dist3d(pA, pB)

# 82. Minimum midshaft diameter
min_midshaft_diameter = np.inf
best_normal           = None
for k in range(len(hull_pts)):
    edge   = hull_pts[(k + 1) % len(hull_pts)] - hull_pts[k]
    edge  /= np.linalg.norm(edge)
    normal = np.array([-edge[1], edge[0]])
    proj_n = hull_pts @ normal
    width  = float(proj_n.max() - proj_n.min())
    if width < min_midshaft_diameter:
        min_midshaft_diameter = width
        best_normal           = normal

i_mn = np.argmin(hull_pts @ best_normal)
i_mx = np.argmax(hull_pts @ best_normal)
pC   = mid_point + hull_pts[i_mn, 0] * x + hull_pts[i_mn, 1] * y
pD   = mid_point + hull_pts[i_mx, 0] * x + hull_pts[i_mx, 1] * y
add_point(pC, "Midshaft_Min_A")
add_point(pD, "Midshaft_Min_B")
make_line("82_Min_midshaft_diameter", pC, pD)
min_midshaft_diameter = dist3d(pC, pD)

# 83. Midshaft circumference
midshaft_circumference = float(hull.area)

# =================================================
# 84 & 85. CONDYLE AP LENGTHS
# =================================================
d_to_med = np.linalg.norm(distal_pts_84_85 - epi_med, axis=1)
d_to_lat = np.linalg.norm(distal_pts_84_85 - epi_lat, axis=1)

medial_condyle_pts  = distal_pts_84_85[d_to_med <  d_to_lat]
lateral_condyle_pts = distal_pts_84_85[d_to_lat <= d_to_med]

if len(medial_condyle_pts) < 50 or len(lateral_condyle_pts) < 50:
    raise RuntimeError(
        "Condyle split failed. "
        "Try increasing DISTAL_FRACTION_84_85 (e.g. 0.12–0.14)."
    )

# 84. Lateral condyle AP
proj_lat_ap = (lateral_condyle_pts - shaft_center) @ v_ap
lat_A = lateral_condyle_pts[np.argmin(proj_lat_ap)]
lat_P = lateral_condyle_pts[np.argmax(proj_lat_ap)]
add_point(lat_A, "84_Lat_Anterior")
add_point(lat_P, "84_Lat_Posterior")
make_line("84_Lateral_condyle_AP", lat_A, lat_P)
len84 = dist3d(lat_A, lat_P)

# 85. Medial condyle AP
proj_med_ap = (medial_condyle_pts - shaft_center) @ v_ap
med_A = medial_condyle_pts[np.argmin(proj_med_ap)]
med_P = medial_condyle_pts[np.argmax(proj_med_ap)]
add_point(med_A, "85_Med_Anterior")
add_point(med_P, "85_Med_Posterior")
make_line("85_Medial_condyle_AP", med_A, med_P)
len85 = dist3d(med_A, med_P)

# =================================================
# OUTPUT
# =================================================
print("=== FEMUR OSTEOMETRIC MEASUREMENTS ===")
print(f"75. Maximum femur length                   : {max_femur_length:.2f} mm  (true 3-D caliper)")
print(f"76. Bicondylar length                      : {bicondylar_length:.2f} mm  (axis-projected)")
print(f"    75 − 76 difference                     : {max_femur_length - bicondylar_length:.2f} mm")
print(f"77. Epicondylar breadth                    : {epicondylar_breadth:.2f} mm")
print(f"78. Max femoral head diameter              : {max_head_diameter:.2f} mm")
print(f"    (#78 sphere inliers)                   : {int(best_inliers.sum())} / {len(proximal_pts)}")
print(f"    [QC] Greater trochanter peak           : {gt_frac  * 100:.1f}%  ({gt_frac  * total_length:.1f} mm from proximal)")
print(f"    [QC] Lesser trochanter peak            : {lt_frac  * 100:.1f}%  ({lt_frac  * total_length:.1f} mm from proximal)")
print(f"    [QC] Subtrochanteric plane             : {sub_frac * 100:.1f}%  ({sub_mm_below_lt:.1f} mm below LT)")
print(f"79. Transverse subtrochanteric diameter    : {len79:.2f} mm")
print(f"80. AP subtrochanteric diameter            : {len80:.2f} mm")
print(f"81. Maximum midshaft diameter              : {max_midshaft_diameter:.2f} mm")
print(f"82. Minimum midshaft diameter              : {min_midshaft_diameter:.2f} mm")
print(f"83. Midshaft circumference                 : {midshaft_circumference:.2f} mm")
print(f"84. Lateral condyle AP length              : {len84:.2f} mm")
print(f"85. Medial condyle AP length               : {len85:.2f} mm")
