import logging
import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from scipy.spatial import ConvexHull, distance


class FemurMeasurements(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Femur Measurements"
        self.parent.categories = ["Quantification"]
        self.parent.dependencies = []
        self.parent.contributors = [""]
        self.parent.helpText = (
            "Automated osteometric measurements of the femur (landmarks 75–85) "
            "from a 3D surface model."
        )
        self.parent.acknowledgementText = ""


class FemurMeasurementsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ── Input ──────────────────────────────────────────────
        inputBox = ctk.ctkCollapsibleButton()
        inputBox.text = "Input"
        self.layout.addWidget(inputBox)
        inputLayout = qt.QFormLayout(inputBox)

        self.modelSelector = slicer.qMRMLNodeComboBox()
        self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelector.selectNodeUponCreation = True
        self.modelSelector.addEnabled = False
        self.modelSelector.removeEnabled = False
        self.modelSelector.noneEnabled = False
        self.modelSelector.showHidden = False
        self.modelSelector.showChildNodeTypes = False
        self.modelSelector.setMRMLScene(slicer.mrmlScene)
        self.modelSelector.setToolTip("Odaberi model femura.")
        inputLayout.addRow("Model femura:", self.modelSelector)

        # ── Parameters ────────────────────────────────────────
        paramsBox = ctk.ctkCollapsibleButton()
        paramsBox.text = "Parametri"
        paramsBox.collapsed = True
        self.layout.addWidget(paramsBox)
        paramsLayout = qt.QFormLayout(paramsBox)

        def dspinbox(lo, hi, step, dec, val):
            sb = qt.QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setSingleStep(step)
            sb.setDecimals(dec); sb.setValue(val)
            return sb

        def ispinbox(lo, hi, step, val):
            sb = qt.QSpinBox()
            sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val)
            return sb

        self.spHeadFrac      = dspinbox(0.05, 0.40, 0.01, 2, 0.15)
        self.spDistal77      = dspinbox(0.05, 0.30, 0.01, 2, 0.12)
        self.spDistal8485    = dspinbox(0.05, 0.30, 0.01, 2, 0.10)
        self.spRansacIters   = ispinbox(100, 5000, 100, 1200)
        self.spInlierThresh  = dspinbox(0.1, 5.0, 0.1, 1, 1.2)
        self.spMinInliers    = ispinbox(50, 1000, 50, 300)

        paramsLayout.addRow("Head fraction:",           self.spHeadFrac)
        paramsLayout.addRow("Distal fraction (#77):",   self.spDistal77)
        paramsLayout.addRow("Distal fraction (#84-85):", self.spDistal8485)
        paramsLayout.addRow("RANSAC iteracije:",        self.spRansacIters)
        paramsLayout.addRow("Inlier prag (mm):",        self.spInlierThresh)
        paramsLayout.addRow("Min. inliers:",            self.spMinInliers)

        # ── Apply ──────────────────────────────────────────────
        self.applyButton = qt.QPushButton("Pokreni mjerenja")
        self.applyButton.toolTip = "Izračunaj sva osteometrijska mjerenja femura."
        self.layout.addWidget(self.applyButton)

        # ── Results ────────────────────────────────────────────
        resultsBox = ctk.ctkCollapsibleButton()
        resultsBox.text = "Rezultati"
        self.layout.addWidget(resultsBox)
        resultsLayout = qt.QVBoxLayout(resultsBox)

        self.resultsTable = qt.QTableWidget()
        self.resultsTable.setColumnCount(2)
        self.resultsTable.setHorizontalHeaderLabels(["Mjerenje", "Vrijednost (mm)"])
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        self.resultsTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.resultsTable.setAlternatingRowColors(True)
        self.resultsTable.setMinimumHeight(300)
        resultsLayout.addWidget(self.resultsTable)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        resultsLayout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

        # ── Connections ────────────────────────────────────────
        self.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.modelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onModelSelected)

        self.logic = FemurMeasurementsLogic()

    def cleanup(self):
        pass

    def onModelSelected(self):
        self.applyButton.enabled = self.modelSelector.currentNode() is not None

    def onApplyButton(self):
        self.statusLabel.setText("Računam...")
        slicer.app.processEvents()
        try:
            params = dict(
                HEAD_FRACTION        = self.spHeadFrac.value,
                DISTAL_FRACTION_77   = self.spDistal77.value,
                DISTAL_FRACTION_84_85= self.spDistal8485.value,
                RANSAC_ITERS         = int(self.spRansacIters.value),
                INLIER_THRESH_MM     = self.spInlierThresh.value,
                MIN_INLIERS          = int(self.spMinInliers.value),
            )
            results = self.logic.runMeasurements(self.modelSelector.currentNode(), params)
            self._showResults(results)
            self.statusLabel.setText("Završeno.")
        except Exception as e:
            self.statusLabel.setText(f"Greška: {e}")
            logging.exception("FemurMeasurements error")

    def _showResults(self, r):
        rows = [
            ("75. Maksimalna dužina femura",        "max_femur_length"),
            ("76. Bikondilarna dužina",             "bicondylar_length"),
            ("77. Epikondilarna širina",            "epicondylar_breadth"),
            ("78. Maks. promjer glave femura",      "max_head_diameter"),
            ("79. Transverzalni subtrohanterični",  "len79"),
            ("80. AP subtrohanterični",             "len80"),
            ("81. Maks. promjer dijafize",          "max_midshaft_diameter"),
            ("82. Min. promjer dijafize",           "min_midshaft_diameter"),
            ("83. Opseg dijafize",                  "midshaft_circumference"),
            ("84. AP lateralnog kondila",           "len84"),
            ("85. AP medijalnog kondila",           "len85"),
        ]
        self.resultsTable.setRowCount(len(rows))
        for i, (label, key) in enumerate(rows):
            val = r.get(key)
            self.resultsTable.setItem(i, 0, qt.QTableWidgetItem(label))
            self.resultsTable.setItem(i, 1, qt.QTableWidgetItem(
                f"{val:.2f}" if val is not None else "N/A"))
        self.resultsTable.resizeColumnsToContents()


class FemurMeasurementsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    # ──────────────────────────────────────────────────────────
    def runMeasurements(self, modelNode, params):
        HEAD_FRACTION         = params["HEAD_FRACTION"]
        DISTAL_FRACTION_77    = params["DISTAL_FRACTION_77"]
        DISTAL_FRACTION_84_85 = params["DISTAL_FRACTION_84_85"]
        RANSAC_ITERS          = params["RANSAC_ITERS"]
        INLIER_THRESH_MM      = params["INLIER_THRESH_MM"]
        MIN_INLIERS           = params["MIN_INLIERS"]

        TROCH_SCAN_START    = 0.06
        TROCH_SCAN_END      = 0.45
        N_TROCH_SLICES      = 150
        TROCH_SMOOTH_WIN    = 7
        TROCH_BAND_HALF_FRAC = 0.007
        SUB_OFFSET_FRAC     = 0.10

        polydata = modelNode.GetPolyData()
        points   = polydata.GetPoints()
        if points is None or points.GetNumberOfPoints() == 0:
            raise RuntimeError("Model nema točaka.")

        pts = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())],
            dtype=float)

        # PCA → long axis
        pts_mean     = pts.mean(axis=0)
        pts_centered = pts - pts_mean
        _, _, Vt     = np.linalg.svd(pts_centered, full_matrices=False)
        axis         = Vt[0]
        proj         = pts_centered @ axis
        min_p, max_p = proj.min(), proj.max()
        total_length = max_p - min_p

        # Local frame
        z = axis
        x = np.cross(z, [1, 0, 0])
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [0, 1, 0])
        x /= np.linalg.norm(x)
        y  = np.cross(z, x)

        # Markup helpers
        fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Auto landmarks")

        def add_point(p, label):
            fid.AddControlPoint(p.tolist() if isinstance(p, np.ndarray) else p)
            fid.SetNthControlPointLabel(fid.GetNumberOfControlPoints() - 1, label)

        def make_line(name, p1, p2):
            line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
            line.AddControlPoint(p1.tolist() if isinstance(p1, np.ndarray) else p1)
            line.AddControlPoint(p2.tolist() if isinstance(p2, np.ndarray) else p2)
            return line

        def dist3d(a, b):
            return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

        # Sphere helpers
        def fit_sphere_ls(P):
            A        = np.column_stack([P, np.ones(len(P))])
            b        = -np.sum(P ** 2, axis=1)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            a, b_, c, d = coef
            center   = np.array([-a/2, -b_/2, -c/2])
            r2       = np.sum(center**2) - d
            return center, float(np.sqrt(max(r2, 0.0)))

        def sphere_4pts(p1, p2, p3, p4):
            P   = np.stack([p1, p2, p3, p4])
            M   = np.stack([P[1]-P[0], P[2]-P[0], P[3]-P[0]])
            rhs = 0.5 * (np.sum(P[1:]**2, axis=1) - np.sum(P[0]**2))
            if abs(np.linalg.det(M)) < 1e-10:
                return None, None
            c = np.linalg.solve(M, rhs)
            return c, float(np.linalg.norm(P[0] - c))

        def ransac(P, n_iters, thresh, min_inl, rng_):
            best = None
            for _ in range(n_iters):
                idx = rng_.choice(len(P), size=4, replace=False)
                c, r = sphere_4pts(P[idx[0]], P[idx[1]], P[idx[2]], P[idx[3]])
                if c is None or not np.isfinite(r):
                    continue
                inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < thresh
                if best is None or inl.sum() > best.sum():
                    best = inl
            cnt = 0 if best is None else int(best.sum())
            return (best if cnt >= min_inl else None), cnt

        # Proximal / distal detection
        mask_low  = proj < (min_p + HEAD_FRACTION * total_length)
        mask_high = proj > (max_p - HEAD_FRACTION * total_length)
        rng = np.random.default_rng(0)
        pts_low, pts_high = pts[mask_low], pts[mask_high]

        inl_low,  cnt_low  = ransac(pts_low,  RANSAC_ITERS, INLIER_THRESH_MM, MIN_INLIERS, rng)
        inl_high, cnt_high = ransac(pts_high, RANSAC_ITERS, INLIER_THRESH_MM, MIN_INLIERS, rng)

        score_low  = cnt_low  / max(len(pts_low),  1)
        score_high = cnt_high / max(len(pts_high), 1)

        if score_low >= score_high:
            distal_is_low = False
            proximal_pts  = pts_low
            best_inliers  = inl_low
        else:
            distal_is_low = True
            proximal_pts  = pts_high
            best_inliers  = inl_high

        if best_inliers is None:
            raise RuntimeError("Fitanje sfere glave femura nije uspjelo. Povećaj HEAD_FRACTION ili INLIER_THRESH_MM.")

        distal_tip_proj = min_p if distal_is_low else max_p

        if distal_is_low:
            distal_pts_77    = pts[proj < (min_p + DISTAL_FRACTION_77    * total_length)]
            distal_pts_8485  = pts[proj < (min_p + DISTAL_FRACTION_84_85 * total_length)]
        else:
            distal_pts_77    = pts[proj > (max_p - DISTAL_FRACTION_77    * total_length)]
            distal_pts_8485  = pts[proj > (max_p - DISTAL_FRACTION_84_85 * total_length)]

        prox_end = max_p if distal_is_low else min_p
        prox_dir = -1.0  if distal_is_low else +1.0

        # 75. Maximum femur length
        i_prox  = np.argmin(proj)
        i_dist  = np.argmax(proj)
        prox_pt = pts[i_prox]
        dist_pt = pts[i_dist]
        add_point(prox_pt, "75_MaxLen_Prox")
        add_point(dist_pt, "75_MaxLen_Dist")
        make_line("75_Maximum_femur_length", prox_pt, dist_pt)
        max_femur_length = dist3d(prox_pt, dist_pt)

        # 78. Femoral head diameter
        center78, radius78 = fit_sphere_ls(proximal_pts[best_inliers])
        max_head_diameter  = 2.0 * radius78
        d_qc = np.cross(axis, [1, 0, 0])
        if np.linalg.norm(d_qc) < 1e-6:
            d_qc = np.cross(axis, [0, 1, 0])
        d_qc /= np.linalg.norm(d_qc)
        pH1 = center78 + radius78 * d_qc
        pH2 = center78 - radius78 * d_qc
        add_point(center78, "FemurHead_Center")
        add_point(pH1, "FemurHead_Max_A")
        add_point(pH2, "FemurHead_Max_B")
        make_line("78_Max_femoral_head_diameter", pH1, pH2)

        # Anatomical directions
        shaft_center = pts_mean
        v_med = center78 - shaft_center
        v_med -= np.dot(v_med, axis) * axis
        if np.linalg.norm(v_med) < 1e-8:
            raise RuntimeError("Nije moguće izračunati medijalnu os.")
        v_med /= np.linalg.norm(v_med)
        v_lat  = -v_med
        v_ap   = np.cross(axis, v_med)
        if np.linalg.norm(v_ap) < 1e-8:
            raise RuntimeError("Nije moguće izračunati AP os.")
        v_ap /= np.linalg.norm(v_ap)

        mid_band_frac = 0.08
        mid_proj_val  = 0.5 * (min_p + max_p)
        mid_mask      = np.abs(proj - mid_proj_val) < (total_length * mid_band_frac)
        mid_band_pts  = pts[mid_mask]
        if len(mid_band_pts) > 20:
            if ((mid_band_pts - pts_mean) @ v_ap).max() > ((mid_band_pts - pts_mean) @ -v_ap).max():
                v_ap = -v_ap

        # 77. Epicondylar breadth
        distal_2d = np.column_stack([
            (distal_pts_77 - pts_mean) @ x,
            (distal_pts_77 - pts_mean) @ y])
        _, _, Vt2 = np.linalg.svd(distal_2d - distal_2d.mean(axis=0), full_matrices=False)
        ml_dir  = Vt2[0]
        proj_ml = distal_2d @ ml_dir

        epi_med = distal_pts_77[np.argmax((distal_pts_77 - pts_mean) @ v_med)]

        _d77_from_tip = np.abs((distal_pts_77 - pts_mean) @ axis - distal_tip_proj)
        _axial_mask   = (_d77_from_tip > total_length * 0.06) & (_d77_from_tip < total_length * 0.18)
        _d77_ap       = (distal_pts_77 - pts_mean) @ v_ap
        _ap_threshold = _d77_ap.min() + 0.60 * (_d77_ap.max() - _d77_ap.min())
        _ap_mask      = _d77_ap < _ap_threshold
        _epi_lat_mask = _axial_mask & _ap_mask
        _epi_lat_pool = distal_pts_77[_epi_lat_mask]
        if len(_epi_lat_pool) < 10:
            _epi_lat_pool = distal_pts_77[_axial_mask] if _axial_mask.sum() > 10 else distal_pts_77
        epi_lat = _epi_lat_pool[np.argmax((_epi_lat_pool - pts_mean) @ v_lat)]

        add_point(epi_med, "Epicondyle_Medial")
        add_point(epi_lat, "Epicondyle_Lateral")
        make_line("77_Epicondylar_breadth", epi_med, epi_lat)
        epicondylar_breadth = dist3d(epi_med, epi_lat)

        # 76. Bicondylar length
        d_to_med_76 = np.linalg.norm(distal_pts_8485 - epi_med, axis=1)
        d_to_lat_76 = np.linalg.norm(distal_pts_8485 - epi_lat, axis=1)
        med_pts_76  = distal_pts_8485[d_to_med_76 <  d_to_lat_76]
        lat_pts_76  = distal_pts_8485[d_to_lat_76 <= d_to_med_76]

        def most_distal(cpts):
            p = (cpts - pts_mean) @ axis
            return p.min() if distal_is_low else p.max()

        condylar_plane_proj = (min if distal_is_low else max)(
            most_distal(med_pts_76), most_distal(lat_pts_76))
        head_prox_proj    = (prox_pt - pts_mean) @ axis
        bicondylar_length = abs(head_prox_proj - condylar_plane_proj)
        p76_prox = pts_mean + head_prox_proj      * axis
        p76_dist = pts_mean + condylar_plane_proj * axis
        add_point(p76_prox, "76_Head_Prox_Axis")
        add_point(p76_dist, "76_Condylar_Plane")
        make_line("76_Bicondylar_length", p76_prox, p76_dist)

        # Trochanter detection
        BAND_HALF   = total_length * TROCH_BAND_HALF_FRAC
        scan_levels = np.linspace(
            prox_end + prox_dir * TROCH_SCAN_START * total_length,
            prox_end + prox_dir * TROCH_SCAN_END   * total_length,
            N_TROCH_SLICES)
        lateral_ext = np.full(N_TROCH_SLICES, np.nan)
        medial_ext  = np.full(N_TROCH_SLICES, np.nan)
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

        sm_lat = moving_avg(lateral_ext, TROCH_SMOOTH_WIN)
        sm_med = moving_avg(medial_ext,  TROCH_SMOOTH_WIN)

        gt_peak_idx = int(np.argmax(sm_lat[:int(N_TROCH_SLICES * 0.60)]))
        gt_level    = scan_levels[gt_peak_idx]
        add_point(pts_mean + gt_level * axis, "GreaterTrochanter_Peak")

        lt_search_start = gt_peak_idx + TROCH_SMOOTH_WIN
        if lt_search_start >= N_TROCH_SLICES:
            raise RuntimeError("Veliki trohanter pronađen preblizu kraja skeniranja.")
        lt_peak_idx = int(np.argmax(sm_med[lt_search_start:])) + lt_search_start
        lt_level    = scan_levels[lt_peak_idx]
        add_point(pts_mean + lt_level * axis, "LesserTrochanter_Peak")

        sub_level  = lt_level + prox_dir * SUB_OFFSET_FRAC * total_length
        sub_origin = pts_mean + sub_level * axis
        add_point(sub_origin, "Subtrochanteric_Level")

        # Subtrochanteric cross-section
        sub_plane  = vtk.vtkPlane()
        sub_plane.SetOrigin(sub_origin)
        sub_plane.SetNormal(axis)
        sub_cutter = vtk.vtkCutter()
        sub_cutter.SetCutFunction(sub_plane)
        sub_cutter.SetInputData(polydata)
        sub_cutter.Update()
        sub_poly = sub_cutter.GetOutput()
        if sub_poly.GetNumberOfPoints() < 20:
            raise RuntimeError("Subtrohanterični presjek ima premalo točaka.")
        sub_pts = np.array(
            [sub_poly.GetPoint(i) for i in range(sub_poly.GetNumberOfPoints())], dtype=float)

        # 79. Transverse subtrochanteric
        proj79 = (sub_pts - sub_origin) @ v_ap
        p79_A  = sub_pts[np.argmin(proj79)]
        p79_B  = sub_pts[np.argmax(proj79)]
        add_point(p79_A, "79_Subtro_Anterior")
        add_point(p79_B, "79_Subtro_Posterior")
        make_line("79_Transverse_subtrochanteric", p79_A, p79_B)
        len79 = dist3d(p79_A, p79_B)

        # 80. AP subtrochanteric
        proj80 = (sub_pts - sub_origin) @ v_med
        p80_A  = sub_pts[np.argmin(proj80)]
        p80_B  = sub_pts[np.argmax(proj80)]
        add_point(p80_A, "80_Subtro_Lateral")
        add_point(p80_B, "80_Subtro_Medial")
        make_line("80_AP_subtrochanteric", p80_A, p80_B)
        len80 = dist3d(p80_A, p80_B)

        # Midshaft cross-section
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
            raise RuntimeError("Presjek dijafize ima premalo točaka.")
        cut_pts = np.array(
            [cut_poly.GetPoint(i) for i in range(cut_poly.GetNumberOfPoints())], dtype=float)

        pts_2d   = np.column_stack([
            (cut_pts - mid_point) @ x,
            (cut_pts - mid_point) @ y])
        hull     = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]

        # 81. Max midshaft diameter
        D    = distance.squareform(distance.pdist(hull_pts))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        pA   = mid_point + hull_pts[i, 0] * x + hull_pts[i, 1] * y
        pB   = mid_point + hull_pts[j, 0] * x + hull_pts[j, 1] * y
        add_point(pA, "Midshaft_Max_A")
        add_point(pB, "Midshaft_Max_B")
        make_line("81_Max_midshaft_diameter", pA, pB)
        max_midshaft_diameter = dist3d(pA, pB)

        # 82. Min midshaft diameter
        min_midshaft_diameter = np.inf
        best_normal = None
        for k in range(len(hull_pts)):
            edge   = hull_pts[(k + 1) % len(hull_pts)] - hull_pts[k]
            edge  /= np.linalg.norm(edge)
            normal = np.array([-edge[1], edge[0]])
            width  = float((hull_pts @ normal).ptp())
            if width < min_midshaft_diameter:
                min_midshaft_diameter = width
                best_normal = normal
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

        # 84 & 85. Condyle AP lengths
        d_to_med = np.linalg.norm(distal_pts_8485 - epi_med, axis=1)
        d_to_lat = np.linalg.norm(distal_pts_8485 - epi_lat, axis=1)
        medial_condyle_pts  = distal_pts_8485[d_to_med <  d_to_lat]
        lateral_condyle_pts = distal_pts_8485[d_to_lat <= d_to_med]

        if len(medial_condyle_pts) < 50 or len(lateral_condyle_pts) < 50:
            raise RuntimeError(
                "Podjela kondila nije uspjela. Povećaj DISTAL_FRACTION_84_85.")

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

        return dict(
            max_femur_length       = max_femur_length,
            bicondylar_length      = bicondylar_length,
            epicondylar_breadth    = epicondylar_breadth,
            max_head_diameter      = max_head_diameter,
            len79                  = len79,
            len80                  = len80,
            max_midshaft_diameter  = max_midshaft_diameter,
            min_midshaft_diameter  = min_midshaft_diameter,
            midshaft_circumference = midshaft_circumference,
            len84                  = len84,
            len85                  = len85,
        )


class FemurMeasurementsTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_moduleLoads()

    def test_moduleLoads(self):
        self.delayDisplay("Modul je uspješno učitan.")
