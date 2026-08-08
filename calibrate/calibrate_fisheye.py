#!/usr/bin/env python3
"""
Fisheye intrinsic calibration (Kannala-Brandt) for the OV5647 + 220deg lenses.

    printed ArUco grid --> per-view corner correspondences
                                     |
                       cv2.fisheye.calibrate --> K, D
                                     |
                       pixels_to_rays() --> unit bearing rays

The output that actually matters is not the reprojection RMS, it is `pixels_to_rays`:
the bearing pipeline unprojects every detection through this model, and a systematic
error there does NOT average out over a triangulation the way centroid noise does.
So this script reports error as a function of image radius -- a fit that is great in
the middle and bad at the rim is the normal failure, and the rim is exactly where the
best-parallax marks live.

THE 90 DEGREE WALL. cv2.fisheye derives theta as atan(||(X/Z, Y/Z)||), so it cannot
represent a field angle of 90deg or more. It does not raise -- it FOLDS BACK. A ray
at 110deg off-axis projects to the same image radius as one at 70deg. Measured on
this build, with an equidistant f of 843.8 px/rad:

    theta      correct r     what cv2.fisheye returns
     88.0        1296.0        1296.0    ok
     90.0        1325.5        1325.5    ok
    110.0        1620.0        1030.9    folded back onto 70deg

These lenses are ~180deg across the frame width and ~225deg across the diagonal, so
the frame corners sit near 112deg and are past the wall. Two consequences:

  - Corner detections must be EXCLUDED FROM THE FIT. Left in, they are fitted
    against a folded-back projection, k3/k4 bend to accommodate the impossible, and
    that corrupts the radial curve across the whole field, including the part that
    had good data. The tell is an outer residual band several times the inner ones,
    which reads exactly like "not enough rim views" and provokes the opposite of
    the correct fix. This script therefore filters by field angle and refits.
  - Everything downstream needs a VALID REGION. `model_theta_limit` derives it from
    the fitted model and `pixels_to_rays` returns NaN outside it, so an unreachable
    pixel is an explicit no-answer rather than a confident wrong bearing. At 180deg
    across the width the boundary is a circle tangent to the left and right frame
    edges: 90% of the full sensor, 96% of the 2592x1296 detector crop, all of it
    the horizon band that matters. Only the corners are lost.

Calibrate ONCE at full sensor resolution and use `adapt_model` for the cropped and
scaled geometries. A crop shifts the principal point and a uniform scale multiplies
fx, fy, cx, cy; D and the valid cone are untouched. Both are exact, so there is
never a reason to recalibrate per sensor mode.

Default target is a plain ArUco GRID (markers on white), not ChArUco:
  - roughly half the ink of a checkerboard, which is mostly black by area
  - each marker is individually identified, so half-out-of-frame views still count,
    which is the only way to constrain the periphery of a lens this wide
  - it is what the wide-FOV calibration tools use (Kalibr's AprilGrid is the same
    idea), so the precision is known-adequate
A4 is large enough. You do not need one view that fills the frame, you need many
views that between them cover it -- see the capture notes at the bottom.

Usage:
    # 1. print at 100% scale on A4, mount FLAT, then measure a marker with calipers
    python calibrate_fisheye.py --make_board board.png

    # 2. calibrate from a folder of stills. --fov is not optional on a lens this
    #    wide: it seeds f, and the seed is what lets pass 1 exclude the corners.
    python calibrate_fisheye.py --images calib/cam0 --marker 0.0351 \
        --separation 0.0125 --fov 180 --out calib/cam0.json

    # 3. sanity-check a saved model on one frame -- writes a valid-region overlay
    #    and a rectilinear view you can eyeball for straight lines
    python calibrate_fisheye.py --check calib/cam0.json --image frame.jpg

    # 4. optionally derive the model for what the detector sees (a 2048x1024 band
    #    scaled to 1280x640) from the full-resolution fit. sender.py does this
    #    itself at startup via estimate.Camera, so it is only needed standalone.
    python calibrate_fisheye.py --adapt calib/cam0.json \
        --crop_left 544 --crop_top 460 --crop_size 2048x1024 \
        --scale_to 1280x640 --out calib/cam0_det.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Fewer bits per marker survives the tangential smear at the rim far better than
# DICT_5X5/6X6 does. Detection failures out there are the main reason calibrations
# come back with good centre error and garbage periphery.
ARUCO_DICT = "DICT_4X4_100"

# The hard ceiling described in the module docstring. Not a tuning knob: it is where
# cv2.fisheye's atan(r) parameterisation stops being injective. --theta_max trims
# further inside it, which is worth a degree or two because the projection is stiff
# as Z approaches 0, but nothing can raise it.
THETA_MODEL_MAX_DEG = 90.0


# --------------------------------------------------------------------------- #
# OpenCV 4.7 rewrote the aruco API. JetPack still ships 4.5.x on some releases,
# so probe for both rather than pinning a version nobody can install.
# --------------------------------------------------------------------------- #
def make_dictionary():
    name = getattr(cv2.aruco, ARUCO_DICT)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(name)
    return cv2.aruco.Dictionary_get(name)


def make_board(args, dictionary):
    """ArUco grid by default; ChArUco if you already have one printed."""
    if args.target == "charuco":
        if hasattr(cv2.aruco, "CharucoDetector"):
            return cv2.aruco.CharucoBoard((args.nx, args.ny), args.marker,
                                          args.marker * 0.75, dictionary)
        return cv2.aruco.CharucoBoard_create(args.nx, args.ny, args.marker,
                                             args.marker * 0.75, dictionary)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.GridBoard((args.nx, args.ny), args.marker,
                                   args.separation, dictionary)
    return cv2.aruco.GridBoard_create(args.nx, args.ny, args.marker,
                                      args.separation, dictionary)


def board_image(board, w: int, h: int, margin: int):
    if hasattr(board, "generateImage"):
        return board.generateImage((w, h), marginSize=margin, borderBits=1)
    return board.draw((w, h), marginSize=margin, borderBits=1)


def marker_objpoints(board) -> dict:
    """id -> (4,3) marker corner positions in board coordinates."""
    if hasattr(board, "getObjPoints"):
        objp, ids = board.getObjPoints(), board.getIds()
    else:
        objp, ids = board.objPoints, board.ids
    return {int(i): np.asarray(o, dtype=np.float64).reshape(4, 3)
            for i, o in zip(np.asarray(ids).reshape(-1), objp)}


def charuco_objpoints(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)


def detector_params(target: str):
    p = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "ArucoDetector")
         else cv2.aruco.DetectorParameters_create())
    # Markers at the rim are compressed to a fraction of their on-axis size and
    # their edges bow, so loosen the two thresholds that reject exactly that.
    p.minMarkerPerimeterRate = 0.01
    p.polygonalApproxAccuracyRate = 0.08
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 8
    if target == "aruco":
        # Marker corners ARE the measurement here, so refinement matters. SUBPIX
        # rather than CONTOUR/APRILTAG deliberately: those fit straight lines to
        # the marker edges, and under this much barrel distortion the edges are
        # visibly curved, which biases the corner outward at the rim.
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        p.cornerRefinementMinAccuracy = 0.01
    return p


def detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def correspondences(gray, board, dictionary, params, target, id2obj):
    """-> (objp Nx1x3, imgp Nx1x2) or (None, None)."""
    if target == "charuco":
        if hasattr(cv2.aruco, "CharucoDetector"):
            det = cv2.aruco.CharucoDetector(board)
            det.setDetectorParameters(params)
            cc, ci, _, _ = det.detectBoard(gray)
        else:
            mc, mi, _ = detect_markers(gray, dictionary, params)
            if mi is None or len(mi) == 0:
                return None, None
            n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
            if n is None or n < 1:
                return None, None
        if ci is None or len(ci) < 1:
            return None, None
        allo = charuco_objpoints(board)
        return (allo[ci.reshape(-1)].reshape(-1, 1, 3),
                np.asarray(cc, dtype=np.float64).reshape(-1, 1, 2))

    corners, ids, _ = detect_markers(gray, dictionary, params)
    if ids is None or len(ids) == 0:
        return None, None
    mids, quads = [], []
    for mid, c in zip(np.asarray(ids).reshape(-1), corners):
        if int(mid) in id2obj:
            mids.append(int(mid))
            quads.append(np.asarray(c, dtype=np.float64).reshape(4, 2))
    mids, quads = reject_spurious(mids, quads)
    if not mids:
        return None, None
    return (np.concatenate([id2obj[m] for m in mids]).reshape(-1, 1, 3),
            np.concatenate(quads).reshape(-1, 1, 2))


def reject_spurious(mids, quads):
    """Throw out detections that cannot belong to the physical board.

    The detector parameters are deliberately loose (see detector_params) so that
    markers compressed to a sliver at the rim still register. In a dim, cluttered
    room that same looseness invents markers out of lamp filaments, floor mesh and
    picture frames -- and roughly one in five of those false positives decodes as an
    id that IS on the board. That is far worse than a missed detection: it plants an
    object point hundreds of pixels from where it belongs, wrecks that view's pose,
    and enough of them stall the global solve on its first step with no error at all,
    just an unchanged K and a D of zeros.

    Two filters, both purely geometric, so neither needs a calibration to work:

      1. An id decoded twice in one view is ambiguous -- there is no way to tell the
         real one from the impostor, so drop BOTH.
      2. Keep only the largest spatially connected group. The board is one rigid
         contiguous object, so its markers form a single blob; a false positive on
         the ceiling is its own isolated group. Markers are linked when their centres
         are within 4x the larger of their side lengths -- neighbours on this grid sit
         at 1.25x (edge) to 1.8x (diagonal), so 4x tolerates heavy foreshortening at
         the rim while still rejecting anything across the frame.
    """
    if not mids:
        return [], []
    keep = [k for k, m in enumerate(mids) if mids.count(m) == 1]
    if len(keep) < 2:
        return ([mids[k] for k in keep], [quads[k] for k in keep])

    ctr = np.array([quads[k].mean(0) for k in keep])
    side = np.array([max(np.linalg.norm(quads[k][0] - quads[k][1]),
                         np.linalg.norm(quads[k][1] - quads[k][2])) for k in keep])
    n = len(keep)
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=2)
    link = d <= 4.0 * np.maximum(side[:, None], side[None, :])

    # Connected components by breadth-first flood, then take the biggest.
    seen, best = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in np.nonzero(link[a])[0]:
                if b not in seen:
                    seen.add(int(b))
                    stack.append(int(b))
        if len(comp) > len(best):
            best = comp
    sel = [keep[c] for c in sorted(best)]
    return ([mids[k] for k in sel], [quads[k] for k in sel])


# --------------------------------------------------------------------------- #
# The model itself. This is the function the bearing pipeline imports.
# --------------------------------------------------------------------------- #
def kb_radial(theta, D) -> np.ndarray:
    """Kannala-Brandt forward radial map: field angle -> normalised image radius.

        r_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
    """
    k = np.asarray(D, dtype=np.float64).reshape(-1)
    t = np.asarray(theta, dtype=np.float64)
    t2 = t * t
    return t * (1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)


def model_theta_limit(D, hard_deg: float = THETA_MODEL_MAX_DEG) -> float:
    """Largest field angle this model can be trusted to invert, in radians.

    Two independent ceilings, whichever bites first:

      1. `hard_deg`, the fold-back wall from the module docstring. Nothing fitted
         by cv2.fisheye.calibrate carries information beyond it however wide the
         lens physically is.
      2. The first turning point of r_d(theta). D is a quartic in theta^2 and a
         fit with a negative tail can stop being monotonic before `hard_deg`.
         Past a turning point radius no longer identifies angle -- the Newton
         solve below has two roots and converges to whichever the seed was
         nearer -- so the model is unusable there even though it is inside the
         wall. Worth checking rather than assuming: this is what a calibration
         with thin rim coverage tends to produce, because k3/k4 are then fitted
         almost entirely by extrapolation.
    """
    t = np.linspace(0.0, np.radians(hard_deg), 4001)
    turn = np.nonzero(np.diff(kb_radial(t, D)) <= 0.0)[0]
    return float(t[turn[0]]) if turn.size else float(np.radians(hard_deg))


def pixels_to_rays(uv, K, D, theta_max: float | None = None,
                   iters: int = 12) -> np.ndarray:
    """Pixels -> unit rays in the camera frame. NaN where the model cannot reach.

    Deliberately NOT cv2.fisheye.undistortPoints: that returns pinhole-normalised
    x/z, y/z, which diverges as the field angle approaches 90deg and is meaningless
    beyond it. Instead invert the Kannala-Brandt radial polynomial for theta
    directly and build the ray on the unit sphere, which stays well-behaved right
    up to the valid limit.

    Rows outside the valid cone come back as NaN, not as a plausible wrong bearing.
    That distinction is the entire safety property here: a folded-back ray from a
    frame corner is a unit vector like any other and looks like a real detection to
    everything downstream, so it would quietly poison a triangulation instead of
    being dropped. Callers gate on np.isnan(rays[:, 0]).

    theta_max (radians) defaults to model_theta_limit(D). Pass the `theta_max_deg`
    stored alongside the calibration to also respect the angle the data actually
    reached, which is usually tighter than the model limit.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    k = np.asarray(D, dtype=np.float64).reshape(-1)

    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    r_d = np.hypot(x, y)
    phi = np.arctan2(y, x)

    if theta_max is None:
        theta_max = model_theta_limit(D)
    # Compare in normalised radius: one comparison, no solve needed to decide.
    # With fx != fy this locus is an ellipse in pixel space, which is right --
    # the anisotropy is part of the model, not an artefact to be averaged away.
    ok = r_d <= kb_radial(theta_max, D)

    # Seed invalid rows at 0 so a divergent solve there cannot produce inf and
    # contaminate neighbouring arithmetic before the mask is applied.
    theta = np.where(ok, r_d, 0.0)
    for _ in range(iters):
        t2 = theta * theta
        poly = 1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4
        dpoly = (2 * k[0] * theta + 4 * k[1] * theta**3
                 + 6 * k[2] * theta**5 + 8 * k[3] * theta**7)
        f = theta * poly - r_d
        df = poly + theta * dpoly
        theta = theta - f / np.where(np.abs(df) < 1e-12, 1e-12, df)

    rays = np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)
    rays[~ok] = np.nan
    return rays


def valid_mask(K, D, size, theta_max: float | None = None) -> np.ndarray:
    """uint8 HxW, 255 exactly where pixels_to_rays returns a real bearing.

    Built by broadcasting two 1-D axes rather than an mgrid, which matters at
    2592x1944: the separable form touches ~40 MB instead of ~160.
    """
    w, h = int(size[0]), int(size[1])
    K = np.asarray(K, dtype=np.float64)
    if theta_max is None:
        theta_max = model_theta_limit(D)
    x = (np.arange(w, dtype=np.float64) - K[0, 2]) / K[0, 0]
    y = (np.arange(h, dtype=np.float64) - K[1, 2]) / K[1, 1]
    r_d = np.hypot(x[None, :], y[:, None])
    return (r_d <= kb_radial(theta_max, D)).astype(np.uint8) * 255


def adapt_model(model: dict, crop_left: int = 0, crop_top: int = 0,
                crop_size=None, scale_to=None) -> dict:
    """Move a fitted model onto a cropped and/or scaled frame. Exact, not fitted.

    Kannala-Brandt keeps all the nonlinearity in D, which acts on coordinates
    already normalised by fx, fy. So a crop is a shift of the principal point and
    a uniform scale is a multiplication of fx, fy, cx, cy; D and the valid cone
    carry over untouched. Calibrate once at full sensor resolution and derive the
    rest -- a second calibration at a second resolution would only add a second
    set of errors to reconcile.

    For sender.py's defaults (2592x1944 -> centred 2592x1296 band -> 1280x640):
        adapt_model(m, crop_top=(1944 - 1296) // 2,
                    crop_size=(2592, 1296), scale_to=(1280, 640))

    A non-uniform scale would need fx and fy scaled by different factors, which is
    representable, but sender.py scales both axes by 2.025 precisely so that box
    coordinates map with one factor -- so refuse anything else rather than let a
    silent aspect change through.
    """
    K = np.array(model["K"], dtype=np.float64)
    w, h = model["image_size"]
    if crop_size is not None:
        w, h = int(crop_size[0]), int(crop_size[1])
    K[0, 2] -= crop_left
    K[1, 2] -= crop_top
    if scale_to is not None:
        sx, sy = float(scale_to[0]) / w, float(scale_to[1]) / h
        if abs(sx - sy) > 1e-9:
            raise ValueError(f"non-uniform scale {sx:.6f} x {sy:.6f}: that changes "
                             f"the aspect ratio, which this model cannot absorb "
                             f"into K without also splitting fx/fy")
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        w, h = int(scale_to[0]), int(scale_to[1])
    out = dict(model)
    out["K"] = K.tolist()
    out["image_size"] = [w, h]
    out["derived_from"] = {"image_size": model["image_size"],
                           "crop_left": crop_left, "crop_top": crop_top,
                           "crop_size": list(crop_size) if crop_size else None,
                           "scale_to": list(scale_to) if scale_to else None}
    return out


def rectilinear_view(img, K, D, out_fov_deg: float = 90.0, out_size=None):
    """Pinhole reprojection over a chosen sub-field, for the straight-line check.

    cv2.fisheye's own undistort is useless on this lens: a rectilinear image of a
    180deg field needs infinite extent, so estimateNewCameraMatrixForUndistortRectify
    either crushes the scene into the middle few percent of the output or lands
    most of the map outside the source. Pick a sub-field instead. 90deg keeps the
    check meaningful -- straight edges must come out straight -- while staying well
    inside the valid cone. Raise it to look further out, at the cost of the corners
    stretching enough to hide the very curvature you are checking for.
    """
    h, w = (img.shape[:2] if out_size is None else (int(out_size[1]), int(out_size[0])))
    K = np.asarray(K, dtype=np.float64)
    fp = (w / 2.0) / np.tan(np.radians(out_fov_deg / 2.0))
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    X, Y = (xx - w / 2.0) / fp, (yy - h / 2.0) / fp
    theta = np.arctan(np.hypot(X, Y))  # pinhole, so < 90deg by construction
    phi = np.arctan2(Y, X)
    r_d = kb_radial(theta, D)
    return cv2.remap(img,
                     (K[0, 0] * r_d * np.cos(phi) + K[0, 2]).astype(np.float32),
                     (K[1, 1] * r_d * np.sin(phi) + K[1, 2]).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def angle_between(uv_a, uv_b, K, D, theta_max: float | None = None) -> float:
    """Angular separation in degrees between two pixels. This is the quantity the
    parallax pipeline actually consumes, so it is the quantity worth validating
    against a physically measured angle.

    NaN if either pixel is outside the valid cone -- deliberately propagated rather
    than clipped to the boundary, since a silently shortened angle is worse than an
    obvious refusal. Pass theta_max=np.pi/2 to force the raw model extrapolation,
    which is only appropriate for reporting (see field_angle_report)."""
    v = pixels_to_rays(np.vstack([uv_a, uv_b]), K, D, theta_max=theta_max)
    return float(np.degrees(np.arccos(np.clip(v[0] @ v[1], -1.0, 1.0))))


def field_angle_report(K, D, size, theta_max: float) -> dict:
    """Where the valid cone lands on this particular frame."""
    w, h = int(size[0]), int(size[1])
    K = np.asarray(K, dtype=np.float64)
    r_lim = kb_radial(theta_max, D)
    # Field angle the model puts at each frame extremity. Evaluated with the cone
    # lifted as far as the polynomial stays invertible -- past 90deg that is pure
    # extrapolation and NOT usable for bearings, but as a report it is the thing
    # worth knowing: how far beyond the cone this frame actually reaches. Capping
    # at 90 instead would just print NaN for every corner of a lens like this.
    raw = model_theta_limit(D, hard_deg=179.0)
    def ang(u, v):
        r = pixels_to_rays([[u, v]], K, D, theta_max=raw)[0]
        if np.isnan(r[0]):
            return float("nan")
        return float(np.degrees(np.arccos(np.clip(r[2], -1.0, 1.0))))
    return {"theta_max_deg": float(np.degrees(theta_max)),
            "r_valid_px": float(K[0, 0] * r_lim),
            "edge_x_deg": ang(w - 1.0, h / 2.0),
            "edge_y_deg": ang(w / 2.0, h - 1.0),
            "corner_deg": ang(w - 1.0, h - 1.0),
            "valid_frac": float((valid_mask(K, D, (w, h), theta_max) > 0).mean()),
            "mrad_per_px": float(1000.0 / K[0, 0])}


def filter_by_theta(obj_pts, img_pts, used, K, D, theta_max, min_corners):
    """Drop correspondences outside the valid cone. -> lists, n_pts, n_views.

    Individual marker corners are dropped, not whole markers: cv2.fisheye.calibrate
    sums an independent residual per correspondence, so a marker straddling the
    boundary still contributes its inboard corners. That matters, because the
    markers nearest the boundary are the ones carrying most of the information
    about k3 and k4 -- rejecting them wholesale would trade the fold-back problem
    for an under-constrained tail, which produces the non-monotonic D that
    model_theta_limit exists to catch.
    """
    K = np.asarray(K, dtype=np.float64)
    r_lim = kb_radial(theta_max, D)
    o2, i2, u2, dropped = [], [], [], 0
    for o, i, p in zip(obj_pts, img_pts, used):
        uv = i.reshape(-1, 2)
        x = (uv[:, 0] - K[0, 2]) / K[0, 0]
        y = (uv[:, 1] - K[1, 2]) / K[1, 1]
        keep = np.hypot(x, y) <= r_lim
        dropped += int((~keep).sum())
        if int(keep.sum()) < min_corners:
            continue
        o2.append(np.ascontiguousarray(o.reshape(-1, 3)[keep]).reshape(-1, 1, 3))
        i2.append(np.ascontiguousarray(i.reshape(-1, 2)[keep]).reshape(-1, 1, 2))
        u2.append(p)
    return o2, i2, u2, dropped, len(obj_pts) - len(o2)


def reject_by_residual(obj_pts, img_pts, used, rvecs, tvecs, K, D, min_corners,
                       floor_px=3.0, k_mad=6.0):
    """Drop correspondences the fitted model cannot explain. -> lists, n_pts, n_views.

    The geometric filters in reject_spurious cannot see a false positive that lands
    ON the board, and a warped or mis-measured target shows up the same way. Once
    there is a model, though, those points are obvious: they reproject nowhere near
    where they were found. Cut at k_mad times the median residual rather than a fixed
    threshold so this scales with however good the fit currently is, with a floor so
    a very good fit does not start trimming honest measurement noise.
    """
    all_e = []
    for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rv, tv, K, D)
        all_e.append(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1))
    med = float(np.median(np.concatenate(all_e)))
    thresh = max(floor_px, k_mad * med)

    o2, i2, u2, dropped = [], [], [], 0
    for o, i, p, e in zip(obj_pts, img_pts, used, all_e):
        keep = e <= thresh
        dropped += int((~keep).sum())
        if int(keep.sum()) < min_corners:
            continue
        o2.append(np.ascontiguousarray(o.reshape(-1, 3)[keep]).reshape(-1, 1, 3))
        i2.append(np.ascontiguousarray(i.reshape(-1, 2)[keep]).reshape(-1, 1, 2))
        u2.append(p)
    return o2, i2, u2, dropped, len(obj_pts) - len(o2), thresh, med


def run_fit(obj_pts, img_pts, size, seed_K, use_guess: bool):
    """One cv2.fisheye.calibrate call. -> (rms, K, D, rvecs, tvecs) or raises.

    CALIB_CHECK_COND deliberately omitted: it aborts the whole solve on a single
    ill-conditioned view rather than telling you which, and near-degenerate views
    are unavoidable when you are deliberately shooting the rim.
    """
    K = np.array(seed_K, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    if use_guess:
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_pts]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_pts]
    return cv2.fisheye.calibrate(
        obj_pts, img_pts, size, K, D, rvecs, tvecs, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))


# --------------------------------------------------------------------------- #
def calibrate(args) -> int:
    dictionary = make_dictionary()
    board = make_board(args, dictionary)
    params = detector_params(args.target)
    id2obj = {} if args.target == "charuco" else marker_objpoints(board)
    n_full = (4 * args.nx * args.ny if args.target == "aruco"
              else (args.nx - 1) * (args.ny - 1))

    paths = sorted(sum([glob.glob(os.path.join(args.images, e))
                        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    if not paths:
        print(f"[fatal] no images in {args.images}")
        return 2
    print(f"[data] {len(paths)} images, target={args.target}, "
          f"{n_full} corners at full visibility")

    obj_pts, img_pts, used, size, coverage = [], [], [], None, None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [skip] unreadable: {os.path.basename(p)}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
            coverage = np.zeros(gray.shape[:2], dtype=np.uint8)
        elif (gray.shape[1], gray.shape[0]) != size:
            # Mixing resolutions silently rescales fx/cx and produces a model that
            # fits nothing. Reject rather than average.
            print(f"  [skip] size {gray.shape[1]}x{gray.shape[0]} != {size}: "
                  f"{os.path.basename(p)}")
            continue

        o, i = correspondences(gray, board, dictionary, params, args.target, id2obj)
        n = 0 if o is None else len(o)
        if n < args.min_corners or (args.full_board_only and n < n_full):
            print(f"  [skip] {n:3d} corners: {os.path.basename(p)}")
            continue

        obj_pts.append(o.astype(np.float64))
        img_pts.append(i.astype(np.float64))
        used.append(p)
        for c in i.reshape(-1, 2).astype(int):
            cv2.circle(coverage, tuple(c), 14, 255, -1)
        print(f"  [ok]   {n:3d} corners: {os.path.basename(p)}")

    if len(obj_pts) < 8:
        print(f"[fatal] only {len(obj_pts)} usable views; want 25+")
        return 3

    # Coverage is the single best predictor of whether this calibration is any good.
    # Report it as a fraction of the frame and as an image you can actually look at.
    cov_frac = float((coverage > 0).mean())
    cov_path = os.path.splitext(args.out)[0] + "_coverage.png"
    cv2.imwrite(cov_path, coverage)
    print(f"\n[coverage] {cov_frac * 100:.1f}% of frame area saw a corner "
          f"-> {cov_path}")
    if cov_frac < 0.45:
        print("[coverage] WARNING: thin. The periphery is almost certainly "
              "under-constrained; the model will extrapolate there and bearings "
              "at the rim will carry a systematic error.")

    theta_guard = np.radians(min(args.theta_max, THETA_MODEL_MAX_DEG))
    seed_K = np.zeros((3, 3), dtype=np.float64)
    if args.fov:
        # An equidistant seed. Extreme lenses can converge to a local minimum from
        # a cold start, we already know roughly what f should be, and pass 1 needs
        # some estimate of f before it can say which corners are past the wall.
        f = (size[0] / 2.0) / np.radians(args.fov / 2.0)
        seed_K[:] = [[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1]]
        print(f"[init] equidistant seed from --fov {args.fov}: f = {f:.1f} px, "
              f"so {np.degrees(theta_guard):.0f}deg lands at r = "
              f"{f * theta_guard:.0f} px of {np.hypot(*size) / 2:.0f} to the corner")
    else:
        print("[init] no --fov given, so pass 1 runs cold and unfiltered. On a lens "
              "past 180deg supply it: without an f estimate there is no way to tell "
              "which detections are beyond the fold-back wall before fitting.")

    def guard(K_est, D_est, label):
        """Drop out-of-cone correspondences and report. Returns False if too few
        views survive."""
        nonlocal obj_pts, img_pts, used
        lim = model_theta_limit(D_est)
        cut = min(lim, theta_guard)
        obj_pts, img_pts, used, n_drop, v_drop = filter_by_theta(
            obj_pts, img_pts, used, K_est, D_est, cut, args.min_corners)
        why = "model turning point" if lim < theta_guard else "--theta_max"
        print(f"\n[guard] {label}: cut at {np.degrees(cut):.1f}deg ({why}), "
              f"dropped {n_drop} corners past it"
              + (f", {v_drop} views left under --min_corners" if v_drop else "")
              + f", {len(obj_pts)} views remain")
        if lim < theta_guard:
            print(f"[guard] WARNING: this D's radial map turns over at "
                  f"{np.degrees(lim):.1f}deg, inside the 90deg wall. The tail "
                  f"(k3, k4) is being set by extrapolation rather than data -- "
                  f"get more coverage between 60 and 85deg.")
        if len(obj_pts) < 8:
            print(f"[fatal] only {len(obj_pts)} views survive the guard")
            return False
        return True

    # The guard has to run BEFORE the first fit, not just between fits. Points past
    # the wall do not merely bias the solve, they can kill it outright: their folded
    # back radii make a view's corner directions collinear and
    # cv2.fisheye's InitExtrinsics aborts on `fabs(norm_u1) > 0`. That is what the
    # equidistant seed buys -- an f estimate good enough to locate the wall before
    # anything has been fitted. Pass 2 then re-cuts with the real K and D, since the
    # boundary moves once D is no longer zero.
    if args.fov:
        if not guard(seed_K, np.zeros(4), "pre-fit, from the --fov seed"):
            return 3
    try:
        fit = run_fit(obj_pts, img_pts, size, seed_K, bool(args.fov))
    except cv2.error as e:
        print(f"[fatal] cv2.fisheye.calibrate (pass 1): {e}\n"
              f"        InitExtrinsics/norm_u1 here means detections past the "
              f"90deg wall reached the solver: supply --fov so they can be cut "
              f"before it runs. A shape assertion instead means non-uniform corner "
              f"counts; retry with --full_board_only.")
        return 4
    print(f"[pass 1] RMS {fit[0]:.4f} px over {len(obj_pts)} views, "
          f"f = {fit[1][0, 0]:.1f} px")

    if not guard(fit[1], fit[2], "post-fit, from the fitted K and D"):
        return 3
    # Reject and refit REPEATEDLY, not once. The first fit is computed from data that
    # still contains the outliers, so its residuals only identify the grossest of
    # them, and a threshold set from that fit's inflated median is far too permissive.
    # Each round the fit tightens, the median drops, the threshold follows it down and
    # exposes the next tier. Converges in three or four rounds; the loop stops when a
    # round drops nothing, so a clean dataset costs one extra fit and no data.
    for rnd in range(1, args.reject_rounds + 1):
        if args.no_reject:
            break
        obj_pts, img_pts, used, n_drop, v_drop, thr, med = reject_by_residual(
            obj_pts, img_pts, used, fit[3], fit[4], fit[1], fit[2],
            args.min_corners, k_mad=args.reject_k)
        print(f"[reject {rnd}] median residual {med:.2f} px, cutting above "
              f"{thr:.2f}: dropped {n_drop} correspondences"
              + (f" and {v_drop} views" if v_drop else "")
              + f", {len(obj_pts)} views remain")
        if len(obj_pts) < 8:
            print(f"[fatal] only {len(obj_pts)} views survive outlier rejection")
            return 3
        if n_drop == 0:
            break
        try:
            fit = run_fit(obj_pts, img_pts, size, seed_K, bool(args.fov))
        except cv2.error as e:
            print(f"[fatal] cv2.fisheye.calibrate (reject round {rnd}): {e}")
            return 4
        print(f"[pass {rnd + 1}] RMS {fit[0]:.4f} px over {len(obj_pts)} views, "
              f"f = {fit[1][0, 0]:.1f} px")

    rms, K, D, rvecs, tvecs = fit
    theta_max = min(model_theta_limit(D), theta_guard)
    print(f"\n[fit] overall RMS {rms:.4f} px over {len(obj_pts)} views")
    print(f"      fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"      D={D.reshape(-1)}")

    # Per-view and per-radius residuals. The radial breakdown is the diagnostic:
    # if the outer bins are several times the inner ones, the model is fitting the
    # centre at the expense of the rim.
    radii, errs, per_view = [], [], []
    for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rv, tv, K, D)
        e = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        per_view.append(float(np.sqrt((e ** 2).mean())))
        radii.append(np.linalg.norm(i.reshape(-1, 2)
                                    - np.array([K[0, 2], K[1, 2]]), axis=1))
        errs.append(e)
    radii, errs = np.concatenate(radii), np.concatenate(errs)

    # Bands are fractions of the VALID radius, not of the distance to the frame
    # corner. Normalising to the corner would put the outermost band almost
    # entirely in the region that was just excluded, so it would read "no data,
    # unconstrained" on a perfectly good calibration and hide a genuinely thin
    # band just inside the cut.
    rep = field_angle_report(K, D, size, theta_max)
    r_max = rep["r_valid_px"]
    print("\n[residual vs image radius]   (fractions of the "
          f"{r_max:.0f} px valid radius)")
    print("   radius band          n     RMS px    field angle")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        m = (radii >= lo * r_max) & (radii < hi * r_max)
        band = f"{np.degrees(hi * theta_max):.0f}deg" if hi <= 1.0 else "cut"
        if m.sum():
            print(f"   {lo:.1f}-{hi:.1f} r_valid  {m.sum():6d}   "
                  f"{np.sqrt((errs[m] ** 2).mean()):8.3f}    to {band}")
        else:
            print(f"   {lo:.1f}-{hi:.1f} r_valid  {0:6d}   "
                  f"{'--':>8}    to {band}  NO DATA, extrapolated")

    print("\n[worst views]")
    for e, p in sorted(zip(per_view, used), reverse=True)[:5]:
        print(f"   {e:.3f} px  {os.path.basename(p)}")

    # How the fitted cone lands on this frame, as a cross-check against the
    # physically measured FOV. A large disagreement means one of them is wrong.
    vm = valid_mask(K, D, size, theta_max)
    seen_in_valid = float((coverage[vm > 0] > 0).mean())
    print(f"\n[valid region] theta_max {rep['theta_max_deg']:.1f} deg "
          f"-> radius {r_max:.0f} px, {rep['valid_frac'] * 100:.1f}% of the frame")
    print(f"   model puts the frame edges at {rep['edge_x_deg']:.1f} deg across "
          f"and {rep['edge_y_deg']:.1f} deg down, corners at "
          f"{rep['corner_deg']:.1f} deg")
    print(f"   implied full-width FOV {2 * rep['edge_x_deg']:.1f} deg "
          f"(cross-check against your protractor measurement)")
    print(f"   angular resolution {rep['mrad_per_px']:.3f} mrad/px "
          f"({np.degrees(1.0 / K[0, 0]):.4f} deg/px)")
    print(f"   corners saw {seen_in_valid * 100:.1f}% of the valid region "
          f"(this is the number that matters, not the whole-frame figure above)")

    mask_path = os.path.splitext(args.out)[0] + "_validmask.png"
    cv2.imwrite(mask_path, vm)
    # Coverage with the excluded region tinted, so one glance separates "did not
    # shoot there" from "cannot be used there".
    cov_rgb = cv2.cvtColor(coverage, cv2.COLOR_GRAY2BGR)
    cov_rgb[vm == 0] = (0, 0, 110)
    cv2.imwrite(cov_path, cov_rgb)
    print(f"   -> {mask_path}  (red in {os.path.basename(cov_path)} = outside cone)")

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": "kannala_brandt", "image_size": list(size),
                   "K": K.tolist(), "D": D.reshape(-1).tolist(),
                   # Which image orientation this model is valid for. sender.py
                   # cross-checks it against its own --no-rotate; a mismatch has
                   # no geometric symptom, only 180-deg-wrong colour and bearing.
                   "rotated_180": not args.captured_no_rotate,
                   "theta_max_deg": rep["theta_max_deg"],
                   "r_valid_px": r_max, "valid_frac": rep["valid_frac"],
                   "rms_px": float(rms), "n_views": len(obj_pts),
                   "coverage_frac": cov_frac,
                   "coverage_frac_in_cone": seen_in_valid,
                   "implied_hfov_deg": 2 * rep["edge_x_deg"],
                   "implied_corner_deg": rep["corner_deg"],
                   "target": args.target, "marker_m": args.marker,
                   "separation_m": args.separation,
                   "grid": [args.nx, args.ny]}, f, indent=2)
    print(f"\n[done] -> {args.out}")

    rim = errs[radii > 0.8 * r_max]
    rim_rms = float(np.sqrt((rim ** 2).mean())) if rim.size else 0.0
    if rms > 1.0:
        print("[verdict] RMS > 1.0 px. Do not build on this. Usual causes, in "
              "order: warped board, focus shifted mid-capture, mixed sensor "
              "modes, too few tilted views.")
    elif rim.size < 200:
        print(f"[verdict] centre is fine but only {rim.size} points sit in the "
              f"outer fifth of the cone, which is not enough to pin k3/k4. Shoot "
              f"more views with the board out at {0.8 * rep['theta_max_deg']:.0f}"
              f"-{rep['theta_max_deg']:.0f}deg off-axis -- that is roughly the "
              f"band from {0.8 * r_max:.0f} to {r_max:.0f} px out, so the board "
              f"should straddle the left or right frame edge at mid-height.")
    elif rim_rms > 3 * rms:
        print(f"[verdict] centre fits ({rms:.3f}), rim does not ({rim_rms:.3f}). "
              f"More views in the outer cone. Note this is now measured INSIDE "
              f"the {rep['theta_max_deg']:.0f}deg cut, so unlike an unguarded fit "
              f"it cannot be the fold-back -- it is genuinely thin data out there. "
              f"(cv2.omnidir is not an option on this build: ccalib is an "
              f"opencv_contrib module and is absent from the installed 4.8.0.)")
    else:
        print("[verdict] usable inside the cone. Two things left, neither of which "
              "reprojection RMS can tell you:")
        print("   1. Validate ANGLES physically. Put two marks a measured angle "
              "apart and compare against angle_between().")
        print("   2. Make sure every consumer honours the NaN from pixels_to_rays "
              "rather than assuming a bearing came back.")
    return 0


def check(args) -> int:
    with open(args.check, encoding="utf-8") as f:
        m = json.load(f)
    K, D = np.array(m["K"]), np.array(m["D"])
    img = cv2.imread(args.image)
    if img is None:
        print(f"[fatal] unreadable: {args.image}")
        return 2
    h, w = img.shape[:2]
    if [w, h] != m["image_size"]:
        print(f"[fatal] model is for {m['image_size']}, image is {[w, h]}. "
              f"A model does not transfer across sensor modes.")
        return 3

    theta_max = (np.radians(m["theta_max_deg"]) if "theta_max_deg" in m
                 else model_theta_limit(D))
    rep = field_angle_report(K, D, (w, h), theta_max)
    print(f"[model] {args.check}  RMS {m['rms_px']:.3f} px  "
          f"implied HFOV {m['implied_hfov_deg']:.1f} deg")
    print(f"[cone]  valid to {rep['theta_max_deg']:.1f} deg off-axis "
          f"= {rep['r_valid_px']:.0f} px radius, {rep['valid_frac'] * 100:.1f}% "
          f"of the frame. Outside it pixels_to_rays returns NaN.")

    print("\n[field angle at each image position]  (deg off optical axis, "
          "'--' = outside the cone)")
    for fy in (0.1, 0.5, 0.9):
        row = []
        for fx in (0.02, 0.25, 0.5, 0.75, 0.98):
            v = pixels_to_rays([[fx * w, fy * h]], K, D, theta_max=theta_max)[0]
            row.append("    --" if np.isnan(v[0]) else
                       f"{np.degrees(np.arccos(np.clip(v[2], -1, 1))):6.1f}")
        print(f"   y={fy:.1f}: " + " ".join(row))

    stem = os.path.splitext(args.image)[0]
    vm = valid_mask(K, D, (w, h), theta_max)
    cv2.imwrite(stem + "_validmask.png", vm)

    # The overlay is the artefact worth actually looking at: the boundary drawn on
    # the real scene, with the dead corners dimmed and iso-angle rings for scale.
    ov = img.copy()
    ov[vm == 0] = (ov[vm == 0] * 0.35).astype(np.uint8)
    for deg in (30, 60, 75):
        if deg < rep["theta_max_deg"]:
            r = K[0, 0] * kb_radial(np.radians(deg), D)
            cv2.ellipse(ov, (int(round(K[0, 2])), int(round(K[1, 2]))),
                        (int(round(r)), int(round(r * K[1, 1] / K[0, 0]))),
                        0, 0, 360, (0, 200, 255), 2)
            cv2.putText(ov, f"{deg}", (int(K[0, 2] + r) - 40, int(K[1, 2]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    r = rep["r_valid_px"]
    cv2.ellipse(ov, (int(round(K[0, 2])), int(round(K[1, 2]))),
                (int(round(r)), int(round(r * K[1, 1] / K[0, 0]))),
                0, 0, 360, (0, 0, 255), 4)
    cv2.putText(ov, f"valid to {rep['theta_max_deg']:.0f} deg  "
                    f"({rep['valid_frac'] * 100:.0f}% of frame)",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.imwrite(stem + "_cone.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Straight lines are the fastest visual check a human can do. Anything
    # architectural in frame should come out straight here. Over a chosen
    # sub-field, because a rectilinear view of the whole 180deg does not exist.
    out = stem + f"_rect{int(args.undistort_fov)}.png"
    cv2.imwrite(out, rectilinear_view(img, K, D, args.undistort_fov))
    print(f"\n[images] {os.path.basename(stem)}_cone.jpg      "
          f"boundary + iso-angle rings on the scene")
    print(f"         {os.path.basename(stem)}_validmask.png  binary mask")
    print(f"         {os.path.basename(out)}       rectilinear over "
          f"{args.undistort_fov:.0f} deg")
    print("   Straight edges in the scene must come out straight in the "
          "rectilinear view, especially ones that ran out near the boundary. "
          "Curvature there = bad outer fit.")
    return 0


def adapt(args) -> int:
    with open(args.adapt, encoding="utf-8") as f:
        m = json.load(f)
    crop_size = ([int(v) for v in args.crop_size.split("x")]
                 if args.crop_size else None)
    scale_to = ([int(v) for v in args.scale_to.split("x")]
                if args.scale_to else None)
    try:
        out = adapt_model(m, crop_left=args.crop_left, crop_top=args.crop_top,
                          crop_size=crop_size, scale_to=scale_to)
    except ValueError as e:
        print(f"[fatal] {e}")
        return 2
    K = np.array(out["K"])
    rep = field_angle_report(K, np.array(out["D"]), out["image_size"],
                             np.radians(out["theta_max_deg"]))
    print(f"[adapt] {m['image_size']} -> {out['image_size']}")
    print(f"        fx={K[0,0]:.3f} fy={K[1,1]:.3f} "
          f"cx={K[0,2]:.3f} cy={K[1,2]:.3f}   (D and the cone are unchanged)")
    print(f"        valid to {rep['theta_max_deg']:.1f} deg = "
          f"{rep['r_valid_px']:.1f} px radius, "
          f"{rep['valid_frac'] * 100:.1f}% of the new frame")
    print(f"        frame edges now at {rep['edge_x_deg']:.1f} deg across, "
          f"{rep['edge_y_deg']:.1f} deg down, corners {rep['corner_deg']:.1f} deg")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("aruco", "charuco"), default="aruco",
                    help="aruco = markers on white, ~half the ink and the better "
                         "choice for wide FOV. charuco only if you already have "
                         "one printed.")
    ap.add_argument("--make_board", metavar="PNG",
                    help="write a printable board sized for A4 and exit")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--images", help="folder of calibration stills")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--captured-no-rotate", action="store_true",
                    help="the stills were captured with calib_server.py's "
                         "--no-rotate, i.e. NOT rotated 180. Recorded into the "
                         "model as rotated_180 so sender.py can refuse to pair a "
                         "calibration with a run in the opposite orientation. "
                         "Getting this wrong is silent: the fit stays good and "
                         "every lidar return is simply coloured from the pixel "
                         "180 deg opposite the right one.")
    ap.add_argument("--nx", type=int, default=4, help="markers across")
    ap.add_argument("--ny", type=int, default=5, help="markers down")
    ap.add_argument("--marker", type=float, default=0.035,
                    help="marker side in METRES. Measure the print with calipers; "
                         "printers rescale, and a 3%% error here is a 3%% scale "
                         "error in every triangulation downstream.")
    ap.add_argument("--separation", type=float, default=0.0125,
                    help="white gap between markers in metres. Raising this and "
                         "shrinking --marker cuts ink further at the cost of "
                         "fewer corners per view.")
    ap.add_argument("--fov", type=float, default=0.0,
                    help="measured horizontal FOV in deg, used to seed f. Worth "
                         "supplying for lenses this extreme.")
    ap.add_argument("--min_corners", type=int, default=12,
                    help="reject views with fewer than this many corners")
    ap.add_argument("--full_board_only", action="store_true",
                    help="keep only views where every corner was found. Costs you "
                         "rim coverage; use only if calibrate() throws a shape "
                         "assertion on your OpenCV build.")
    ap.add_argument("--no_reject", action="store_true",
                    help="skip residual-based outlier rejection. Only to see what "
                         "the raw data does; a false positive that decodes as a "
                         "board id will otherwise dominate the solve.")
    ap.add_argument("--reject_rounds", type=int, default=5,
                    help="max reject-and-refit rounds. Stops early once a round "
                         "drops nothing.")
    ap.add_argument("--reject_k", type=float, default=6.0,
                    help="cut correspondences above this multiple of the median "
                         "residual (floor 3 px). Lower is more aggressive.")
    ap.add_argument("--theta_max", type=float, default=88.0,
                    help="trim the valid cone to this field angle in deg. Cannot "
                         "exceed 90: that is where cv2.fisheye's atan(r) folds "
                         "back, not a tuning choice. Default 88 leaves a margin, "
                         "since the projection stiffens as Z approaches 0.")
    ap.add_argument("--check", metavar="JSON", help="inspect a saved model")
    ap.add_argument("--image", help="frame to use with --check")
    ap.add_argument("--undistort_fov", type=float, default=90.0,
                    help="field of view in deg of the rectilinear check image. A "
                         "rectilinear view of the full 180 does not exist, so this "
                         "is a sub-field; raise it to look further out.")
    ap.add_argument("--adapt", metavar="JSON",
                    help="derive a model for a cropped/scaled frame from a "
                         "full-resolution fit. Exact, not a refit.")
    ap.add_argument("--crop_left", type=int, default=0)
    ap.add_argument("--crop_top", type=int, default=0,
                    help="for sender.py's centred band: (1944 - 1296) // 2 = 324")
    ap.add_argument("--crop_size", metavar="WxH", help="e.g. 2592x1296")
    ap.add_argument("--scale_to", metavar="WxH", help="e.g. 1280x640")
    args = ap.parse_args()

    if args.theta_max > THETA_MODEL_MAX_DEG:
        print(f"[fatal] --theta_max {args.theta_max} exceeds the {THETA_MODEL_MAX_DEG}"
              f" deg fold-back wall. Nothing cv2.fisheye fits carries information "
              f"out there; see the module docstring.")
        return 2

    if args.make_board:
        d = make_dictionary()
        b = make_board(args, d)
        # Size the raster from the physical footprint so a 100%-scale print comes
        # out dimensionally correct without any further arithmetic.
        #
        # The two targets have DIFFERENT footprints and must not share a formula.
        # A ChArUco board is a solid checkerboard: nx by ny squares of --marker with
        # no gaps, and the aruco marker sits inside the white square at 0.75 of it
        # (see make_board). Feeding it the aruco-grid footprint makes the canvas
        # aspect disagree with the board's, and generateImage scales the board to
        # whatever canvas it is given -- so the print comes out STRETCHED, which is
        # unrecoverable: a non-uniform scale is not a gauge freedom, it corrupts the
        # fx/fy ratio and the distortion tail with it.
        if args.target == "charuco":
            mm_x = args.nx * args.marker * 1000
            mm_y = args.ny * args.marker * 1000
        else:
            mm_x = (args.nx * args.marker + (args.nx - 1) * args.separation) * 1000
            mm_y = (args.ny * args.marker + (args.ny - 1) * args.separation) * 1000
        # Check both orientations: a wide grid on A4 landscape is perfectly normal
        # and testing portrait only cries wolf on every one of them. A3 too, since
        # bigger markers are the usual fix for a lens that has to be shot up close.
        fits = [name for name, pw, ph in (("A4", 190, 277), ("A3", 277, 400))
                if (mm_x <= pw and mm_y <= ph) or (mm_x <= ph and mm_y <= pw)]
        if fits:
            print(f"[fit] footprint {mm_x:.0f}x{mm_y:.0f} mm fits {'/'.join(fits)}"
                  f"{' (landscape)' if mm_x > 190 else ''}")
        else:
            print(f"[warn] footprint {mm_x:.0f}x{mm_y:.0f} mm does not fit A4 or A3 "
                  f"in either orientation; shrink --marker or --separation")
        px = args.dpi / 25.4
        margin = int(5 * px)
        cv2.imwrite(args.make_board,
                    board_image(b, int(mm_x * px) + 2 * margin,
                                int(mm_y * px) + 2 * margin, margin))
        print(f"[done] {args.make_board}  {args.nx}x{args.ny} markers "
              f"at {args.dpi} dpi")
        print("\n       PRINT: 100% scale, NO 'fit to page'. Greyscale/draft is "
              "fine -- detection needs contrast, not ink density. Mount FLAT.")
        print("       SCREEN: display at 100% zoom with OS scaling off. "
              "Backlight to 100% (PWM dimming bands a rolling shutter).")
        print("\n       Then MEASURE, because printers and screen scaling both "
              "lie. Do not measure a single marker -- measure the longest span "
              "and divide, so the caliper error is spread over a long baseline:")
        print(f"         horizontal, outer black edge of leftmost marker to "
              f"outer black edge of rightmost, same row:  {mm_x:.2f} mm nominal")
        print(f"         vertical, same but top to bottom of a column:        "
              f"  {mm_y:.2f} mm nominal")
        print(f"\n       scale = measured / nominal, then pass "
              f"--marker {args.marker:.4f}*scale")
        print(f"       Compute scale from BOTH spans. If they disagree by more "
              f"than ~0.3%% the display or printer is stretching one axis and "
              f"the geometry is unusable until you fix it.")
        return 0

    if args.adapt:
        return adapt(args)

    if args.check:
        if not args.image:
            print("[fatal] --check needs --image")
            return 2
        return check(args)

    if not args.images:
        ap.print_help()
        return 2
    return calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
