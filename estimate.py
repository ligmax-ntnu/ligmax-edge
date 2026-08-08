#!/usr/bin/env python3
"""Bearing and range for each detection, with honest uncertainties.

    detector box (1280x640)
        |  map back through crop+scale
    full-res box  --> subpixel edge refinement on the Y plane
        |
    pixels_to_rays (Kannala-Brandt)  -->  unit bearing vector
        |                                      |
    azimuth / elevation                 angular diameter
                                               |
                                      range = (D/2) / sin(alpha/2)

Two things this module exists to get right.

TIMESTAMPS. `sender.py` used to stamp frames with time.time() after inference, which
is the wrong instant by however long capture, conversion and the detector took. What
a triangulator needs is when the PHOTONS arrived. GStreamer already carries that as
the buffer PTS; `CaptureClock` converts it to wall clock. And because this is a
rolling shutter, there is no single capture instant for a frame -- the bottom of the
sensor is read tens of milliseconds after the top -- so a per-detection time derived
from its own image row is also provided, which matters on a boat that is rolling.

RANGE UNCERTAINTY. Range from apparent size degrades as z^2: sigma_z/z = sigma_alpha
/ alpha, and alpha itself shrinks as 1/z. A 40 cm buoy at 50 m is 7 px across on this
sensor, so one pixel of edge error is 15 % of the range. The functions here return
sigma alongside every estimate and a `valid` flag, so a caller can reject rather than
be quietly misled. Do not treat range as usable at long distance just because a
number came back -- check the sigma.

The bearing sigma is dominated by the CALIBRATION, not by the box: split-half
validation of the fisheye fit measured ~0.25 deg, while a box centre good to 2 full-res
pixels is 0.14 deg. Note that the calibration part is a fixed model error, so it is
CORRELATED between detections and between frames -- it does not average out over a
track, and it does not cancel between the two cameras. Only the centroid part is
independent noise.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np

# Njord competition buoys. Diameter, and how well we know it.
BUOY_DIAMETER_M = 0.40
BUOY_DIAMETER_SIGMA_M = 0.02

# Measured by split-half validation of the fisheye calibration; see calibrate/.
CALIB_BEARING_SIGMA_DEG = 0.25
# Box edge repeatability in full-resolution pixels when refinement declines to run.
BOX_EDGE_SIGMA_PX = 3.0


def _kb_radial(theta, D):
    k = np.asarray(D, dtype=np.float64).reshape(-1)
    t = np.asarray(theta, dtype=np.float64)
    t2 = t * t
    return t * (1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)


class Camera:
    """One fisheye camera model plus the crop/scale that produced the detector input.

    Holds BOTH geometries on purpose. Boxes arrive in detector pixels, but every
    measurement worth making happens at full sensor resolution -- the detector input
    is downscaled, so a box edge there is worth ~1.6 full-res pixels of quantisation
    before any other error. Keeping the mapping here means callers never hand-roll it
    and get the principal-point shift wrong.
    """

    def __init__(self, model: dict, crop_left=0, crop_top=0, crop_w=None, crop_h=None,
                 net_w=1280, net_h=640):
        self.K = np.array(model["K"], dtype=np.float64)
        self.D = np.array(model["D"], dtype=np.float64).reshape(-1)
        self.image_size = tuple(int(v) for v in model["image_size"])
        self.theta_max = math.radians(float(model.get("theta_max_deg", 88.0)))
        self.r_valid_norm = float(_kb_radial(self.theta_max, self.D))
        self.crop = (int(crop_left), int(crop_top),
                     int(crop_w or self.image_size[0]), int(crop_h or self.image_size[1]))
        self.net = (int(net_w), int(net_h))
        self.scale = self.crop[2] / float(net_w)      # full-res px per detector px

    @classmethod
    def load(cls, path, **kw):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), **kw)

    # ---------------------------------------------------------------- geometry
    def to_full(self, uv):
        """Detector-input pixels -> full sensor pixels."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return np.stack([uv[:, 0] * self.scale + self.crop[0],
                         uv[:, 1] * self.scale + self.crop[1]], axis=1)

    def rays(self, uv_full):
        """Full-res pixels -> unit rays in the camera frame. NaN outside the cone.

        Camera frame is OpenCV's: +x right, +y down, +z along the optical axis.
        """
        uv = np.asarray(uv_full, dtype=np.float64).reshape(-1, 2)
        x = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        r_d = np.hypot(x, y)
        phi = np.arctan2(y, x)
        ok = r_d <= self.r_valid_norm
        theta = np.where(ok, r_d, 0.0)
        k = self.D
        for _ in range(12):
            t2 = theta * theta
            poly = 1.0 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4
            dpoly = (2 * k[0] * theta + 4 * k[1] * theta**3
                     + 6 * k[2] * theta**5 + 8 * k[3] * theta**7)
            f = theta * poly - r_d
            df = poly + theta * dpoly
            theta = theta - f / np.where(np.abs(df) < 1e-12, 1e-12, df)
        v = np.stack([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)], axis=1)
        v[~ok] = np.nan
        return v

    def to_net(self, uv_full):
        """Full sensor pixels -> detector-input pixels. Inverse of `to_full`.

        The result is deliberately NOT clipped to the network frame: a point can
        project to a real pixel on the sensor that the detector's crop does not
        cover, and the caller needs to see that it fell outside rather than have
        it clamped onto the border.
        """
        uv = np.asarray(uv_full, dtype=np.float64).reshape(-1, 2)
        return np.stack([(uv[:, 0] - self.crop[0]) / self.scale,
                         (uv[:, 1] - self.crop[1]) / self.scale], axis=1)

    def project(self, p_cam):
        """Camera-frame 3D points -> (uv_full, valid). The inverse of `rays`.

        Forward Kannala-Brandt: theta off the optical axis, radius r = theta *
        poly(theta), placed at azimuth phi around the principal point. Points
        behind the camera need no special case -- theta comes out past 90 deg and
        so fails the cone test below, which is the honest answer rather than the
        fold-back cv2.fisheye would produce (see the README: 110 deg lands at the
        same radius as 70 deg, silently).

        `valid` is the calibrated cone (88 deg here), not the image rectangle.
        Whether a pixel is actually inside the crop is a separate question and
        belongs to the caller, which knows what it is sampling.
        """
        p = np.asarray(p_cam, dtype=np.float64).reshape(-1, 3)
        rxy = np.hypot(p[:, 0], p[:, 1])
        # atan2, not acos(z/|p|): correct through and past 90 deg, and it does
        # not lose precision for points near the optical axis.
        theta = np.arctan2(rxy, p[:, 2])
        phi = np.arctan2(p[:, 1], p[:, 0])
        r_d = _kb_radial(theta, self.D)
        uv = np.stack([self.K[0, 0] * r_d * np.cos(phi) + self.K[0, 2],
                       self.K[1, 1] * r_d * np.sin(phi) + self.K[1, 2]], axis=1)
        valid = (theta <= self.theta_max) & np.isfinite(uv).all(axis=1)
        return uv, valid

    def bearing(self, uv_full):
        """-> (azimuth_deg, elevation_deg, field_angle_deg), NaN outside the cone.

        Azimuth is positive to the right of the optical axis, elevation positive up
        (negated from +y, which points down). Both are in the CAMERA frame -- turning
        them into boat-relative angles needs the mount rotation, which this module
        deliberately does not guess at.
        """
        v = self.rays(uv_full)[0]
        if not np.isfinite(v[0]):
            return float("nan"), float("nan"), float("nan")
        return (math.degrees(math.atan2(v[0], v[2])),
                math.degrees(-math.asin(max(-1.0, min(1.0, v[1])))),
                math.degrees(math.acos(max(-1.0, min(1.0, v[2])))))

    def mrad_per_px(self, uv_full):
        """Local angular scale, which varies across a fisheye frame by ~2x.

        Measured rather than assumed: step one pixel and see how far the ray moves.
        A single global mrad/px would understate the centre and overstate the rim.
        """
        uv = np.asarray(uv_full, dtype=np.float64).reshape(2)
        a = self.rays([uv])[0]
        b = self.rays([uv + [1.0, 0.0]])[0]
        if not (np.isfinite(a[0]) and np.isfinite(b[0])):
            return float("nan")
        return 1000.0 * math.acos(max(-1.0, min(1.0, float(a @ b))))

    def angle_between(self, uv_a, uv_b):
        """Angular separation in radians between two full-res pixels; NaN if either
        is outside the cone."""
        v = self.rays(np.vstack([np.reshape(uv_a, 2), np.reshape(uv_b, 2)]))
        if not (np.isfinite(v[0, 0]) and np.isfinite(v[1, 0])):
            return float("nan")
        return math.acos(max(-1.0, min(1.0, float(v[0] @ v[1]))))


# -------------------------------------------------------------------- timestamps
class CaptureClock:
    """Converts a GStreamer buffer PTS into wall-clock capture time.

    A buffer's PTS is running time in the pipeline clock: wall = base_time + pts,
    expressed in the clock's own domain. GstSystemClock is CLOCK_MONOTONIC here, so a
    monotonic->epoch offset is also needed. That offset is sampled ONCE and reused:
    resampling per frame would inject NTP's slew into the frame-to-frame intervals,
    which is precisely the signal a parallax pipeline is trying to measure. The
    consequence is that absolute time may drift slowly against UTC while relative
    times stay exact -- the right trade for this use.
    """

    def __init__(self, base_time_ns: int, fps: float, readout_frac: float = 0.9):
        import time
        self.base_time_ns = int(base_time_ns)
        self.mono_to_epoch = time.time() - time.monotonic()
        # Rolling-shutter readout of the active rows. On the OV5647 at 2592x1944 the
        # active lines occupy nearly the whole frame period, so the bottom of the
        # frame is exposed ~60 ms after the top. Approximate, and worth overriding if
        # you ever measure it: it is a per-detection time offset, not a fudge factor.
        self.readout_s = readout_frac / float(fps)

    def frame_time(self, pts_ns: int) -> float:
        """Epoch seconds at the START of this frame's readout (top row)."""
        return (self.base_time_ns + int(pts_ns)) / 1e9 + self.mono_to_epoch

    def row_time(self, pts_ns: int, y_full: float, height: int) -> float:
        """Epoch seconds when the given sensor ROW was actually exposed."""
        return self.frame_time(pts_ns) + self.readout_s * (float(y_full) / height)


# ------------------------------------------------------------------- refinement
def refine_extent(gray_full, box_full, pad=0.35):
    """Subpixel horizontal extent of the buoy in a full-res grayscale frame.

    Returns (x_left, x_right, y_centre, quality) in full-res pixels, or None.

    Width rather than height, because these buoys float: the waterline cuts the
    bottom off by an unknown amount, so vertical extent is not the diameter, while the
    horizontal extent through the centre is. That single choice matters more than any
    refinement subtlety here.

    Deliberately simple -- Otsu inside a padded ROI, largest component, then a linear
    interpolation of the intensity profile across each edge. It reports `quality` from
    the edge contrast relative to local noise so the caller can widen sigma or fall
    back to the raw box, rather than pretending every refinement is equally good.
    """
    h, w = gray_full.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_full]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    px, py = bw * pad, bh * pad
    rx1, ry1 = int(max(0, math.floor(x1 - px))), int(max(0, math.floor(y1 - py)))
    rx2, ry2 = int(min(w, math.ceil(x2 + px))), int(min(h, math.ceil(y2 + py)))
    roi = gray_full[ry1:ry2, rx1:rx2]
    if roi.size < 64 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None

    blur = cv2.GaussianBlur(roi, (0, 0), 1.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The buoy may be darker or brighter than the water; keep whichever polarity
    # gives a component that actually looks like the detector's box.
    best = None
    for m in (mask, 255 - mask):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, n):
            sx, sy, sw, sh, area = stats[j]
            if area < 0.15 * bw * bh or sw < 3 or sh < 3:
                continue
            # overlap with the detector box, in ROI coordinates
            ox1, oy1 = max(sx, x1 - rx1), max(sy, y1 - ry1)
            ox2, oy2 = min(sx + sw, x2 - rx1), min(sy + sh, y2 - ry1)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            iou = ((ox2 - ox1) * (oy2 - oy1)) / (bw * bh + area - (ox2 - ox1) * (oy2 - oy1))
            if best is None or iou > best[0]:
                best = (iou, sx, sy, sw, sh, cent[j][1])
    if best is None or best[0] < 0.25:
        return None
    _, sx, sy, sw, sh, cy_roi = best

    # Subpixel edges from the intensity profile along the component's centre row.
    row = int(round(min(max(cy_roi, 1), roi.shape[0] - 2)))
    prof = cv2.GaussianBlur(roi[max(0, row - 1):row + 2, :].mean(0).astype(np.float32)
                            .reshape(1, -1), (0, 0), 1.0).ravel()
    inside = prof[sx:sx + sw]
    if inside.size < 3:
        return None
    lvl_in = float(np.median(inside))
    outs = np.concatenate([prof[:sx], prof[sx + sw:]])
    if outs.size < 3:
        return None
    lvl_out = float(np.median(outs))
    contrast = abs(lvl_in - lvl_out)
    noise = float(np.std(prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 2.0).ravel()))
    if contrast < 4.0:
        return None
    half = 0.5 * (lvl_in + lvl_out)

    def cross(i0, step):
        """Walk out from the component edge to where the profile crosses the
        half-amplitude level, then interpolate between the bracketing samples."""
        i = i0
        for _ in range(max(4, int(0.5 * sw))):
            j = i + step
            if j < 0 or j >= prof.size:
                return None
            a, b = prof[i], prof[j]
            if (a - half) * (b - half) <= 0 and a != b:
                return i + (half - a) / (b - a) * step
            i = j
        return None

    xl = cross(sx, -1)
    xr = cross(sx + sw - 1, +1)
    if xl is None or xr is None or xr - xl < 3:
        return None
    quality = contrast / max(noise, 1e-6)
    return (rx1 + xl, rx1 + xr, ry1 + cy_roi, quality)


# ----------------------------------------------------------------------- range
def range_from_angular_width(cam: Camera, x_left, x_right, y_centre,
                             sigma_edge_px, diameter_m=BUOY_DIAMETER_M,
                             diameter_sigma_m=BUOY_DIAMETER_SIGMA_M):
    """Range to a sphere from the angular width of its silhouette.

    For a sphere the tangent lines give sin(alpha/2) = (D/2)/z exactly, so
        z = (D/2) / sin(alpha/2)
    Using the small-angle z = D/alpha instead costs 2 % at 2 m; using atan (as for a
    flat target of width D) is simply the wrong geometry for a ball.

    -> dict with range_m, sigma_m, rel_sigma, alpha_mrad and a `valid` flag.
    """
    a = cam.angle_between([x_left, y_centre], [x_right, y_centre])
    out = {"alpha_mrad": None, "range_m": None, "sigma_m": None,
           "rel_sigma": None, "valid": False, "why": None}
    if not np.isfinite(a) or a <= 0:
        out["why"] = "outside the valid cone"
        return out
    out["alpha_mrad"] = round(1000.0 * a, 3)
    s = math.sin(a / 2.0)
    if s <= 1e-9 or s >= 1.0:
        out["why"] = "degenerate angular width"
        return out
    z = (diameter_m / 2.0) / s

    # sigma_alpha from the two edges, each independent, converted at the LOCAL
    # angular scale -- a fisheye's mrad/px varies about twofold across the frame.
    mpp = cam.mrad_per_px([0.5 * (x_left + x_right), y_centre])
    if not np.isfinite(mpp):
        out["why"] = "outside the valid cone"
        return out
    sigma_a = math.sqrt(2.0) * sigma_edge_px * mpp / 1000.0
    # dz/dalpha for z = (D/2)/sin(a/2)
    dz_da = -(diameter_m / 2.0) * math.cos(a / 2.0) / (2.0 * s * s)
    var = (dz_da * sigma_a) ** 2 + (z * diameter_sigma_m / diameter_m) ** 2
    sigma = math.sqrt(var)
    out.update(range_m=round(z, 3), sigma_m=round(sigma, 3),
               rel_sigma=round(sigma / z, 4), valid=True,
               alpha_sigma_mrad=round(1000.0 * sigma_a, 3))
    return out


def bearing_sigma_deg(cam: Camera, uv_full, sigma_centroid_px):
    """Bearing uncertainty: calibration (correlated) + centroid (independent).

    Returned separately as well as combined, because a triangulator must treat them
    differently -- the calibration term is the same error on every detection from this
    camera and will not average away over a track.
    """
    mpp = cam.mrad_per_px(uv_full)
    if not np.isfinite(mpp):
        return None
    ind = math.degrees(sigma_centroid_px * mpp / 1000.0)
    return {"sigma_deg": round(math.hypot(CALIB_BEARING_SIGMA_DEG, ind), 4),
            "sigma_calib_deg": CALIB_BEARING_SIGMA_DEG,
            "sigma_centroid_deg": round(ind, 4),
            "mrad_per_px": round(mpp, 4)}


def estimate(cam: Camera, box_net, gray_full=None, clock: CaptureClock = None,
             pts_ns=None, diameter_m=BUOY_DIAMETER_M):
    """Everything geometric for one detection. Box is in detector-input pixels.

    Keys are grouped so a consumer can ignore what it does not need, and every
    estimate carries its own sigma and validity rather than a bare number.
    """
    x1, y1, x2, y2 = [float(v) for v in box_net]
    full = cam.to_full([[x1, y1], [x2, y2]])
    fx1, fy1 = full[0]
    fx2, fy2 = full[1]
    out = {}

    # Truncation check FIRST: a buoy clipped by the crop edge has an underestimated
    # width, which inflates range without any other symptom. Better to refuse.
    m = 1.5      # detector pixels of margin; a box this close to the edge is clipped
    truncated = (x1 <= m or y1 <= m
                 or x2 >= cam.net[0] - m or y2 >= cam.net[1] - m)
    out["truncated"] = bool(truncated)

    # ---- refinement on the full-res Y plane
    sigma_edge = BOX_EDGE_SIGMA_PX
    method = "detector_box"
    xl, xr, yc = fx1, fx2, 0.5 * (fy1 + fy2)
    if gray_full is not None:
        r = refine_extent(gray_full, (fx1, fy1, fx2, fy2))
        if r is not None:
            xl, xr, yc, q = r
            method = "refined_edges"
            # Edge sigma from measured edge contrast-to-noise, floored so a
            # flattering ROI cannot claim implausible precision.
            sigma_edge = max(0.4, min(BOX_EDGE_SIGMA_PX, 2.0 / math.sqrt(max(q, 1e-6))))
    out["width_method"] = method
    out["edge_sigma_px"] = round(sigma_edge, 3)
    out["width_px_full"] = round(xr - xl, 2)

    # ---- bearing from the refined horizontal centre, box vertical centre
    ucx, ucy = 0.5 * (xl + xr), yc
    az, el, field = cam.bearing([ucx, ucy])
    out["bearing_deg"] = None if math.isnan(az) else round(az, 4)
    out["elevation_deg"] = None if math.isnan(el) else round(el, 4)
    out["field_angle_deg"] = None if math.isnan(field) else round(field, 3)
    out["in_valid_cone"] = bool(not math.isnan(az))
    bs = bearing_sigma_deg(cam, [ucx, ucy], max(0.5, 0.5 * sigma_edge))
    if bs:
        out.update(bs)
    v = cam.rays([[ucx, ucy]])[0]
    out["ray_cam"] = None if not np.isfinite(v[0]) else [round(float(t), 6) for t in v]

    # ---- range
    rng = range_from_angular_width(cam, xl, xr, yc, sigma_edge, diameter_m)
    if truncated and rng["valid"]:
        rng["valid"] = False
        rng["why"] = "box touches the crop edge; width is a lower bound"
    out["range"] = rng

    # ---- time this detection's own rows were exposed
    if clock is not None and pts_ns is not None:
        out["t_capture"] = round(clock.frame_time(pts_ns), 6)
        out["t_row"] = round(clock.row_time(pts_ns, ucy, cam.image_size[1]), 6)
    return out


def crop_for_aim(model: dict, aim_deg: float, crop_w: int, crop_h: int):
    """Pick a src-crop window centred on a given bearing, clamped to the sensor.

    aim_deg is where the crop CENTRE should point, in degrees right of the optical
    axis (negative = left). The pair here diverges with its overlap on cam0's right
    and cam1's left, so cam0 wants a positive aim and cam1 a negative one of the same
    size. Returns (crop_left, crop_top, actual_aim_deg) -- `actual` because a wide
    crop cannot be shifted far before it runs off the sensor, and silently clamping
    without saying so would misreport where the detector is looking.
    """
    K = np.array(model["K"], dtype=np.float64)
    D = np.array(model["D"], dtype=np.float64).reshape(-1)
    W, H = model["image_size"]
    r_norm = _kb_radial(math.radians(abs(aim_deg)), D)
    cx_target = K[0, 2] + math.copysign(r_norm * K[0, 0], aim_deg)
    left = int(round(cx_target - crop_w / 2.0))
    left = max(0, min(W - crop_w, left))
    top = max(0, min(H - crop_h, int(round((H - crop_h) / 2.0))))
    # Report the bearing the clamped crop actually centres on.
    cam = Camera(model, left, top, crop_w, crop_h)
    az, _, _ = cam.bearing([[left + crop_w / 2.0, top + crop_h / 2.0]])
    return left, top, (None if math.isnan(az) else round(az, 2))
