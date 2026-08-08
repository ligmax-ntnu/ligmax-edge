#!/usr/bin/env python3
"""Colour a lidar sweep from the cameras, and give detections a true range.

    rig = Rig.load("rig.json")
    cloud = fuse(sweep, rig, views)     # views: one View per camera, dets included

    lidar sweep (angle, range)
        |  Rig: lidar pose -> RIG frame, the one common frame on the wire
    p_rig  ----------------------------------------> shipped as x/y/z
        |  Rig: rig -> camera frame, per camera
    p_cam
        |  estimate.Camera.project  (forward Kannala-Brandt)
    full-res pixel -> detector pixel
        |                        |
    sample RGB              inside a detection box?
        |                        |
    per-point colour        per-detection RANGE from the lidar

Why this is worth doing
-----------------------
Range from apparent size degrades as z**2: a 40 cm buoy reads +-5 % at 10 m and
+-22 % at 100 m (README). The lidar reads +-3 cm flat. Anywhere the two overlap
-- which is everything inside the C1's 12 m -- the lidar range is better by an
order of magnitude, and it is the near field where being wrong actually hurts.
So `fuse` writes a `lidar` block onto any detection whose box the returns fall
inside, and leaves the size-based estimate alongside it rather than overwriting:
they are independent measurements and a consumer that can see both can tell when
they disagree.

The colour goes the other way. A 2D scanner returns a range and nothing else, so
a wall, a buoy and a person are the same measurement; the camera is what says
which. Colour is sampled off the detector frame -- the same pixels the boxes were
drawn from -- and then run through the OV5647 correction matrix HERE (`_correct`),
so what goes on the wire is comparable to the corrected preview a viewer shows.
The README puts that pass at the receiver, and for a 819k-pixel frame it belongs
there; a 400-point sweep is 2000x smaller and the argument does not carry.

Time, which is the part that is easy to get wrong
-------------------------------------------------
A sweep is not an instant. It takes ~100 ms -- longer than the camera's 71.4 ms
frame -- so a return at the back of a rotation is more than a whole frame older
than one at the front. And the camera frame the sweep is coloured against was
captured ~250 ms before it reaches this code. Both are handled explicitly: `lidar.LidarReader.sweep_near` selects by
the frame's `t_capture` out of a buffer, and the achieved skew rides on the wire
as `skew_ms`. Past `max_skew` the points still ship -- a lidar return is valid
whatever the cameras were doing -- but they ship UNCOLOURED, because a colour
sampled from a frame that does not line up is worse than no colour at all.
"""
from __future__ import annotations

import json
import math

import numpy as np

# RPLidar C1 datasheet distance accuracy. A floor under any per-target sigma:
# averaging more returns off one buoy cannot beat the sensor's own calibration.
RANGE_ACCURACY_M = 0.03

# How much deeper than its nearest return a detection's returns may be and still
# count as the same object. A Njord mark is 0.40 m across, so its own visible
# depth is under that; the rest of the slack absorbs a buoy seen at an angle and
# a little projection error. Raise it for larger targets -- but not far, or the
# sea behind the buoy starts counting as the buoy.
FOREGROUND_GATE_M = 0.5

# Colour is averaged over a small window rather than taken from one pixel, so a
# single hot pixel or a hair of projection error does not decide the answer.
#
# The window is biased DOWNWARD rather than centred, and that is deliberate. The
# lens sits ~5 cm above the scan plane, so a return projects slightly high on
# whatever it hit and the object's own body is the pixels just below it. Reading
# a few rows down is both more representative and more forgiving of the residual
# error in a tape-measured mounting height -- the geometry is still applied in
# full, this just stops the last centimetre of it from mattering.
#
# Kept small in both axes because the window straddles the silhouette edge on a
# distant buoy, where a wide one would average the target with the sea behind it.
PATCH_X = 2            # +-2 px horizontally -> 5 wide
PATCH_DY = (0, 3)      # 0..+3 px DOWN from the projected pixel -> 4 tall

# JetPack ships no ISP colour tuning for the OV5647, so Argus hands back close to
# sensor-native RGB: every Bayer sensor has heavy spectral crosstalk between its
# filters, and measured against SMPTE 75% bars these primaries come out at
# 0.30-0.41 saturation instead of 1.00 -- which is why reds arrive brown. This is
# the matrix that undoes it, fitted in ../camera-test (see ov5647_color.py there).
# Rows sum to ~1, so neutrals stay neutral and brightness is preserved.
#
# receiver.py imports this rather than keeping its own copy: one sensor, one
# matrix, and a frame and a point sampled from that frame must not be corrected
# differently or the plot disagrees with the picture it was sampled from.
OV5647_CCM = np.array(((1.714, -0.538, -0.177),
                       (-0.097, 1.646, -0.549),
                       (-0.113, -0.911, 2.024)), dtype=np.float32)


def _rot(yaw_deg, pitch_deg, roll_deg):
    """Yaw-pitch-roll -> rotation matrix, in the rig frame (+x right, +y down, +z fwd).

    Applied yaw @ pitch @ roll, so roll happens first in the body and yaw last.
    Right-handed about each axis, which given +y points DOWN means a positive yaw
    swings to starboard and a positive pitch lifts the nose -- see rig.json.
    """
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


class Pose:
    """Rigid pose of a body in the rig frame: p_rig = R @ p_body + t."""

    __slots__ = ("R", "t", "yaw", "pitch", "roll")

    def __init__(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, xyz_m=(0.0, 0.0, 0.0)):
        self.yaw, self.pitch, self.roll = float(yaw_deg), float(pitch_deg), float(roll_deg)
        self.R = _rot(yaw_deg, pitch_deg, roll_deg)
        self.t = np.asarray(xyz_m, dtype=np.float64).reshape(3)

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("yaw_deg", 0.0), d.get("pitch_deg", 0.0),
                   d.get("roll_deg", 0.0), d.get("xyz_m", (0.0, 0.0, 0.0)))

    def to_rig(self, p_body):
        return np.asarray(p_body, dtype=np.float64).reshape(-1, 3) @ self.R.T + self.t

    def from_rig(self, p_rig):
        return (np.asarray(p_rig, dtype=np.float64).reshape(-1, 3) - self.t) @ self.R

    def describe(self):
        return (f"yaw {self.yaw:+.2f} pitch {self.pitch:+.2f} roll {self.roll:+.2f} deg, "
                f"at ({self.t[0]:+.3f}, {self.t[1]:+.3f}, {self.t[2]:+.3f}) m")


class Rig:
    """Lidar and camera poses in one frame. Hand-measured; see rig.json."""

    def __init__(self, spec):
        ld = spec.get("lidar", {})
        self.lidar = Pose.from_dict(ld)
        self.angle_dir = 1.0 if int(ld.get("angle_dir", 1)) >= 0 else -1.0
        self.angle_zero = float(ld.get("angle_zero_deg", 0.0))
        self.cams = [Pose.from_dict(spec.get(f"cam{i}", {})) for i in (0, 1)]
        self.spec = spec

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def describe(self):
        lines = [f"  lidar: {self.lidar.describe()}, "
                 f"angle_dir {self.angle_dir:+.0f} zero {self.angle_zero:+.1f} deg"]
        for i, c in enumerate(self.cams):
            lines.append(f"  cam{i}:  {c.describe()}")
        return "\n".join(lines)

    def sweep_to_rig(self, sweep):
        """Sweep -> (N,3) points in the rig frame.

        The C1 is planar, so every point has y = 0 in its own frame; the lidar
        pose is what can tilt that plane in the rig, and the default pose is
        identity because the mounting was measured from the lidar's centre.
        """
        a = np.radians(self.angle_dir * (sweep.angle_deg - self.angle_zero))
        d = sweep.dist_m
        p = np.stack([d * np.sin(a), np.zeros_like(d), d * np.cos(a)], axis=1)
        return self.lidar.to_rig(p)


class View:
    """One camera's contribution: geometry, the frame to sample, and its boxes.

    `rgb` is the detector-input frame (net_w x net_h) -- the same pixels the boxes
    are expressed in, already in system memory, so sampling it costs nothing extra
    and no coordinate mapping can drift between colour and box.
    """

    __slots__ = ("index", "cam", "rgb", "dets")

    def __init__(self, index, cam, rgb, dets):
        self.index = index
        self.cam = cam
        self.rgb = rgb
        self.dets = dets


def _sample(rgb, u, v):
    """Mean RGB over the PATCH_X x PATCH_DY window below each (u, v).

    One fancy-index over broadcast offsets rather than a Python loop of gathers:
    the window is 20 pixels, and 20 separate numpy calls per camera per frame is
    dispatch overhead this frame budget cannot spare. Edges are clamped, so a
    return near the border reads a lopsided window rather than falling off it.
    """
    h, w = rgb.shape[:2]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    du = np.arange(-PATCH_X, PATCH_X + 1)
    dv = np.arange(PATCH_DY[0], PATCH_DY[1] + 1)
    uu = np.clip(ui[:, None, None] + du[None, None, :], 0, w - 1)
    vv = np.clip(vi[:, None, None] + dv[None, :, None], 0, h - 1)
    return np.rint(rgb[vv, uu].mean(axis=(1, 2))).astype(np.uint8)


# Saturation gain the full matrix delivers, measured over 1.6M pixels of real
# frames off this rig: mean chroma 30.1 raw -> 68.4 corrected. Used to work out
# how much matrix is left to apply once the ISP has already done some of the job.
CCM_CHROMA_GAIN = 2.27

# What the two together should add up to. Deliberately ABOVE CCM_CHROMA_GAIN --
# a little hot is the requested error direction, and undersaturated is the
# failure that actually costs you, since a detector trained on normal cameras
# reads a washed-out mark as the wrong class.
TARGET_CHROMA_GAIN = 2.7


def ccm_strength_for(saturation):
    """How much of the matrix to apply, given what the ISP already did.

    `nvarguscamerasrc saturation` scales chroma inside the ISP and the matrix
    scales it again, so applying both at full strength is a double correction:
    measured at saturation 2.0 it clips 53% of the frame, and clipped chroma is
    information destroyed before the detector or the plot can use it.

    Returns 1.0 at saturation 1.0, so a rig that leaves the ISP alone keeps
    exactly the behaviour it had. Clamped to [0, 1]: the matrix is a correction,
    not a gain stage, and running it past full strength distorts hue rather than
    adding saturation.
    """
    s = max(0.05, float(saturation))
    k = (TARGET_CHROMA_GAIN / s - 1.0) / (CCM_CHROMA_GAIN - 1.0)
    return float(min(1.0, max(0.0, k)))


def _correct(rgb, strength=1.0):
    """Sensor-native RGB -> colour-corrected RGB, for an (N,3) uint8 array.

    The README puts this pass at the receiver because a 1280x640 FRAME is
    819k pixels and the Jetson has no budget for it. A sweep is ~400 points --
    2000x less, measured at 0.06 ms -- so the argument does not carry here, and
    doing it at the source is what makes the number on the wire mean one thing:
    every consumer (the Pi, the dashboard, receiver.py) got a raw value it had
    to guess at otherwise, and the dashboard was left boosting saturation by eye
    to compensate.

    The matrix is defined in LINEAR light, so the values are linearised first
    and re-encoded after -- applied straight to gamma-encoded numbers it shifts
    hues instead of just restoring saturation. Uncoloured points are (0,0,0) and
    stay there: the rows sum to ~1, so black maps to black.
    """
    if strength <= 0.0:
        return rgb          # identity still costs a lossy round-trip
    m = OV5647_CCM
    if strength < 1.0:
        # Blend back toward identity. The off-diagonal terms are large because
        # the crosstalk is, so a partial matrix is also less chroma noise.
        m = np.eye(3, dtype=np.float32) * (1.0 - strength) + m * strength
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    out = np.clip(lin @ m.T, 0.0, 1.0)
    srgb = np.where(out <= 0.0031308, out * 12.92,
                    1.055 * np.power(out, 1.0 / 2.4) - 0.055)
    return np.rint(srgb * 255.0).astype(np.uint8)


def fuse(sweep, rig, views, max_skew=0.06, t_caps=None,
         foreground_gate=FOREGROUND_GATE_M, build_cloud=True, ccm_strength=1.0):
    """Colour a sweep and hang a lidar range off every detection it lands in.

    `views` may hold None for a camera that has no calibration or no frame this
    round; those simply do not contribute colour. `t_caps` is each camera's
    frame capture time, against which every point is timed individually.

    The time gate is PER POINT, not per sweep, and that distinction is the whole
    reason this reads the way it does. A rotation lasts ~100 ms, so its midpoint
    is up to 50 ms from any given frame no matter how well the two are running --
    gate on the sweep as a whole and colour switches off permanently while
    nothing is actually wrong. What decides whether a return can be coloured is
    when THAT RETURN was measured, which `Sweep.times` already knows.

    Mutates each detection dict in `views[i].dets`, adding a `lidar` block to the
    ones that got returns. Returns the columnar point cloud that goes on the wire.
    """
    p_rig = rig.sweep_to_rig(sweep)
    n = p_rig.shape[0]
    t_pt = sweep.times()

    cam_of = np.full(n, -1, dtype=np.int8)
    rgb_of = np.zeros((n, 3), dtype=np.uint8)
    det_of = np.full(n, -1, dtype=np.int32)
    # Field angle of the winning camera, so the second camera only takes a point
    # off the first if it sees it further from the rim, where the fisheye is
    # better behaved and the calibration is better constrained.
    best_field = np.full(n, np.inf)
    uv_of = np.zeros((n, 2), dtype=np.float64)
    # Timely for at least one camera. Reported separately from `coloured` so the
    # two reasons a return has no colour stay distinguishable: outside every
    # lens (geometry, expected -- most of a rotation is) versus measured at the
    # wrong moment (timing, worth investigating).
    timely_any = np.zeros(n, dtype=bool)

    for view in views:
        if view is None or view.cam is None or view.rgb is None:
            continue
        # Timeliness, per point: a return measured while this frame was being
        # exposed can be coloured from it; one measured most of a rotation later
        # cannot, and gets no colour rather than a wrong one.
        if t_caps is None or t_caps[view.index] is None:
            timely = np.ones(n, dtype=bool)
        else:
            timely = np.abs(t_pt - t_caps[view.index]) <= max_skew
            timely_any |= timely
            if not timely.any():
                continue
        p_cam = rig.cams[view.index].from_rig(p_rig)
        # Narrow to what can possibly land on this lens BEFORE projecting. A
        # camera sees one hemisphere and the scanner sweeps the whole circle, so
        # about half of every rotation is behind each lens -- and the projection
        # is the expensive part of this function (an arctan2, an 8th-order
        # polynomial and two more trig calls per point), against a frame budget
        # with ~10 ms of slack. z > 0 is exactly the "in front of the lens" test
        # and costs one comparison.
        cand = np.flatnonzero(timely & (p_cam[:, 2] > 0.0))
        if cand.size == 0:
            continue
        pc = p_cam[cand]
        uv_full, in_cone = view.cam.project(pc)
        uv_net = view.cam.to_net(uv_full)
        h, w = view.rgb.shape[:2]
        ok = (in_cone
              & (uv_net[:, 0] >= 0) & (uv_net[:, 0] <= w - 1)
              & (uv_net[:, 1] >= 0) & (uv_net[:, 1] <= h - 1))
        if not ok.any():
            continue
        sel, pc, uv_net = cand[ok], pc[ok], uv_net[ok]
        # Off-axis angle, as the tie-break between two cameras that both see it:
        # the one looking at it more squarely wins, where the fisheye is better
        # behaved and the calibration better constrained.
        field = np.arccos(np.clip(pc[:, 2] / np.linalg.norm(pc, axis=1), -1.0, 1.0))
        better = field < best_field[sel]
        if not better.any():
            continue
        take = sel[better]
        best_field[take] = field[better]
        cam_of[take] = view.index
        uv_of[take] = uv_net[better]
        rgb_of[take] = _sample(view.rgb, uv_net[better, 0], uv_net[better, 1])

    # ---- detections: which returns fall inside which box
    for view in views:
        if view is None or not view.dets:
            continue
        mine = cam_of == view.index
        if not mine.any():
            continue
        u, v = uv_of[:, 0], uv_of[:, 1]
        for j, det in enumerate(view.dets):
            box = det.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            hit = mine & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            k = int(hit.sum())
            if k == 0:
                continue
            r = np.linalg.norm(p_rig[hit], axis=1)
            # A box is never all target. It is a rectangle around a round buoy, so
            # its corners are open water, and at range the sea behind shows through
            # either side -- returns inside one box are routinely bimodal. Taking
            # the median of that lands BETWEEN the buoy and the background, which
            # is a range where nothing exists. So keep only the foreground cluster:
            # everything within a buoy's depth of the nearest return, which is the
            # object the box was drawn around.
            r_near = float(r.min())
            fg = r <= r_near + foreground_gate
            rf = r[fg]
            kf = int(fg.sum())
            # Sigma of the median of the foreground, floored at the sensor's own
            # accuracy -- more returns off one buoy cannot beat the C1's
            # calibration, only its noise.
            sem = (1.253 * float(np.std(rf)) / math.sqrt(kf)) if kf > 1 else RANGE_ACCURACY_M
            hit_fg = np.zeros_like(hit)
            hit_fg[np.flatnonzero(hit)[fg]] = True
            det["lidar"] = {
                "n": k,                 # returns inside the box
                "n_used": kf,           # of those, the foreground cluster
                "range_m": round(float(np.median(rf)), 3),
                "sigma_m": round(math.hypot(sem, RANGE_ACCURACY_M), 3),
                "nearest_m": round(r_near, 3),
                "spread_m": round(float(rf.max() - rf.min()), 3),
                # n_used well below n means the box straddled a depth edge. Worth
                # seeing rather than smoothing away: it is also how a box drawn
                # around the wrong thing shows up.
                "mixed": bool(kf < k),
                "bearing_deg": round(math.degrees(math.atan2(
                    float(np.median(p_rig[hit_fg, 0])),
                    float(np.median(p_rig[hit_fg, 2])))), 3),
                "cam": view.index,
            }
            # Only the foreground points belong to the buoy; the ones that came
            # back from the sea behind it are not part of this detection.
            det_of[hit_fg] = det.get("id") if det.get("id") is not None else -1

    coloured = int((cam_of >= 0).sum())
    if not build_cloud:
        # The detections above already have their ranges, which is the half of
        # this that every frame needs. The point cloud goes on the wire once per
        # rotation, so on a frame that is reusing a sweep the Pi already has,
        # skip converting eight 400-long arrays to JSON-able lists for nobody.
        return None
    return {
        "seq": int(sweep.seq),
        "frame": "rig",
        "n": int(n),
        "coloured": coloured,
        "in_time": int(timely_any.sum()) if t_caps is not None else int(n),
        "t_start": round(float(sweep.t_start), 6),
        "t_end": round(float(sweep.t_end), 6),
        "hz": round(1.0 / sweep.period, 2) if sweep.period > 0 else None,
        # Reported against the sweep MIDPOINT, so it is a single readable number
        # per camera. It is normally tens of ms and that is fine -- see the note
        # on per-point gating above before treating a big value as a fault.
        "skew_ms": (None if t_caps is None else
                    [None if t is None else round(1000.0 * (sweep.t_mid - t), 1)
                     for t in t_caps]),
        # Columnar, not a list of objects: ~400 points a sweep at 10 Hz, and the
        # per-object key repetition is most of the bytes if you let it be.
        #
        # Rounded with np.round and converted with .tolist(), NOT a comprehension
        # of round(float(v)). The comprehension is the same arithmetic but pays
        # Python call overhead per element, and at 8 arrays x ~400 points that
        # measured 4 ms of a frame budget with about 10 ms of slack in it. The
        # vectorised form is ~0.3 ms and produces identical numbers.
        "x": np.round(p_rig[:, 0], 3).tolist(),
        "y": np.round(p_rig[:, 1], 3).tolist(),
        "z": np.round(p_rig[:, 2], 3).tolist(),
        "dt_ms": np.round(1000.0 * (t_pt - sweep.t_start), 1).tolist(),
        "q": sweep.quality.tolist(),
        "cam": cam_of.tolist(),
        # Colour-corrected here rather than left sensor-native: see `_correct`.
        # `ccm_strength` is what the ISP's own saturation has NOT already done --
        # `ccm_strength_for`, which the caller works out from its saturation.
        "rgb": _correct(rgb_of, ccm_strength).reshape(-1).tolist(),
        "det": det_of.tolist(),
    }
