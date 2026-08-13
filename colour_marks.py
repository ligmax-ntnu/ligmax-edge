#!/usr/bin/env python3
"""Red and green marks from COLOUR alone, below the lidar's horizon.

    finder = ColourMarks(cams[0], rig.cams[0] if rig else None, net_w, net_h)
    boxes  = finder.find(rgbs[0])       # [(x1, y1, x2, y2, conf, cls), ...]

The boxes come out in **detector-input pixels**, which is what `estimate.estimate`
takes, so `sender.py` hands them to the same geometry function a YOLO box goes
through and every field on the wire - bearing, ray, range, sigmas, timestamps - is
produced by identical code. The only new thing here is where the box came from.

Why a second buoy detector exists at all
----------------------------------------
Both lidars are dead (2026-08-11, 2026-08-12) and the surprise task is scored on
passing red and green marks on the correct side. On the vessel,
`self_driving/perception/world.absorb_detections` refuses to let a camera detection
create a mark - the buoy detector is weak and a phantom buoy is worse than a missing
one - so with no lidar posing the questions, `behaviours/buoys.py` runs with an empty
world model and a scored leg silently becomes blind GNSS transit.

Something has to put a mark on that chart. The choice was between trusting the YOLO,
which is trained on a few dozen photographs and is the weakest instrument on this
boat, and trusting the thing the vessel *already* trusts for exactly this question:
the hue and saturation windows in `self_driving/config.py`, which is how every
coloured lidar return has been called red or green all along. This module is those
windows, run on the frame instead of on 400 points.

That is a smaller thing to be wrong about. A hue window has no training set to be
unrepresentative of, it fails in ways an operator can see on `/surprise_task`'s mask,
and its thresholds can be moved in the day's light without rebuilding an engine.

Two marks it cannot make, and does not pretend to
-------------------------------------------------
**A cardinal.** Which cardinal a yellow mark is lives in its topmark's two black
cones, which is a shape question, not a colour one. Yellow is deliberately not
detected here rather than detected and shipped as "some cardinal": a mark whose side
is unknown is worse than no mark, because `buoys._cardinal` slows the boat and holds
its line for one it can see and cannot name.

**Anything above the horizon.** The whole reason a colour test on a frame is
credible is the cut described below.

The cut, which is the entire trick
----------------------------------
Signal red at Havet is not only on buoys. It is on the arena's own signage, on
somebody's jacket on the pontoon, on a hull across the water and in the low sun off
wet paint; RAL 3001 and a red roof are the same hue. Every one of those is **above
the horizon** from a boat sitting in the water, and every buoy is below it.

So the mask is only ever applied below the line where a ray's elevation crosses zero
*in the rig frame* - the plane the lidar used to sweep, which is why it reads as "the
lidar line" on the dashboard. On a fisheye at a +-75 deg yaw that line is a strong
curve and not a row, so it is computed per column from the camera model and the
mount, once, at startup (`_horizon_rows`) - a flat cut at the same average height
would eat open water on one side of the frame and admit shoreline on the other.

Two consequences worth stating, because both will be seen on the water:

  * **a mark further away than the horizon is invisible to this module.** That is
    correct rather than a limitation - past the horizon a 40 cm buoy is below the
    pixel noise anyway, and `MARK_MAX_RANGE_M` on the vessel throws it away.
  * **the cut moves with the boat's attitude and this module does not know it.** The
    rig's pitch is `rig.json`'s static figure, so a bow-up trim or a wake lifts the
    true horizon above the computed one and lets a slice of shoreline in. It is
    bounded by how much this hull pitches, and the answer if it bites is `--colour-cut`
    with a fixed conservative fraction, not a smarter horizon fed by an IMU this
    process cannot see.

Every threshold below is a **mirror** of the vessel's own, and that is a real cost:
two copies of four numbers, on two boards, kept in step by hand. The alternative was
shipping the frame to the Pi, which is 819k pixels down a link built for 400 points.
When they disagree, the vessel's copy is the one that counts - it is what decides
whether the boat steers - and the way to see the disagreement is `/surprise_task`'s
mask against the marks on the chart.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - the Jetson has it; a laptop checking syntax may not
    import cv2
    CV2 = True
except Exception:
    cv2 = None
    CV2 = False


# --- the vessel's windows, mirrored from self_driving/config.py ---------------
#
# Hue in degrees, saturation 0..1. Red wraps, so it is two windows. The saturation
# bars are deliberately ASYMMETRIC and that is not sloppiness: a warm cast lifts the
# red channel, which raises the maximum on a red mark (chroma and saturation grow)
# and the minimum on a green one (both shrink), so one threshold for both cannot be
# right. `classify.py` carries the full argument.
HUE_RED_LOW_MAX = 20.0
HUE_RED_HIGH_MIN = 335.0
HUE_GREEN_MIN = 62.0
HUE_GREEN_MAX = 200.0
MIN_SATURATION_RED = 0.55
MIN_SATURATION_GREEN = 0.22

# Below this there is no colour to speak of, only sensor noise given a hue by the
# division. Value, not saturation: a very dark pixel can have a high saturation and a
# meaningless hue, which is most of what a shadow on water is.
MIN_VALUE = 0.10

#: The detector's own class ids, so a colour box is indistinguishable from a YOLO box
#: to everything downstream except its `src`. `protocol.CLASS_NAMES`: 0 green, 1 red.
CLS_GREEN = 0
CLS_RED = 1

#: What this module stamps on every detection it makes. The vessel's
#: `config.MARK_SOURCES` names it, and `world._create_mark` reads it to decide
#: whether this box is allowed to create a track.
SOURCE = "colour"


def _hsv(rgb):
    """(hue degrees, saturation, value) from an HxWx3 uint8 RGB frame.

    Written out rather than `cv2.cvtColor` for one reason: OpenCV's 8-bit HSV packs
    hue into 0..179, which is half a degree of quantisation right where the red
    window's edge sits (20 deg becomes bin 10), and the two colours this has to
    separate are the two the quantisation hurts most. float32 costs ~4 ms on a
    480 px frame and removes the question.
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    safe = np.where(chroma <= 1e-6, 1.0, chroma)
    hue = np.where(
        mx == r, ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    ) * 60.0
    hue = np.where(chroma <= 1e-6, 0.0, hue)
    sat = np.where(mx <= 1e-6, 0.0, chroma / np.where(mx <= 1e-6, 1.0, mx))
    return hue, sat, mx


def _horizon_rows(cam, pose, net_w, net_h):
    """Per-column detector-input row where elevation crosses zero. See the docstring.

    Returns an int array of length `net_w`; a column whose ray never reaches level -
    the whole column is above the horizon, or outside the calibrated cone - gets
    `net_h`, i.e. nothing in that column is ever counted. Refusing the column is the
    conservative direction: the failure it avoids is admitting shoreline.
    """
    us, vs = np.meshgrid(
        np.arange(net_w, dtype=np.float64), np.arange(net_h, dtype=np.float64)
    )
    uv_full = cam.to_full(np.stack([us.ravel(), vs.ravel()], axis=1))
    rays = cam.rays(uv_full)
    if pose is not None:
        # Direction only - a ray has no position, so the mount's translation must not
        # be applied. `Pose.to_rig` would add it; this is that method's rotation half.
        rays = rays @ pose.R.T
    # +y is DOWN in both the camera frame and the rig frame, so y >= 0 is level or
    # below. NaN (outside the cone) compares False, which is the answer we want.
    down = (rays[:, 1] >= 0.0).reshape(net_h, net_w)
    rows = np.argmax(down, axis=0)
    rows[~down.any(axis=0)] = net_h
    return rows.astype(np.int32)


class ColourMarks:
    """The colour test for one camera. Build once, `find` per frame."""

    # A blob has to be at least this many detector pixels to be a mark rather than a
    # glint. Small on purpose: a 40 cm buoy at 20 m is only a few pixels across in a
    # 640 px letterboxed frame, and the thing this is really rejecting is the
    # single-pixel speckle any saturation threshold produces on moving water.
    MIN_AREA_PX = 6
    # ...and at most this fraction of the counted region. A blob bigger than this is
    # not a buoy: it is the hull, a pontoon, a wall, or the whole frame going red in
    # low sun. Rejecting it matters more than finding it, because a huge blob's
    # centroid is meaningless and its apparent width would put a mark 1 m away.
    MAX_AREA_FRAC = 0.04
    # Roughly round. A Njord mark is a sphere; a long horizontal streak is a
    # reflection on water and a long vertical one is a pole or a mast.
    MIN_ASPECT = 0.35
    MAX_ASPECT = 2.9
    # How much of its own bounding box the blob must fill. A circle fills pi/4 = 0.79
    # of its box, so 0.45 is loose enough for a part-occluded or clipped mark and
    # tight enough to reject the L-shapes and arcs that reflections make.
    MIN_FILL = 0.45

    def __init__(self, cam, pose, net_w, net_h, cut_frac=None):
        self.cam = cam
        self.net_w = int(net_w)
        self.net_h = int(net_h)
        self.reason = ""
        if not CV2:
            # Reported rather than raised: a board without cv2 should lose this
            # feature and keep capturing, previews, the tags and the YOLO. The
            # dashboard sees no colour marks and the vessel's mark source shows off.
            self.reason = "cv2 is missing, so the colour test cannot label blobs"
            self.rows = None
            return
        if cut_frac is not None:
            # The override, `--colour-cut`. A flat row, honestly flat: this exists
            # for a rig whose pitch is wrong or unknown, and for reproducing what
            # /surprise_task's slider shows.
            row = int(round(max(0.0, min(1.0, float(cut_frac))) * self.net_h))
            self.rows = np.full(self.net_w, row, dtype=np.int32)
            self.reason = f"flat cut at {row}/{self.net_h} rows (--colour-cut)"
            return
        try:
            self.rows = _horizon_rows(cam, pose, self.net_w, self.net_h)
            usable = int((self.rows < self.net_h).sum())
            self.reason = (
                f"horizon from the camera model and the mount: "
                f"{usable}/{self.net_w} columns see level water, "
                f"cut rows {int(self.rows.min())}..{int(self.rows[self.rows < self.net_h].max()) if usable else self.net_h}"
            )
        except Exception as exc:      # a bad rig.json must not stop the frame loop
            self.rows = None
            self.reason = f"could not compute the horizon: {exc}"

    # ------------------------------------------------------------------ per frame

    def find(self, rgb):
        """`[(x1, y1, x2, y2, conf, cls), ...]` in detector-input pixels.

        `conf` is the fraction of the bounding box that actually passed the colour
        window - the blob's fill. That is a real measurement and not a borrowed
        score: a round mark fills about 0.79 of its box and a smear fills a third of
        it, so the number means "how much does this look like a buoy" in the only
        terms this detector has. It is what `MARK_MIN_CONF_COLOUR` is compared
        against on the vessel.
        """
        if self.rows is None or rgb is None:
            return []
        height, width = rgb.shape[0], rgb.shape[1]
        if width != self.net_w or height != self.net_h:
            # The horizon table is per column of a known frame size. A resize between
            # startup and now means the table indexes the wrong columns, and a
            # silently misaligned cut is exactly the failure the cut exists to
            # prevent - so rebuild rather than stretch.
            self.net_w, self.net_h = width, height
            try:
                self.rows = _horizon_rows(self.cam, None, width, height)
            except Exception:
                return []

        hue, sat, val = _hsv(rgb)
        lit = val >= MIN_VALUE
        red = lit & (sat >= MIN_SATURATION_RED) & (
            (hue <= HUE_RED_LOW_MAX) | (hue >= HUE_RED_HIGH_MIN)
        )
        green = lit & (sat >= MIN_SATURATION_GREEN) & (
            (hue >= HUE_GREEN_MIN) & (hue <= HUE_GREEN_MAX)
        )

        # The cut. Built as a full mask rather than by slicing rows, because the
        # horizon is a curve: every column has its own first legal row.
        ramp = np.arange(height, dtype=np.int32)[:, None]
        below = ramp >= self.rows[None, :]
        red &= below
        green &= below

        counted = int(below.sum())
        if counted <= 0:
            return []
        max_area = max(self.MIN_AREA_PX, int(counted * self.MAX_AREA_FRAC))

        out = []
        for mask, cls in ((red, CLS_RED), (green, CLS_GREEN)):
            out.extend(self._blobs(mask, cls, max_area))
        # Most confident first, which is the order the YOLO's rows arrive in and the
        # order `--max-cardinals`-style budgets downstream assume.
        out.sort(key=lambda box: box[4], reverse=True)
        return out

    def _blobs(self, mask, cls, max_area):
        if not mask.any():
            return []
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        found = []
        for index in range(1, count):          # 0 is the background
            x, y, w, h, area = (int(v) for v in stats[index][:5])
            if area < self.MIN_AREA_PX or area > max_area:
                continue
            if w <= 0 or h <= 0:
                continue
            aspect = w / h
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue
            fill = area / float(w * h)
            if fill < self.MIN_FILL:
                continue
            # Half-open box, like the detector's: x + w is one past the last column.
            found.append((float(x), float(y), float(x + w), float(y + h),
                          round(float(fill), 4), cls))
        return found

    def describe(self):
        if self.rows is None:
            return f"colour marks: OFF - {self.reason}"
        return f"colour marks: {self.reason}"
