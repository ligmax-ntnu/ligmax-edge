#!/usr/bin/env python3
"""ArUco tags on a dock: found on the full-resolution fisheye frame, measured in
the rig frame.

    finder = TagFinder(cams, rig)                  # estimate.Camera x2, fusion.Rig
    tags = finder.find(0, gray_full)               # full-res Y plane, cam0
    tags += finder.find(1, gray_full_cam1)

NJORD §9.3 marks the assigned berth with **three 18 x 18 cm AR tags**. This module
is what turns them into geometry. It reports where each tag is and which way it
faces, in the rig frame, and it says nothing at all about what any of it *means* -
which tag is a berth's left wall, whether a berth is occupied, and which berth to
drive into are decisions for `ligmax-pi/nodes/self_driving/perception/artags.py`,
the same way this repo ships detections and lets the Pi build the world model.

Why this is not the same problem as the buoy detector
----------------------------------------------------
The detector's 2:1 band is swung 15 deg off each lens's axis, and **the tags on a
dock are not in it**. `rig.json` has the cameras looking port and starboard with
their fields meeting across the bow, so anything ahead of the boat is ~75 deg off
both optical axes: outside the crop, and near the edge of the frame where the
fisheye is at its most violent. So this reads the **whole sensor frame** - the same
image `sender.py` uploads as a full-resolution still - and the `cam*.json` fits,
which are for the full frame, apply without re-deriving a principal point.

The warp at the frame edge, and what it does and does not cost
--------------------------------------------------------------
This is the thing everyone expects to be the problem, so here are the numbers,
computed from the committed `cam*.json` fits at the geometry `rig.json` describes.

**It does not cost resolution.** A fisheye of this kind is near-equidistant: the
radial scale is roughly the focal length per radian *everywhere*, not just on
axis. An 18 cm tag dead ahead - 75 deg off both lens axes, the worst place on this
boat - projects to almost exactly the same number of pixels as it would at the
centre of the frame:

    range      1.0 m   1.5 m   2.0 m   3.0 m   4.0 m   5.0 m   6.0 m   8.0 m
    tag edge   157 px  105 px   79 px   53 px   40 px   32 px   26 px   20 px

and swinging the same tag from dead ahead round to the beam changes that by under
5 % (52.6 px -> 50.8 px at 3 m). The berth is worked at 0.5-3 m, where the tag is
50-160 px across. There is plenty of tag.

**It does not bend the tag either, at this size.** The honest worry about a wide
lens is that straight lines are not straight, so a corner refinement that assumes
they are will sit in the wrong place. Measured as the departure of a tag edge's
true image from the straight chord between its own two corner images:

    range              1.0 m     2.0 m     3.0 m     5.0 m
    sagitta, dead ahead  0.11 px   0.01 px   0.004 px  0.001 px
    sagitta, on the beam 0.20 px   0.02 px   0.008 px  0.002 px

A fifth of a pixel at 1 m and a two-hundredth at 3 m. The lens is savage over a
2592-pixel frame and locally affine over a 50-pixel tag, so `cornerSubPix` is
sound and there is **no remap, no rectified patch and no undistorted image**
anywhere in this module. That matters beyond tidiness: an undistorted full frame
at this field angle is enormous, mostly empty, and would have to be rebuilt for
every view direction.

**What the warp does cost is any code that treats the frame as a pinhole.** Over
the whole frame the departure is hundreds of pixels, so the pose must not come
from `solvePnP` on raw pixels with a pinhole `K`. It comes from the four corner
**rays** - `estimate.Camera.rays`, the exact Kannala-Brandt model, the same
function the buoy bearings go through - which are then handed to `solvePnP` as
ideal normalised coordinates with an identity camera matrix. All of the distortion
is dealt with before `solvePnP` sees anything, and none of it is approximated.

Positions, not normals
----------------------
Every geometric figure here is reported, but they are not equally good and a
consumer that treats them as equal will be misled.

**A tag's position is well conditioned.** Bearing is calibration-limited at about
0.25 deg (`estimate.CALIB_BEARING_SIGMA_DEG`), and range from a square's four
corners is roughly `sigma_z/z = sigma_px/edge_px` - about 1 % at 3 m and 0.3 % at
1 m with half-pixel corners. That is the lidar's league, from a sensor that also
tells you *which* tag it is.

**A tag's normal is not, and it is worse than "noisy".** A single planar square
has the classic two-fold pose ambiguity - two poses, mirrored about the line of
sight, that reproject almost identically - and at these sizes IPPE cannot reliably
tell them apart. Round-tripped through this module with *exact* synthetic corners
(known pose -> Kannala-Brandt projection -> `_measure`), the positions come back
to 0.09 deg of bearing and 3 mm of range across the whole working envelope, while
the normal does this:

    tag square-on to the boat        normal good to 2-4 deg
    tag yawed 15 deg off square      normal out by 28 deg
    tag yawed 20 deg off square      normal out by 38-42 deg
    tag yawed 25 deg off square      normal out by 49 deg

The error is close to twice the true tilt with the sign reversed, which is the
signature of the mirrored branch being chosen. **And `ambiguity` was 0.01 for all
of those** - the wrong branch fitted the corners as well as the right one, so that
figure does not rescue this. It is reported because a value near 1.0 is still
proof the normal is worthless; a low value is simply not proof of the opposite.
`incidence_deg` is the better guide, and only in the negative sense: below about
5 deg the normal is usable, above ~10 deg it is not.

So the consequence for whoever assembles a berth out of these is a rule, not a
preference: **never take the way into a berth from one tag's normal.** Two tags
2 m apart, each placed to 5 cm, give the line between them to about 1.5 deg, and
that is where the geometry has to come from. Where only one tag is visible the
honest fallback is the operator's own waypoint bearing - GNSS, laid by hand,
pointing into the berth - which is what `ligmax-pi` does.

The one number that has to be right
-----------------------------------
`rig.json`'s camera yaws, +-75 deg, which are **the mounting as described by hand
and never verified** - that file says so itself and says to check them. Everything
here is rotated by them, so a yaw that is 3 deg out puts every tag 3 deg out.

It matters more than it looks, because of how the fields divide. Each camera's
88 deg cone reaches only about 12 deg past the bow:

    cam0   rig bearings -162 .. +12 deg      (port, and just past the bow)
    cam1   rig bearings  -12 .. +162 deg     (starboard, and just past the bow)

So a berth 2 m wide seen from 1 m away has its two sides at +-45 deg, which is
**one tag in each camera**. A berth is therefore assembled across the pair, and
the yaws are what hold the two halves in register.

Two things here exist to catch that, and both are reported rather than acted on:

  * a tag within +-12 deg of the bow is seen by **both** cameras, and the two
    bearings for one tag id are a direct measurement of the yaw error. Put a tag
    dead ahead on the bench and `bench_check()` prints what to change.
  * the **measured** distance between two tags whose real separation is known -
    a 2 m berth mouth - is the same measurement in disguise, and it is available
    while the boat is working. `ligmax-pi` publishes it as `mouth_m` beside the
    nominal figure for exactly this reason.

Nothing here reads the lidar, so the *other* known-wrong number in `rig.json` -
`lidar.yaw_deg`, stale since the unit was remounted - cannot affect a tag.

OpenCV
------
`cv2.aruco` moved out of contrib in 4.7 and the old procedural API was removed.
Both are handled (`_Backend`), because what JetPack ships is not something this
repo gets to choose. If the build has no `aruco` at all this module imports fine
and `TagFinder` refuses to construct with a message naming the package to install,
rather than taking `sender.py` down at start-up.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:                                 # pragma: no cover
    cv2 = None

#: The black square's outer edge, metres. NJORD §9.3 / §10.4: the tags are printed
#: on A4 and the square measures 18 cm, which is what the detector's corners are
#: the corners of. Not the paper, and not any white margin around it.
TAG_M = 0.18

#: 4x4, 50 ids. The nine tags in `ArUco_tags_on_dock.zip` decode as ids 0-7 in
#: every 4x4 dictionary, so the smallest is the right one: fewest ids means the
#: largest Hamming distance between them and the fewest false positives off wet
#: concrete and shackles. **The handbook does not publish the family**, so this is
#: what the organisers' own files turned out to be and it is overridable.
DICT_NAME = "DICT_4X4_50"

#: Forward window to search, degrees of rig bearing and elevation. The tags are on
#: a dock ahead of the boat, and cropping to that costs nothing and buys back three
#: quarters of a 5 Mpx `detectMarkers`. Wide enough to hold a 2 m berth from half a
#: metre out (+-64 deg) with margin.
WINDOW_DEG = 70.0
ELEVATION_DEG = 25.0

#: Smallest tag worth believing, pixels along the shortest edge. A 4x4 marker is
#: 6 cells including its border, so this is a bit over 2 px per cell - about where
#: OpenCV stops finding them at all. At the geometry above it corresponds to
#: roughly 10 m dead ahead, which is further than the task needs.
MIN_EDGE_PX = 14.0

#: Corner repeatability used for the reported range sigma. Half a pixel is what
#: `cornerSubPix` achieves on a clean high-contrast square; `estimate.py` uses
#: 3.0 px for an unrefined detector box, and a tag corner is a much better
#: measurement than a YOLO box edge.
CORNER_SIGMA_PX = 0.5


class TagError(RuntimeError):
    """Raised at construction when the OpenCV build cannot do this at all."""


class _Backend:
    """`detectMarkers` across the 4.7 API change, chosen once at construction."""

    def __init__(self, dict_name, refine=True, min_perimeter_px=56.0,
                 image_dim=1282):
        if cv2 is None:
            raise TagError("OpenCV is not installed, so no tag can be found")
        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            raise TagError(
                "this OpenCV build has no cv2.aruco (it lived in opencv-contrib "
                f"before 4.7; this is {cv2.__version__}). Install "
                "opencv-contrib-python, or a 4.7+ build where aruco is in core."
            )
        if not hasattr(aruco, dict_name):
            raise TagError(f"cv2.aruco has no dictionary {dict_name!r}")
        which = getattr(aruco, dict_name)

        # 4.7+: getPredefinedDictionary / DetectorParameters / ArucoDetector.
        # Earlier: Dictionary_get / DetectorParameters_create / detectMarkers().
        if hasattr(aruco, "ArucoDetector"):
            self.dictionary = aruco.getPredefinedDictionary(which)
            self.params = aruco.DetectorParameters()
            self._tune(self.params, aruco, refine, min_perimeter_px, image_dim)
            self._detector = aruco.ArucoDetector(self.dictionary, self.params)
            self._legacy = False
        else:                                       # pragma: no cover - old JetPack
            self.dictionary = aruco.Dictionary_get(which)
            self.params = aruco.DetectorParameters_create()
            self._tune(self.params, aruco, refine, min_perimeter_px, image_dim)
            self._detector = None
            self._legacy = True
        self._aruco = aruco

    @staticmethod
    def _tune(params, aruco, refine, min_perimeter_px, image_dim):
        """The parameters that decide whether this fits in a frame period.

        **`minMarkerPerimeterRate` is the one that matters, by two orders of
        magnitude.** It is a fraction of the image's larger dimension, and OpenCV
        walks every contour at least that long looking for a quad. Measured on a
        real 2592x1944 still from this boat, cropped to the forward search window:

            rate 0.0005 (min perimeter 0.6 px)      6400 ms per frame
            rate 0.0437 (min perimeter 56 px)         57 ms per frame

        Same tag found either way. The first number is not a slow frame, it is a
        stalled capture loop -- 90 frame periods -- so this is derived from the
        smallest tag worth reporting rather than set to a small-looking constant:

            min perimeter = 4 * min_edge_px

        Subpixel refinement is the reason the pose is worth computing at all. The
        default `CORNER_REFINE_NONE` gives integer corners, and at 50 px a pixel of
        corner error is 2 % of range; measured on the same still it moves the
        corners by 1-3 px and costs about 3 ms. It is sound on this lens -- see the
        sagitta table in the module docstring.

        The adaptive-threshold sweep is 3 passes rather than OpenCV's 3 at wider
        spacing: measured 59 ms against 76 ms for the 5-pass version that was here
        first, finding the same tag. Wet paper in shadow against bright water is the
        case it exists for, and 5 to 21 covers it.
        """
        if refine and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            try:
                params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
                params.cornerRefinementWinSize = 5
                params.cornerRefinementMaxIterations = 50
                params.cornerRefinementMinAccuracy = 0.01
            except Exception:                       # pragma: no cover
                pass
        try:
            params.minMarkerPerimeterRate = min_perimeter_px / max(image_dim, 1)
            params.maxMarkerPerimeterRate = 4.0
        except Exception:                           # pragma: no cover
            pass
        try:
            params.adaptiveThreshWinSizeMin = 5
            params.adaptiveThreshWinSizeMax = 21
            params.adaptiveThreshWinSizeStep = 8
        except Exception:                           # pragma: no cover
            pass

    def detect(self, gray):
        if self._legacy:                            # pragma: no cover - old JetPack
            corners, ids, _ = self._aruco.detectMarkers(
                gray, self.dictionary, parameters=self.params)
        else:
            corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []
        return [(int(i), c.reshape(4, 2).astype(np.float64))
                for i, c in zip(ids.flatten(), corners)]


class TagFinder:
    """Finds tags in both cameras' full frames and measures them in the rig frame.

    `cams` is the pair of `estimate.Camera` models and `rig` is a `fusion.Rig`,
    i.e. exactly what `sender.py` already built for the buoy bearings and the lidar
    colouring. Nothing here is constructed from a file path, so a run with no
    calibration gets no tags rather than tags in an invented frame.
    """

    def __init__(self, cams, rig, *, tag_m=TAG_M, dict_name=DICT_NAME,
                 window_deg=WINDOW_DEG, elevation_deg=ELEVATION_DEG,
                 min_edge_px=MIN_EDGE_PX, refine=True, max_tags=16):
        if len(cams) != 2:
            raise TagError("need both camera models")
        if any(c is None for c in cams):
            raise TagError(
                "a tag's position is meaningless without the fisheye model it was "
                "measured through -- calibrate/calib/cam0.json and cam1.json have "
                "to load (see --calib)"
            )
        if rig is None:
            raise TagError(
                "rig.json is what turns a camera-frame pose into a boat-relative "
                "one; without it there is no bearing to report"
            )
        self.cams = list(cams)
        self.rig = rig
        self.tag_m = float(tag_m)
        self.min_edge_px = float(min_edge_px)
        self.max_tags = int(max_tags)
        self.dict_name = dict_name

        # The tag's own corners, in its own frame, in the order OpenCV returns
        # them: top-left, top-right, bottom-right, bottom-left, with +x right,
        # +y up and +z out of the face towards whoever is looking at it. This
        # ordering is what SOLVEPNP_IPPE_SQUARE is specified against, and getting
        # it wrong yields a pose that reprojects perfectly and faces backwards.
        h = 0.5 * self.tag_m
        self.object_points = np.array([[-h, h, 0.0], [h, h, 0.0],
                                       [h, -h, 0.0], [-h, -h, 0.0]])

        self.rois = [self._roi(i, window_deg, elevation_deg) for i in (0, 1)]
        # Built last, because the perimeter threshold is a fraction of the image it
        # runs on and the image is the search window, not the sensor frame. Derived
        # from the SMALLEST window of the two so neither camera is stricter than it
        # was asked to be.
        dims = [max(x1 - x0, y1 - y0)
                for roi in self.rois if roi is not None
                for (x0, y0, x1, y1) in (roi,)]
        image_dim = min(dims) if dims else max(self.cams[0].image_size)
        self.backend = _Backend(dict_name, refine=refine,
                                min_perimeter_px=4.0 * self.min_edge_px,
                                image_dim=image_dim)
        self.errors = 0
        self.last_error = None

    # ------------------------------------------------------------------- setup
    def _roi(self, index, window_deg, elevation_deg):
        """Bounding box in full-res pixels of the forward window, or None.

        Computed by projecting the window rather than assumed, so it follows the
        rig and the calibration instead of being a pair of magic numbers that
        quietly stop meaning anything when a camera is re-bolted. Points outside
        the calibrated cone or off the sensor simply do not contribute, which is
        why a 140 x 50 deg window comes out as a quarter of the frame and not all
        of it.
        """
        cam = self.cams[index]
        pose = self.rig.cams[index]
        w, h = cam.image_size
        brg = np.radians(np.linspace(-window_deg, window_deg, 61))
        elv = np.radians(np.linspace(-elevation_deg, elevation_deg, 21))
        b, e = np.meshgrid(brg, elv, indexing="ij")
        # Rig frame: +x starboard, +y DOWN, +z forward. Elevation is positive up,
        # hence the minus on y.
        d = np.stack([np.cos(e) * np.sin(b), -np.sin(e), np.cos(e) * np.cos(b)],
                     axis=-1).reshape(-1, 3)
        p_cam = d @ pose.R                          # rotation only: a direction
        uv, valid = cam.project(p_cam)
        keep = (valid & (p_cam[:, 2] > 0)
                & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        if not keep.any():
            return None
        u = uv[keep, 0]
        v = uv[keep, 1]
        # A tag is up to ~160 px across at half a metre and its centre may sit on
        # the window's edge, so the box is grown by that much or it would clip the
        # very tags it exists to find.
        pad = 96.0
        x0 = int(max(0, math.floor(u.min() - pad)))
        y0 = int(max(0, math.floor(v.min() - pad)))
        x1 = int(min(w, math.ceil(u.max() + pad)))
        y1 = int(min(h, math.ceil(v.max() + pad)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return (x0, y0, x1, y1)

    def describe(self):
        out = [f"  dictionary {self.dict_name}, tag {self.tag_m * 100:.0f} cm, "
               f"min edge {self.min_edge_px:.0f} px"]
        for i, roi in enumerate(self.rois):
            if roi is None:
                out.append(f"  cam{i}: forward window not visible -- check rig.json")
                continue
            x0, y0, x1, y1 = roi
            w, h = self.cams[i].image_size
            out.append(
                f"  cam{i}: search {x1 - x0}x{y1 - y0} at ({x0},{y0}) "
                f"= {100.0 * (x1 - x0) * (y1 - y0) / (w * h):.0f}% of {w}x{h}, "
                f"yaw {self.rig.cams[i].yaw:+.0f} deg (UNVERIFIED -- see rig.json)"
            )
        return "\n".join(out)

    # ------------------------------------------------------------------ finding
    def find(self, index, gray_full):
        """Tags in one camera's full-resolution luma plane. `[]` if none.

        `gray_full` is `sender.sample_to_y`'s array: the whole sensor frame, one
        byte per pixel, upright. Handed the detector's downscaled input instead
        this would still find tags and every range would be wrong, so the shape is
        checked against the calibration rather than trusted.
        """
        if gray_full is None:
            return []
        cam = self.cams[index]
        w, h = cam.image_size
        if gray_full.shape[0] != h or gray_full.shape[1] != w:
            self.errors += 1
            self.last_error = (
                f"cam{index} frame is {gray_full.shape[1]}x{gray_full.shape[0]} "
                f"but the calibration is for {w}x{h}; a tag measured through the "
                f"wrong model is wrong by whatever the scale factor is"
            )
            return []
        roi = self.rois[index]
        if roi is None:
            return []
        x0, y0, x1, y1 = roi
        patch = gray_full[y0:y1, x0:x1]
        if not patch.flags["C_CONTIGUOUS"]:
            patch = np.ascontiguousarray(patch)

        try:
            found = self.backend.detect(patch)
        except Exception as exc:                    # never let a frame die of this
            self.errors += 1
            self.last_error = f"cam{index} detectMarkers: {exc}"
            return []

        out = []
        for tag_id, corners in found[: self.max_tags]:
            corners = corners + np.array([x0, y0], dtype=np.float64)
            try:
                item = self._measure(index, tag_id, corners)
            except Exception as exc:                # geometry must not kill a frame
                self.errors += 1
                self.last_error = f"cam{index} tag {tag_id}: {exc}"
                continue
            if item is not None:
                out.append(item)
        return out

    def _measure(self, index, tag_id, corners):
        """One tag's corners -> the reported dict, or None if not worth reporting."""
        cam = self.cams[index]
        edges = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
                 for i in range(4)]
        edge_px = min(edges)
        if edge_px < self.min_edge_px:
            return None

        # Corner pixels -> exact unit rays through the Kannala-Brandt model, then
        # to ideal normalised coordinates. THIS is where the fisheye is dealt
        # with; everything after it is a pinhole problem with an identity K.
        rays = cam.rays(corners)
        if not np.isfinite(rays).all():
            # At least one corner is outside the calibrated 88 deg cone, so no
            # bearing exists for it. A pose fitted from three good corners and one
            # invented one is worse than no pose.
            return None
        if (rays[:, 2] <= 1e-6).any():
            return None
        image_points = np.stack([rays[:, 0] / rays[:, 2],
                                 rays[:, 1] / rays[:, 2]], axis=1)

        pose = self._solve(image_points)
        if pose is None:
            return None
        rvec, tvec, reproj, ambiguity = pose

        R_tag, _ = cv2.Rodrigues(rvec)
        # Camera -> rig. The virtual pinhole shares the real lens's centre, so
        # only the rotation applies to a direction, but a POINT also needs the
        # camera's offset from the rig origin -- 5 cm, which is 1.4 deg of bearing
        # at 2 m and therefore not ignorable inside a 2 m berth.
        cam_pose = self.rig.cams[index]
        p_rig = cam_pose.to_rig(tvec.reshape(1, 3))[0]
        normal_rig = cam_pose.R @ (R_tag @ np.array([0.0, 0.0, 1.0]))

        rng_cam = float(np.linalg.norm(tvec))
        # sigma_z/z = sigma_px/edge_px: the range comes from apparent size, and
        # the apparent size is known to a corner's worth of pixels. Not the whole
        # story -- it takes no account of the tag being printed slightly off 18 cm
        # -- so a 2 mm printing tolerance is folded in as well.
        rel = CORNER_SIGMA_PX / max(edge_px, 1.0)
        sigma_m = math.hypot(rng_cam * rel, rng_cam * 0.002 / self.tag_m)

        bearing = math.degrees(math.atan2(p_rig[0], p_rig[2]))
        horiz = math.hypot(p_rig[0], p_rig[2])
        elevation = math.degrees(math.atan2(-p_rig[1], horiz))

        # How square-on the tag is: 0 deg means it faces the camera exactly. A tag
        # seen past ~70 deg of this is a tag whose corners are nearly collinear,
        # and its pose degrades long before the detector stops finding it.
        los = p_rig / max(float(np.linalg.norm(p_rig)), 1e-9)
        incidence = math.degrees(
            math.acos(max(-1.0, min(1.0, float(np.dot(-normal_rig, los)))))
        )

        return {
            "id": int(tag_id),
            "cam": int(index),
            "corners": [[round(float(u), 2), round(float(v), 2)]
                        for u, v in corners],
            "centre_px": [round(float(corners[:, 0].mean()), 1),
                          round(float(corners[:, 1].mean()), 1)],
            "edge_px": round(edge_px, 1),
            # Boat-relative, which is the only frame the Pi wants any of this in.
            "pos_rig": [round(float(v), 4) for v in p_rig],
            "range_m": round(float(np.linalg.norm(p_rig)), 3),
            "sigma_m": round(sigma_m, 3),
            "bearing_deg": round(bearing, 2),
            "elevation_deg": round(elevation, 2),
            # Which way the face points. NOT to be used as a berth's axis: see the
            # measured table in the module docstring. `ambiguity` near 1.0 proves
            # the normal is worthless; a low value proves nothing, because the
            # mirrored branch fits just as well at these sizes.
            "normal_rig": [round(float(v), 4) for v in normal_rig],
            "facing_deg": round(math.degrees(math.atan2(normal_rig[0],
                                                        normal_rig[2])), 2),
            "incidence_deg": round(incidence, 1),
            "ambiguity": None if ambiguity is None else round(ambiguity, 3),
            "reproj_px": round(reproj, 3),
        }

    def _solve(self, image_points):
        """IPPE on a square: `(rvec, tvec, reproj_px_equivalent, ambiguity)`.

        `solvePnPGeneric` returns *both* planar solutions and both their errors,
        which is the only honest way to report how much the tag's normal is worth.
        Where the build has no `solvePnPGeneric`, the single-solution path is used
        and `ambiguity` is None rather than a fabricated 0.
        """
        obj = self.object_points.reshape(-1, 1, 3)
        img = image_points.reshape(-1, 1, 2)
        eye = np.eye(3)
        zero = np.zeros(5)
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

        if hasattr(cv2, "solvePnPGeneric"):
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, eye, zero, flags=flag)
            if not n:
                return None
            errors = [float(e) for e in np.asarray(errs).reshape(-1)] if errs is not None else []
            best = 0
            if errors:
                best = int(np.argmin(errors))
            rvec, tvec = rvecs[best], tvecs[best]
            if tvec[2] <= 0:
                return None
            ambiguity = None
            if len(errors) >= 2:
                lo = min(errors)
                hi = max(errors)
                ambiguity = 1.0 if hi <= 1e-12 else min(1.0, lo / hi)
            reproj = errors[best] if errors else 0.0
        else:                                       # pragma: no cover
            ok, rvec, tvec = cv2.solvePnP(obj, img, eye, zero, flags=flag)
            if not ok or tvec[2] <= 0:
                return None
            ambiguity = None
            reproj = 0.0

        # The error above is in NORMALISED units, because that is the space
        # solvePnP was handed. Multiply by the focal length to get something a
        # human can compare against a pixel.
        return rvec, tvec, reproj, ambiguity

    # ------------------------------------------------------------ bench helper
    def bench_check(self, tags):
        """What the two cameras disagree by about a tag they can both see.

        `rig.json`'s camera yaws are hand-described and unverified, and they are
        what puts one camera's half of a berth in register with the other's. The
        pair overlaps for about 24 deg across the bow, so a tag placed dead ahead
        is measured twice and the difference is the yaw error, undiluted.

        Returns one entry per tag id seen by both cameras. `bearing_error_deg` is
        what to *add* to `cam0.yaw_deg` and subtract from `cam1.yaw_deg` to bring
        them together; it does not say which of the two is wrong, so the absolute
        fix still needs the tag to be truly ahead - a tape measure off the
        centreline, not an eyeball.
        """
        by_id = {}
        for t in tags:
            by_id.setdefault(t["id"], {})[t["cam"]] = t
        out = []
        for tag_id, pair in sorted(by_id.items()):
            if 0 not in pair or 1 not in pair:
                continue
            a, b = pair[0], pair[1]
            out.append({
                "id": tag_id,
                "cam0_bearing_deg": a["bearing_deg"],
                "cam1_bearing_deg": b["bearing_deg"],
                "bearing_error_deg": round(b["bearing_deg"] - a["bearing_deg"], 2),
                "cam0_range_m": a["range_m"],
                "cam1_range_m": b["range_m"],
                "range_error_m": round(b["range_m"] - a["range_m"], 3),
                "edge_px": [a["edge_px"], b["edge_px"]],
            })
        return out


def stats_line(tags, finder):
    """One readable line for the Jetson's console, in `_lidar_line`'s spirit."""
    if not tags:
        extra = "" if finder is None or not finder.errors else \
            f" err={finder.errors}"
        return f"tags=0{extra}"
    ids = sorted({t["id"] for t in tags})
    nearest = min(tags, key=lambda t: t["range_m"])
    worst = max(t["ambiguity"] or 0.0 for t in tags)
    line = (f"tags={len(tags)} ids={','.join(str(i) for i in ids)} "
            f"near={nearest['range_m']:.2f}m@{nearest['bearing_deg']:+.0f}deg "
            f"edge={nearest['edge_px']:.0f}px amb<={worst:.2f}")
    if finder is not None and finder.errors:
        line += f" err={finder.errors}"
    return line
