#!/usr/bin/env python3
"""Measure the camera yaws from AR-tag photos. The one number `rig.json` guesses.

What this settles
-----------------
`rig.json` puts cam0 at yaw -75 deg and cam1 at +75, and says of those numbers,
about itself, that they are "the mounting as described by hand" and "VERIFY THEM
... before trusting a bearing". Everything the boat believes about *which side*
an object is on is rotated by them. For NJORD 9.2 that is not a detail: the
difference between "head-on" and "crossing from starboard" is the difference
between altering course and standing still, and both are decided from a bearing.

The measurement needs no rig, no water and no lidar. The pair overlaps for about
24 deg across the bow, so **a tag held ahead of the boat is measured twice**, by
two cameras whose intrinsics are known to about a pixel. The two answers should
agree. What they disagree by is the yaw error, undiluted, and `artags.bench_check`
is the function that reports it - this script only feeds it stills instead of a
live capture.

What it CANNOT settle, and this matters
---------------------------------------
The disagreement says the two cameras are out of register with each other. It
does **not** say which of them is wrong, because a tag that is not truly dead
ahead moves both readings together. Splitting the error evenly assumes the pair
is symmetric about the bow, which is what `rig.json` assumes too.

To pin the absolute yaw you need a tag placed on the centreline **by a tape
measure, not by eye** - then its true rig bearing is 0 and each camera's own
error is `bearing - 0`. Hand-held photos cannot give you that, so this reports
the relative figure honestly and says so.

    python test/test_camera_yaw.py ../rl_ar_tags_front
    python test/test_camera_yaw.py ../rl_ar_tags_front --tag-m 0.18
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import artags          # noqa: E402
import estimate        # noqa: E402
import fusion          # noqa: E402

import cv2             # noqa: E402
import numpy as np     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.dirname(HERE)


def pairs(directory):
    """`{stamp: {0: path, 1: path}}` from `<stamp>-cam<N>.jpg` filenames."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.jpg"))):
        m = re.match(r"(.+)-cam([01])\.jpg$", os.path.basename(path))
        if m:
            found.setdefault(m.group(1), {})[int(m.group(2))] = path
    return {k: v for k, v in found.items() if 0 in v and 1 in v}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="directory of <stamp>-cam0.jpg / -cam1.jpg pairs")
    ap.add_argument("--calib", default=os.path.join(EDGE, "calibrate", "calib"))
    ap.add_argument("--rig", default=os.path.join(EDGE, "rig.json"))
    ap.add_argument("--tag-m", type=float, default=artags.TAG_M,
                    help="printed tag edge, metres. The dock set is 0.18.")
    ap.add_argument("--dict", default=artags.DICT_NAME)
    ap.add_argument("--window", type=float, default=180.0,
                    help="rig-bearing search window. Wide by default: the live "
                         "pipeline crops to the forward 70 deg because a berth is "
                         "ahead, but a hand-held tag lands wherever it lands and "
                         "cropping it away would look exactly like a tag that was "
                         "not detected.")
    ap.add_argument("--min-px", type=float, default=artags.MIN_EDGE_PX)
    args = ap.parse_args()

    # Full-frame geometry: these stills ARE the sensor frame (2592x1944), so
    # there is no crop and no downscale to describe. The live path hands
    # `estimate.Camera` a crop and a net size; here both are the image itself.
    cams = []
    for name in ("cam0.json", "cam1.json"):
        model = estimate.Camera.load(os.path.join(args.calib, name))
        w, h = model.image_size
        cams.append(estimate.Camera.load(os.path.join(args.calib, name),
                                         crop_left=0, crop_top=0,
                                         crop_w=w, crop_h=h, net_w=w, net_h=h))
    rig = fusion.Rig.load(args.rig)

    finder = artags.TagFinder(cams, rig, tag_m=args.tag_m, dict_name=args.dict,
                              window_deg=args.window, min_edge_px=args.min_px)
    print(finder.describe())
    print()

    sets = pairs(args.images)
    if not sets:
        raise SystemExit(f"no <stamp>-cam0.jpg / -cam1.jpg pairs in {args.images}")

    errors, rows = [], []
    for stamp, both in sorted(sets.items()):
        frames = [cv2.imread(both[0]), cv2.imread(both[1])]
        if any(f is None for f in frames):
            print(f"{stamp}: unreadable")
            continue
        # artags wants RGB, like everything downstream of the Jetson's pipeline.
        tags = []
        for i in (0, 1):
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            tags.extend(finder.find(i, rgb))

        seen = {}
        for t in tags:
            seen.setdefault(t["cam"], []).append(t["id"])
        both_cams = finder.bench_check(tags)

        if not both_cams:
            print(f"{stamp}: cam0 saw {sorted(seen.get(0, []))}, "
                  f"cam1 saw {sorted(seen.get(1, []))} -- no tag in BOTH")
            continue
        for row in both_cams:
            errors.append(row["bearing_error_deg"])
            rows.append((stamp, row))
            print(f"{stamp}: tag {row['id']:>3}  "
                  f"cam0 {row['cam0_bearing_deg']:+7.2f} deg / "
                  f"{row['cam0_range_m']:.2f} m   "
                  f"cam1 {row['cam1_bearing_deg']:+7.2f} deg / "
                  f"{row['cam1_range_m']:.2f} m   "
                  f"-> disagree {row['bearing_error_deg']:+6.2f} deg, "
                  f"{row['range_error_m']:+.3f} m")

    print()
    if not errors:
        print("NO TAG WAS SEEN BY BOTH CAMERAS, so the yaws are unmeasured.")
        print("The overlap is only about 24 deg wide across the bow. Re-shoot with")
        print("the tag held straight off the bow, square to it, 1-3 m out, and")
        print("filling a decent part of the frame.")
        return 1

    arr = np.array(errors, dtype=float)
    print(f"{len(arr)} paired sighting(s)")
    print(f"  disagreement  mean {arr.mean():+.2f} deg   "
          f"median {np.median(arr):+.2f}   sd {arr.std(ddof=0):.2f}   "
          f"range {arr.min():+.2f} .. {arr.max():+.2f}")
    print()
    print("  Splitting it evenly between the two cameras:")
    print(f"    cam0.yaw_deg  {rig.cams[0].yaw:+.1f}  ->  "
          f"{rig.cams[0].yaw + arr.mean() / 2.0:+.2f}")
    print(f"    cam1.yaw_deg  {rig.cams[1].yaw:+.1f}  ->  "
          f"{rig.cams[1].yaw - arr.mean() / 2.0:+.2f}")
    print()
    print("  This is a RELATIVE measurement. It says the two cameras are out of")
    print("  register by the figure above; it does not say which one is wrong.")
    print("  For that, place a tag on the centreline with a tape measure - its")
    print("  true rig bearing is then 0 and each camera's own error is readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
