#!/usr/bin/env python3
"""Draw the lidar's returns onto a real camera frame, to check rig.json.

    ./.venv/bin/python test/test_lidar_overlay.py                 # both cameras
    ./.venv/bin/python test/test_lidar_overlay.py --cam 0
    ./.venv/bin/python test/test_lidar_overlay.py --yaw 12.5      # try a value
    ./.venv/bin/python test/test_lidar_overlay.py --dy -0.07 --shots 5

This is the tool that turns the hand-measured numbers in rig.json into something
you can be WRONG about visibly. The mounting offsets are tape-measure figures;
nothing else in the pipeline can tell you they are off, because a slightly wrong
transform still produces a full, plausible, entirely mis-registered point cloud.

What a correct overlay looks like
---------------------------------
Put something with a hard vertical edge 1-3 m in front of a camera -- a door
frame, a table leg, a person standing still. The returns off it must land ON it
in the image, not beside it.

  * points sit consistently LEFT or RIGHT of where they belong  -> yaw
  * points sit above/below the object, converging with range    -> dy (height)
  * the error grows the further off-centre you look             -> yaw, not dx
  * near objects land right, far ones drift (or the reverse)    -> dz, or dy
  * the whole world is mirrored left-for-right                  -> angle_dir
  * everything is rotated by a constant                         -> angle_zero_deg

The overrides here change ONE run without touching the file, so you can sweep a
value and keep the one that lands. Write the winner into rig.json when it looks
right; nothing reads these flags but this script.

Colour encodes range (near red -> far blue) rather than the camera's own colour,
because the question here is geometric: a return drawn in the colour it sampled
would agree with the background by construction and hide the very error you are
looking for.
"""

import argparse
import math
import os
import subprocess
import sys
import time

import gi
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp  # noqa: E402,F401

import estimate  # noqa: E402
import fusion  # noqa: E402
import lidar as lidar_mod  # noqa: E402
import sender  # noqa: E402

NET_W, NET_H = 1280, 640


def build(cam_id, mode, crop, net_w, net_h, rotate):
    """One camera, cropped exactly as sender.py crops it, RGB to system memory.

    Same crop or the overlay is meaningless: a point projected into a different
    window lands somewhere else entirely.
    """
    w, h, fps = sender.SENSOR[mode]
    cl, ct, cw, ch = crop
    flip = "" if not rotate else " flip-method=2"
    return (
        f"nvarguscamerasrc sensor-id={cam_id} sensor-mode={mode} "
        f"wbmode={sender.WB_MODES.index('daylight')} do-timestamp=true "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 "
        f"! nvvideoconvert{flip} "
        f"! video/x-raw(memory:NVMM),width={w},height={h},format=NV12 "
        f"! nvvideoconvert src-crop={cl}:{ct}:{cw}:{ch} compute-hw=GPU "
        f"! video/x-raw,format=RGB,width={net_w},height={net_h} "
        # The queue is not optional, and its absence does not look like a missing
        # queue. Without it this tool captures fine and then HANGS on shutdown,
        # inside send_event(EOS) -- once we stop pulling samples there is nothing
        # willing to drop a buffer, the pads stay blocked, and the event cannot
        # travel. sender.py carries leaky=downstream on every branch for the same
        # reason. Hanging here is not cosmetic: the process then has to be killed,
        # which is exactly the un-drained teardown that latches Argus.
        f"! queue max-size-buffers=2 leaky=downstream "
        f"! appsink name=out emit-signals=false max-buffers=2 drop=true sync=false"
    )


def range_colour(r, max_range=12.0):
    """Near red -> mid green -> far blue, so depth is readable at a glance."""
    f = max(0.0, min(1.0, r / max_range))
    if f < 0.5:
        g = f / 0.5
        return (int(255 * (1 - g)), int(200 * g), 40)
    g = (f - 0.5) / 0.5
    return (30, int(200 * (1 - g)), int(60 + 195 * g))


def overlay(cam_id, args, rig, reader):
    model_path = os.path.join(args.calib, f"cam{cam_id}.json")
    with open(model_path, encoding="utf-8") as f:
        import json
        model = json.load(f)

    w, h, fps = sender.SENSOR[args.mode]
    crop_w = min(w, max(NET_W, args.crop_w))
    crop_h = min(h, int(round(crop_w * NET_H / NET_W)))
    aim = args.aim_deg if cam_id == 0 else -args.aim_deg
    cl, ct, got = estimate.crop_for_aim(model, aim, crop_w, crop_h)
    if args.crop_top is not None:
        ct = max(0, min(h - crop_h, args.crop_top))
    cam = estimate.Camera(model, cl, ct, crop_w, crop_h, NET_W, NET_H)
    print(f"cam{cam_id}: crop {crop_w}x{crop_h} at ({cl},{ct}), aim {got:+.1f} deg")

    Gst.init(None)
    pipe = Gst.parse_launch(build(cam_id, args.mode, (cl, ct, crop_w, crop_h),
                                  NET_W, NET_H, not args.no_rotate))
    sink = pipe.get_by_name("out")
    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print(f"cam{cam_id}: pipeline failed to start", file=sys.stderr)
        return
    pipe.get_state(Gst.CLOCK_TIME_NONE)
    clock = estimate.CaptureClock(pipe.get_base_time(), fps)

    written = []
    try:
        # Let Argus settle and the sweep buffer fill before taking the shot that
        # gets saved; the first frames after PLAYING are also the worst exposed.
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and reader.latest() is None:
            time.sleep(0.1)
        for shot in range(args.shots):
            rgb = pts = None
            for _ in range(30):
                s = sink.try_pull_sample(int(0.5 * Gst.SECOND))
                if s is None:
                    continue
                rgb, pts = sender.sample_to_rgb(s, NET_W, NET_H)
                if rgb is not None:
                    break
            if rgb is None:
                print(f"cam{cam_id}: no frame", file=sys.stderr)
                return
            t_cap = clock.frame_time(pts)
            sweep, skew = reader.sweep_near(t_cap)
            if sweep is None:
                print("no lidar sweep buffered yet", file=sys.stderr)
                return

            p_rig = rig.sweep_to_rig(sweep)
            p_cam = rig.cams[cam_id].from_rig(p_rig)
            uv_full, in_cone = cam.project(p_cam)
            uv = cam.to_net(uv_full)
            r = np.linalg.norm(p_rig, axis=1)
            vis = (in_cone & (uv[:, 0] >= 0) & (uv[:, 0] < NET_W)
                   & (uv[:, 1] >= 0) & (uv[:, 1] < NET_H))

            im = Image.fromarray(rgb)
            d = ImageDraw.Draw(im)
            for i in np.flatnonzero(vis):
                x, y = float(uv[i, 0]), float(uv[i, 1])
                c = range_colour(float(r[i]), args.max_range)
                d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=c)
            # The optical axis, as the reference the yaw error is measured against.
            ax = cam.to_net(np.array([[cam.K[0, 2], cam.K[1, 2]]]))[0]
            d.line([ax[0], 0, ax[0], NET_H], fill=(255, 255, 255))
            d.line([0, ax[1], NET_W, ax[1]], fill=(90, 90, 90))

            pose = rig.cams[cam_id]
            tag = (f"cam{cam_id}  {int(vis.sum())}/{len(sweep)} returns in frame  "
                   f"skew {1000 * skew:+.0f} ms  "
                   f"yaw {pose.yaw:+.2f} dy {pose.t[1]:+.3f}")
            d.rectangle([0, 0, d.textlength(tag) + 6, 14], fill=(0, 0, 0))
            d.text((3, 1), tag, fill=(255, 255, 255))

            name = os.path.abspath(f"lidar_overlay_cam{cam_id}_{shot}.jpg")
            im.save(name, quality=90)
            written.append(name)
            nearest = float(r[vis].min()) if vis.any() else float("nan")
            print(f"  shot {shot}: {int(vis.sum())} returns in frame, "
                  f"nearest {nearest:.2f} m, skew {1000 * skew:+.0f} ms -> {name}")
    finally:
        # EOS, never a hard stop: tearing capture down mid-stream latches Argus
        # in an error state that only a power cycle clears.
        pipe.send_event(Gst.Event.new_eos())
        pipe.get_bus().timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.EOS)
        pipe.set_state(Gst.State.NULL)
        pipe.get_state(3 * Gst.SECOND)
    return written


def _passthrough(args):
    """Rebuild the flags a per-camera child needs, minus --cam itself."""
    out = []
    for flag, val in (("--rig", args.rig), ("--calib", args.calib),
                      ("--mode", args.mode), ("--crop-w", args.crop_w),
                      ("--crop-top", args.crop_top), ("--aim-deg", args.aim_deg),
                      ("--shots", args.shots), ("--max-range", args.max_range),
                      ("--lidar-port", args.lidar_port), ("--yaw", args.yaw),
                      ("--pitch", args.pitch), ("--roll", args.roll),
                      ("--dx", args.dx), ("--dy", args.dy), ("--dz", args.dz),
                      ("--angle-dir", args.angle_dir),
                      ("--angle-zero", args.angle_zero)):
        if val is not None:
            out += [flag, str(val)]
    if args.no_rotate:
        out.append("--no-rotate")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", type=int, default=None, choices=(0, 1),
                    help="one camera; default does both in turn")
    ap.add_argument("--rig", default="rig.json")
    ap.add_argument("--calib", default="calibrate/calib")
    ap.add_argument("--mode", type=int, default=0, choices=sorted(sender.SENSOR))
    ap.add_argument("--crop-w", type=int, default=2048)
    ap.add_argument("--crop-top", type=int, default=None)
    ap.add_argument("--aim-deg", type=float, default=15.0)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--max-range", type=float, default=12.0,
                    help="range that saturates the colour ramp")
    ap.add_argument("--lidar-port", default=None)
    # One-run overrides, so a value can be swept without editing rig.json.
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--roll", type=float, default=None)
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--dy", type=float, default=None)
    ap.add_argument("--dz", type=float, default=None)
    ap.add_argument("--angle-dir", type=int, default=None, choices=(-1, 1))
    ap.add_argument("--angle-zero", type=float, default=None)
    args = ap.parse_args()

    with open(args.rig, encoding="utf-8") as f:
        import json
        spec = json.load(f)
    if args.angle_dir is not None:
        spec.setdefault("lidar", {})["angle_dir"] = args.angle_dir
    if args.angle_zero is not None:
        spec.setdefault("lidar", {})["angle_zero_deg"] = args.angle_zero
    if args.cam is None:
        # One camera per PROCESS, not per loop iteration. Building a second
        # nvarguscamerasrc pipeline after tearing the first down inside the same
        # process wedges: cam0 shoots fine and cam1 then sits there until it is
        # killed. Alone, each takes about 8 s. Re-exec rather than debug Argus.
        rc = 0
        for i in (0, 1):
            rc |= subprocess.call([sys.executable, os.path.abspath(__file__),
                                   "--cam", str(i)] + _passthrough(args))
        return rc

    cams = (args.cam,)
    for i in cams:
        key = f"cam{i}"
        c = spec.setdefault(key, {})
        # Sign convention for cam1's yaw is the same as cam0's, so --yaw 12 means
        # 12 deg to STARBOARD on both. Pass a negative for cam1's usual pose.
        for flag, field in (("yaw", "yaw_deg"), ("pitch", "pitch_deg"),
                            ("roll", "roll_deg")):
            v = getattr(args, flag)
            if v is not None:
                c[field] = v
        xyz = list(c.get("xyz_m", [0.0, 0.0, 0.0]))
        for j, flag in enumerate(("dx", "dy", "dz")):
            v = getattr(args, flag)
            if v is not None:
                xyz[j] = v
        c["xyz_m"] = xyz
    rig = fusion.Rig(spec)
    print("rig in use:")
    print(rig.describe())

    reader = lidar_mod.LidarReader(args.lidar_port)
    reader.start()
    try:
        for i in cams:
            overlay(i, args, rig, reader)
    finally:
        reader.shutdown()
        reader.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
