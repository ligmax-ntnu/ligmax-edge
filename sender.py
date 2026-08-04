#!/usr/bin/env python3
"""Dual OV5647 -> YOLO26 detector -> cardinal classifier -> TCP to the viewer.

Runs on the Jetson. For each camera it emits a small preview JPEG plus the
detections, and the receiver draws the boxes.

Geometry, which is the fiddly part:
  sensor mode 0 gives 2592x1944 (4:3, full field of view, 14 fps). The detector
  wants 1280x640 (2:1). Letterboxing 4:3 into 2:1 would waste a third of the input
  on grey bars, and stretching would distort. So we CROP a 2:1 band -- 2048x1024 by
  default, scaled by 1.6 -- with no padding and no distortion, so box coordinates
  map to the preview with a single uniform scale. Full sensor width is deliberately
  NOT used: its left and right edges fall outside the calibration's valid cone,
  where no bearing exists. --crop-w sets the width, --aim-deg swings the window
  toward the pair's overlap, --crop-top picks the band.

The preview JPEG, the detector frame and the full-resolution measurement frame are
all branches of the same buffer, matched by PTS -- boxes can never be drawn on, or
measured against, the wrong frame. Note that src-crop happens exactly ONCE, at a
two-level tee: nvvideoconvert writes its crop rectangle into the shared
NvBufSurface metadata, so two croppers hanging off one tee race and segfault.

With a calibration present, every detection also carries a bearing, a range and
honest uncertainties for both, plus the time its own sensor rows were exposed --
see estimate.py and protocol.py.

Only detections of the cardinal class go through the second-stage classifier, so
its cost scales with how many cardinal marks are actually in view.

The mount is inverted, so the frame is rotated 180 on the VIC before the tee, and
white balance is pinned to daylight rather than left on auto -- see build_pipeline
for why both are done where they are. Detections carry a persistent id from
Tracker, since the exported engine has no tracker of its own.

  ./sender.py --host 192.168.99.135 --port 3338
  ./sender.py --host 192.168.99.135 --preview 640x320 --conf 0.3
  ./sender.py --host 192.168.99.135 --no-rotate --wb auto --no-track
"""
import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time

import gi
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

gi.require_version("Gst", "1.0")
# GstApp must be imported for appsink's try_pull_sample/pull_sample to exist as
# methods; without it parse_launch hands back a plain Gst.Element and the calls
# fail with AttributeError at runtime.
gi.require_version("GstApp", "1.0")
# GstVideo is needed for buffer_get_video_meta: the full-resolution NV12 buffers are
# row-padded to a hardware alignment, and the meta is the only reliable stride.
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstApp, GstVideo  # noqa: E402,F401

import estimate
import protocol
from trt import Engine

SENSOR = {0: (2592, 1944, 14), 1: (1920, 1080, 29), 2: (1296, 972, 28)}

# nvarguscamerasrc's wbmode enum, in its order.
WB_MODES = ["off", "auto", "incandescent", "fluorescent", "warm-fluorescent",
            "daylight", "cloudy-daylight", "twilight", "shade"]


class Sender(threading.Thread):
    """Ships frames to the viewer without ever blocking the inference loop.

    A bounded queue that drops the oldest entry is deliberate: if the network
    stalls or the viewer goes away, the camera pipeline must keep running. Stale
    preview frames are worthless anyway.
    """

    daemon = True

    def __init__(self, host, port, depth=4):
        super().__init__()
        self.host, self.port = host, port
        self.q = queue.Queue(maxsize=depth)
        self.sock = None
        self.stop_flag = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.connected = False

    def submit(self, header, jpeg):
        try:
            self.q.put_nowait((header, jpeg))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait((header, jpeg))
            except queue.Full:
                self.dropped += 1

    def run(self):
        backoff = 0.5
        while not self.stop_flag.is_set():
            if self.sock is None:
                try:
                    s = socket.create_connection((self.host, self.port), timeout=3)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock = s
                    self.connected = True
                    backoff = 0.5
                    print(f"[sender] connected to {self.host}:{self.port}", flush=True)
                except OSError as e:
                    self.connected = False
                    print(f"[sender] connect failed ({e}); retrying in {backoff:.1f}s",
                          flush=True)
                    self.stop_flag.wait(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
            try:
                header, jpeg = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.sock.sendall(protocol.encode(header, jpeg))
                self.sent += 1
            except OSError as e:
                print(f"[sender] send failed ({e}); reconnecting", flush=True)
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
                self.connected = False

    def shutdown(self):
        self.stop_flag.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


def build_pipeline(args, w, h, fps, crops, net_w, net_h, pw, ph):
    """Two cameras, each tee'd into detector, preview-JPEG and full-res branches.

    The tee now sits at FULL sensor resolution, before the crop, because range
    estimation measures the buoy's angular width and the detector input is downscaled
    1.6x -- a box edge there is already worth 1.6 sensor pixels before any other
    error, and range error is proportional to edge error. So a third branch carries
    the uncropped frame to system memory and estimate.py measures on that.

    That branch asks for NV12 rather than RGB deliberately. NV12's first plane IS
    luminance, so the "conversion" is a straight copy the VIC can do, and the Y plane
    is all the edge refinement needs -- colour is only used by the cardinal
    classifier, which works on the detector-resolution crop. Requesting RGB here
    instead would force a full-resolution colour conversion, which measured about
    4 fps in calibrate/calib_server.py and would not fit the 71.4 ms frame budget.
    """
    chunks = []
    # The mount is inverted, so rotate 180. Done here, before the tee, so it costs
    # one VIC pass shared by both branches and the detector sees the same upright
    # frame the viewer does -- boxes then need no coordinate flip anywhere.
    #
    # Not done with the sensor's flip registers: those change the Bayer phase
    # (BGGR -> RGGB) and would silently wreck the demosaic. See ../camera-test.
    #
    # src-crop is in input (pre-flip) pixels, so --crop-top keeps meaning "band
    # measured down the raw sensor frame" whether or not the rotation is on.
    flip = "" if args.no_rotate else " flip-method=2"
    for sid in (0, 1):
        cl, ct, cw, ch = crops[sid]
        # nvjpegenc only accepts video/x-raw(memory:NVMM) -- feeding it system
        # memory corrupts the conversion path and takes CUDA down with it.
        #
        # The rotation happens once here, before the tee, so all three branches
        # share one VIC pass and every branch sees the same upright frame. src-crop
        # is applied per branch AFTER it, in post-rotation coordinates.
        # TWO tees, not one, and src-crop happens exactly once.
        #
        # The obvious structure -- one full-resolution tee with src-crop on each
        # consumer -- segfaults. nvvideoconvert writes its crop rectangle into the
        # shared NvBufSurface metadata, and a tee hands the SAME surface to every
        # branch, so two croppers race on one buffer: "Failed in mem copy" out of
        # nvbufsurftransform, then a core dump. Each element works fine alone, which
        # is what makes it a trap.
        #
        # So: tee a{sid} at full resolution feeds only the measurement branch and the
        # single cropper; tee t{sid} then distributes the already-cropped net-size
        # frame to the detector and the preview, which is the arrangement that was
        # working before this change.
        chunks.append(
            f"nvarguscamerasrc name=cam{sid} sensor-id={sid} sensor-mode={args.mode} "
            f"wbmode={WB_MODES.index(args.wb)} do-timestamp=true "
            f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 "
            f"! nvvideoconvert{flip} "
            f"! video/x-raw(memory:NVMM),width={w},height={h},format=NV12 "
            f"! tee name=a{sid} "
            # measurement branch: full sensor frame, NV12 in system memory. Only the
            # Y plane is read. leaky=downstream so a slow consumer here can never
            # backpressure the detector.
            f"a{sid}. ! queue max-size-buffers=2 leaky=downstream "
            f"! nvvidconv ! video/x-raw,format=NV12,width={w},height={h} "
            f"! appsink name=full{sid} emit-signals=false max-buffers=4 drop=true sync=false "
            # the one and only crop+scale, then the second tee
            f"a{sid}. ! queue max-size-buffers=2 leaky=downstream "
            f"! nvvideoconvert src-crop={cl}:{ct}:{cw}:{ch} "
            f"! video/x-raw(memory:NVMM),width={net_w},height={net_h},format=NV12 "
            f"! tee name=t{sid} "
            # detector branch: RGB in system memory, already at network size.
            # compute-hw=GPU is required: nvvideoconvert defaults to VIC on Jetson
            # and VIC cannot do NV12->RGB, which fails the branch and the pipeline
            # with it ("RGB/BGR Format transformation is not supported by VIC").
            f"t{sid}. ! queue max-size-buffers=2 leaky=downstream "
            f"! nvvideoconvert compute-hw=GPU "
            f"! video/x-raw,format=RGB,width={net_w},height={net_h} "
            f"! appsink name=det{sid} emit-signals=false max-buffers=2 drop=true sync=false "
            # preview branch: downscale the cropped frame, then hardware JPEG. Same
            # crop as the detector, so boxes overlay with one uniform scale.
            f"t{sid}. ! queue max-size-buffers=2 leaky=downstream "
            f"! nvvideoconvert ! video/x-raw(memory:NVMM),width={pw},height={ph},format=I420 "
            f"! nvjpegenc quality={args.quality} "
            f"! appsink name=jpg{sid} emit-signals=false max-buffers=3 drop=true sync=false"
        )
    return " ".join(chunks)


def sample_to_y(sample, w, h):
    """Y (luminance) plane of an NV12 buffer as a h x w uint8 view, plus its PTS.

    Honours the stride from the buffer's video meta: nvvideoconvert pads rows to a
    hardware alignment at these widths, and assuming w would shear the image.
    """
    buf = sample.get_buffer()
    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None, None
    try:
        stride = w
        caps = sample.get_caps()
        meta = None
        try:
            meta = GstVideo.buffer_get_video_meta(buf)
        except Exception:
            meta = None
        if meta is not None and meta.n_planes > 0:
            stride = int(meta.stride[0])
        elif len(mi.data) >= (w + 63) // 64 * 64 * h * 3 // 2:
            stride = (w + 63) // 64 * 64
        need = stride * h
        if len(mi.data) < need:
            return None, None
        y = np.frombuffer(mi.data[:need], dtype=np.uint8).reshape(h, stride)
        return y[:, :w].copy(), buf.pts
    finally:
        buf.unmap(mi)


def sample_to_rgb(sample, w, h):
    buf = sample.get_buffer()
    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None, None
    try:
        # GStreamer rounds each row up to a 4-byte boundary; RGB at these widths is
        # already aligned, but honour the stride from caps rather than assuming.
        stride = w * 3
        need = stride * h
        arr = np.frombuffer(mi.data[:need], dtype=np.uint8).reshape(h, w, 3).copy()
    finally:
        buf.unmap(mi)
    return arr, buf.pts


def sample_to_bytes(sample):
    buf = sample.get_buffer()
    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None, None
    try:
        data = bytes(mi.data)
    finally:
        buf.unmap(mi)
    return data, buf.pts


def classify_crop(rgb, box, engine, size=96):
    """Second stage: resize shortest edge to `size`, centre-crop, /255.

    Mirrors ultralytics classify_transforms, whose default mean/std are a no-op,
    so there is no ImageNet normalisation to apply here.
    """
    h, w = rgb.shape[:2]
    x1 = max(0, min(w - 1, int(box[0])))
    y1 = max(0, min(h - 1, int(box[1])))
    x2 = max(x1 + 1, min(w, int(box[2])))
    y2 = max(y1 + 1, min(h, int(box[3])))
    patch = rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return None, 0.0
    im = Image.fromarray(patch)
    iw, ih = im.size
    s = size / min(iw, ih)
    im = im.resize((max(size, int(round(iw * s))), max(size, int(round(ih * s)))),
                   Image.BILINEAR)
    iw, ih = im.size
    l, t = (iw - size) // 2, (ih - size) // 2
    im = im.crop((l, t, l + size, t + size))
    blob = (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = engine.infer(blob)[0][0]
    # yolo26n-cls emits softmax probabilities already.
    idx = int(np.argmax(out))
    return idx, float(out[idx])


class Tracker:
    """Assigns each buoy an id that persists from frame to frame.

    There is no tracker inside the engine to reuse. Ultralytics' `model.track()`
    is a Python-layer wrapper (ByteTrack / BoT-SORT) around the PyTorch model; it
    is not part of the graph and does not survive an ONNX/TensorRT export, so
    association has to happen here. It is cheap -- a handful of boxes per frame.

    Boxes are predicted forward one frame at their last measured velocity before
    matching. That matters more than it sounds: buoys are small at range, so when
    the boat pitches a box can travel further than its own height between frames,
    and raw IoU for the correct pair is then exactly zero. Hence the second,
    distance-based gate; IoU alone hands out a fresh id every few frames in any
    swell.

    Association is on geometry only, never class. A red buoy that reads "green"
    for one frame is still the same buoy, and gating on class would split it into
    two tracks -- the whole point of the id is to make that flicker visible
    rather than to hide it.
    """

    def __init__(self, max_age, iou_thresh=0.15, dist_gate=2.0):
        self.max_age = max_age          # frames a lost track coasts before it dies
        self.iou_thresh = iou_thresh
        self.dist_gate = dist_gate      # centre travel, in mean box side lengths
        self.tracks = []
        self.next_id = 1

    def _score(self, preds, boxes):
        """Cost matrix plus a mask of which pairs are allowed to match at all."""
        p = np.asarray(preds, dtype=np.float32)
        d = np.asarray(boxes, dtype=np.float32)

        x1 = np.maximum(p[:, None, 0], d[None, :, 0])
        y1 = np.maximum(p[:, None, 1], d[None, :, 1])
        x2 = np.minimum(p[:, None, 2], d[None, :, 2])
        y2 = np.minimum(p[:, None, 3], d[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ap = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
        ad = np.clip(d[:, 2] - d[:, 0], 0, None) * np.clip(d[:, 3] - d[:, 1], 0, None)
        iou = inter / np.maximum(ap[:, None] + ad[None, :] - inter, 1e-6)

        pc = np.stack([(p[:, 0] + p[:, 2]) * 0.5, (p[:, 1] + p[:, 3]) * 0.5], 1)
        dc = np.stack([(d[:, 0] + d[:, 2]) * 0.5, (d[:, 1] + d[:, 3]) * 0.5], 1)
        dist = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=2)
        # Normalised by box size so a 12 px buoy and a 200 px one get gates in
        # proportion to themselves rather than one absolute pixel threshold.
        side = 0.5 * (np.sqrt(ap)[:, None] + np.sqrt(ad)[None, :])
        norm = dist / np.maximum(side, 1e-6)

        # IoU dominates while boxes overlap; distance orders the pairs that do not.
        return (1.0 - iou) + 0.25 * norm, (iou >= self.iou_thresh) | (norm <= self.dist_gate)

    def update(self, boxes):
        """boxes: sequence of (x1, y1, x2, y2). Returns one id per box, in order."""
        for t in self.tracks:
            t["misses"] += 1
        preds = [[t["box"][k] + t["vel"][k] for k in range(4)] for t in self.tracks]

        ids = [0] * len(boxes)
        if preds and len(boxes):
            cost, ok = self._score(preds, boxes)
            for a, b in zip(*linear_sum_assignment(cost)):
                if not ok[a, b]:
                    continue                    # optimal but implausible: leave both free
                t = self.tracks[a]
                new = [float(v) for v in boxes[b]]
                # Velocity is measured-to-measured and smoothed, so one noisy box
                # nudges the prediction instead of redirecting it.
                t["vel"] = [0.5 * t["vel"][k] + 0.5 * (new[k] - t["box"][k])
                            for k in range(4)]
                t["box"] = new
                t["misses"] = 0
                ids[b] = t["id"]

        for b in range(len(boxes)):
            if ids[b] == 0:
                self.tracks.append({"id": self.next_id, "vel": [0.0] * 4,
                                    "box": [float(v) for v in boxes[b]], "misses": 0})
                ids[b] = self.next_id
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_age]
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.99.135")
    ap.add_argument("--port", type=int, default=3338)
    ap.add_argument("--mode", type=int, default=0, choices=sorted(SENSOR),
                    help="sensor mode: 0=2592x1944@14 (default, full FOV), "
                         "1=1920x1080@29, 2=1296x972@28. Modes 3 and 4 do not "
                         "stream on this board.")
    ap.add_argument("--engine", default="best_640x1280_b2_fp16.engine")
    ap.add_argument("--cls-engine", default="best-cls_fp16.engine")
    ap.add_argument("--preview", default="640x320", help="preview WxH per camera")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop-top", type=int, default=None,
                    help="top of the 2:1 band in sensor pixels (default: centred)")
    ap.add_argument("--crop-w", type=int, default=2048,
                    help="width of the 2:1 detector window in sensor pixels; height "
                         "is half of it. Smaller = more pixels per buoy and less "
                         "distortion, at the cost of field. The full 2592 is not a "
                         "sensible choice: its left and right edges fall outside the "
                         "calibration's valid cone, so no bearing exists there.")
    ap.add_argument("--aim-deg", type=float, default=15.0,
                    help="degrees to swing the detector window toward the pair's "
                         "OVERLAP, which is on cam0's right and cam1's left. Applied "
                         "as +aim to cam0 and -aim to cam1. Clamped to whatever the "
                         "crop width allows, and the achieved value is printed.")
    ap.add_argument("--calib", default="calibrate/calib",
                    help="directory holding cam0.json / cam1.json, as written by "
                         "calibrate/calibrate_fisheye.py. Without them there are no "
                         "bearings or ranges, only boxes.")
    ap.add_argument("--calib-names", default="cam0.json,cam1.json")
    ap.add_argument("--no-estimate", action="store_true",
                    help="skip bearing/range estimation and the full-resolution "
                         "branch that feeds it")
    ap.add_argument("--buoy-diameter", type=float, default=0.40,
                    help="buoy diameter in metres; 0.40 for Njord competition marks")
    ap.add_argument("--readout-frac", type=float, default=0.9,
                    help="fraction of the frame period the rolling shutter spends "
                         "reading active rows, used for the per-detection timestamp. "
                         "~0.9 on the OV5647 at full resolution, so the bottom of "
                         "the frame is exposed ~60 ms after the top.")
    ap.add_argument("--no-rotate", action="store_true",
                    help="upright mount; skip the 180 degree rotation")
    ap.add_argument("--wb", default="daylight", choices=WB_MODES,
                    help="Argus white balance. Pinned rather than auto because the "
                         "detector classifies by colour and auto WB shifting hue "
                         "between frames is worse than a consistent small error.")
    ap.add_argument("--no-track", action="store_true",
                    help="do not assign persistent ids to detections")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--max-cardinals", type=int, default=6,
                    help="most cardinal crops to classify per camera per frame. "
                         "The frame budget at mode 0 is 71.4 ms, of which "
                         "preprocessing takes 6.9 and the detector 54.0, so only "
                         "~10 ms is left; each crop costs ~1.1 ms. Highest "
                         "confidence first; the rest are reported unclassified.")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args()

    w, h, fps = SENSOR[args.mode]
    pw, ph = (int(v) for v in args.preview.split("x"))

    det = Engine(args.engine)
    b, _, net_h, net_w = det.in_shape
    if b != 2:
        print(f"detector engine is batch {b}; this app batches the two cameras "
              f"together and needs batch 2", file=sys.stderr)
        return 2
    print(f"detector {args.engine}: in {det.in_shape} out {det.out_shape}")

    cls = None
    if not args.no_classify:
        try:
            cls = Engine(args.cls_engine)
            print(f"classifier {args.cls_engine}: in {cls.in_shape} out {cls.out_shape}")
        except Exception as e:
            print(f"[warn] no cardinal classifier ({e}); continuing without it")

    # Crop a net_w:net_h-shaped window so the scale is uniform and there is no
    # padding. Aspect must match the network exactly: a non-uniform scale would need
    # fx and fy scaled by different factors, and estimate.Camera refuses that rather
    # than silently returning wrong angles.
    crop_w = min(w, max(net_w, args.crop_w))
    crop_h = min(h, int(round(crop_w * net_h / net_w)))
    scale = crop_w / net_w

    cams, crops = [None, None], []
    names = [s.strip() for s in args.calib_names.split(",")]
    for sid in (0, 1):
        model = None
        if not args.no_estimate:
            path = os.path.join(args.calib, names[sid])
            try:
                with open(path, encoding="utf-8") as f:
                    model = json.load(f)
            except OSError as e:
                print(f"[warn] no calibration for cam{sid} ({e}); "
                      f"boxes only, no bearing or range", file=sys.stderr)
        if model is not None and list(model["image_size"]) != [w, h]:
            print(f"[warn] cam{sid} calibration is for {model['image_size']} but the "
                  f"sensor is {[w, h]}; a model does not transfer across modes. "
                  f"Estimation disabled for this camera.", file=sys.stderr)
            model = None
        # cam0 swings toward its right, cam1 toward its left: that is where the pair
        # overlaps, verified by matching the same scene features in both frames.
        aim = args.aim_deg if sid == 0 else -args.aim_deg
        if model is not None:
            cl, ct, got = estimate.crop_for_aim(model, aim, crop_w, crop_h)
        else:
            cl = max(0, min(w - crop_w, (w - crop_w) // 2))
            ct, got = None, None
        if args.crop_top is not None:
            ct = max(0, min(h - crop_h, args.crop_top))
        elif ct is None:
            ct = (h - crop_h) // 2
        crops.append((cl, ct, crop_w, crop_h))
        if model is not None:
            cams[sid] = estimate.Camera(model, cl, ct, crop_w, crop_h, net_w, net_h)
            c = cams[sid]
            y = ct + crop_h / 2.0
            hf = c.angle_between([cl + 0.5, y], [cl + crop_w - 0.5, y])
            vf = c.angle_between([cl + crop_w / 2.0, ct + 0.5],
                                 [cl + crop_w / 2.0, ct + crop_h - 0.5])
            print(f"cam{sid}: crop {crop_w}x{crop_h} at ({cl},{ct}), aim asked "
                  f"{aim:+.1f} got {got:+.1f} deg, covers "
                  f"{math.degrees(hf):.1f}x{math.degrees(vf):.1f} deg, "
                  f"{c.mrad_per_px([cl + crop_w / 2.0, y]):.3f} mrad/px at centre")
        else:
            print(f"cam{sid}: crop {crop_w}x{crop_h} at ({cl},{ct}), no calibration")
    print(f"sensor {w}x{h}@{fps} -> net {net_w}x{net_h} "
          f"(uniform {scale:.3f}x, no letterbox)")
    print(f"preview {pw}x{ph} q{args.quality} -> {args.host}:{args.port}")
    do_est = any(c is not None for c in cams)

    Gst.init(None)
    desc = build_pipeline(args, w, h, fps, crops, net_w, net_h, pw, ph)
    try:
        pipe = Gst.parse_launch(desc)
    except GLib.Error as e:
        print(f"pipeline build failed: {e}\n\n{desc}", file=sys.stderr)
        return 2

    dets = [pipe.get_by_name(f"det{i}") for i in (0, 1)]
    jpgs = [pipe.get_by_name(f"jpg{i}") for i in (0, 1)]
    fulls = [pipe.get_by_name(f"full{i}") for i in (0, 1)]

    tx = Sender(args.host, args.port)
    tx.start()

    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("failed to start pipeline", file=sys.stderr)
        return 2

    # base_time is only valid once the pipeline is running, and it is what turns a
    # buffer PTS into an absolute capture instant. Read it after PLAYING, never before.
    pipe.get_state(Gst.CLOCK_TIME_NONE)
    clock = estimate.CaptureClock(pipe.get_base_time(), fps, args.readout_frac)
    print(f"capture clock: base_time {pipe.get_base_time()} ns, "
          f"rolling-shutter readout {clock.readout_s * 1000:.1f} ms top-to-bottom")

    bus = pipe.get_bus()
    # One tracker per camera: they see different scenes, and ids are per-camera.
    # A lost track coasts for about a second whatever the sensor mode's rate.
    trackers = None if args.no_track else [Tracker(max_age=max(3, int(round(fps))))
                                           for _ in (0, 1)]
    blob = np.empty((2, 3, net_h, net_w), dtype=np.float32)
    jpeg_cache = [{}, {}]
    seq = [0, 0]
    y_cache = [{}, {}]      # full-res Y planes keyed by PTS
    y_stale = [0, 0]        # detector frames with no matching full-res frame
    est_err = [0]
    frames = 0
    t_stats = time.monotonic()
    fps_meas = 0.0
    cards = 0

    print("running; ctrl-c to stop", flush=True)
    try:
        while True:
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"pipeline error: {err.message}\n{dbg}", file=sys.stderr)
                break

            # Drain whatever preview JPEGs are ready, keyed by PTS so each one can
            # be matched to the detector frame it came from.
            for i in (0, 1):
                while True:
                    s = jpgs[i].try_pull_sample(0)
                    if s is None:
                        break
                    data, pts = sample_to_bytes(s)
                    if data is not None:
                        jpeg_cache[i][pts] = data
                if len(jpeg_cache[i]) > 12:
                    for k in sorted(jpeg_cache[i])[:-8]:
                        del jpeg_cache[i][k]

            # Both cameras run at the same rate off the same clock, so waiting for
            # a pair is cheap and lets the detector run as one batch of two.
            samples = []
            for i in (0, 1):
                s = dets[i].try_pull_sample(int(0.5 * Gst.SECOND))
                if s is None:
                    samples = []
                    break
                samples.append(s)
            if len(samples) != 2:
                continue

            rgbs, ptss = [], []
            for i, s in enumerate(samples):
                arr, pts = sample_to_rgb(s, net_w, net_h)
                if arr is None:
                    break
                rgbs.append(arr)
                ptss.append(pts)
            if len(rgbs) != 2:
                continue

            # Full-resolution Y plane matched to the detector frame BY PTS, not by
            # "whichever arrived last". The measurement branch has no inference in it
            # so it runs a frame or two ahead; taking the newest would measure the
            # buoy on a different frame than the one that detected it, which for a
            # moving target is a silent bearing error. Cache a few and look up the
            # exact PTS, exactly as jpeg_cache already does for the preview.
            ys = [None, None]
            if do_est:
                for i in (0, 1):
                    while True:
                        s = fulls[i].try_pull_sample(0)
                        if s is None:
                            break
                        y_arr, y_pts = sample_to_y(s, w, h)
                        if y_arr is not None:
                            y_cache[i][y_pts] = y_arr
                    # Full frames are 5 MB each, so keep the window short.
                    if len(y_cache[i]) > 4:
                        for k in sorted(y_cache[i])[:-3]:
                            del y_cache[i][k]
                    ys[i] = y_cache[i].pop(ptss[i], None)
                    if ys[i] is None:
                        y_stale[i] += 1

            for i in (0, 1):
                # Fused multiply into a preallocated buffer: the obvious
                # astype()/255 form measured 17 ms per camera here, this one 3.5 ms.
                np.multiply(rgbs[i].transpose(2, 0, 1), np.float32(1.0 / 255.0),
                            out=blob[i])

            out = det.infer(blob)[0]        # (2, 300, 6)
            now = time.time()
            frames += 1

            for i in (0, 1):
                rows = out[i]
                keep = rows[rows[:, 4] >= args.conf]
                ids = (trackers[i].update(keep[:, :4]) if trackers
                       else [None] * len(keep))
                items = []
                # Rows arrive score-sorted from the end-to-end head, so taking the
                # first N cardinals is the same as taking the most confident ones.
                budget = args.max_cardinals
                for n, (x1, y1, x2, y2, score, c) in enumerate(keep):
                    c = int(round(float(c)))
                    d = {
                        "id": ids[n],
                        "cls": c,
                        "name": protocol.CLASS_NAMES.get(c, str(c)),
                        "conf": round(float(score), 4),
                        "box": [round(float(x1), 1), round(float(y1), 1),
                                round(float(x2), 1), round(float(y2), 1)],
                        "card": None,
                        "card_conf": None,
                    }
                    if (cls is not None and c == protocol.CARDINAL_CLASS_ID
                            and budget > 0):
                        budget -= 1
                        idx, p = classify_crop(rgbs[i], (x1, y1, x2, y2), cls)
                        if idx is not None:
                            d["card"] = protocol.CARDINAL_NAMES.get(idx, str(idx))
                            d["card_conf"] = round(p, 4)
                            cards += 1
                    if cams[i] is not None:
                        try:
                            d.update(estimate.estimate(
                                cams[i], (x1, y1, x2, y2), gray_full=ys[i],
                                clock=clock, pts_ns=ptss[i],
                                diameter_m=args.buoy_diameter))
                        except Exception as e:      # never let geometry kill a frame
                            est_err[0] += 1
                            d["estimate_error"] = str(e)[:120]
                    items.append(d)

                jpeg = jpeg_cache[i].pop(ptss[i], None)
                if jpeg is None and jpeg_cache[i]:
                    # Fall back to the newest preview if the exact PTS is missing;
                    # at most one frame of skew, and only when a branch hiccups.
                    jpeg = jpeg_cache[i].pop(max(jpeg_cache[i]), None)
                if jpeg is None:
                    continue

                seq[i] += 1
                # ts is now the CAPTURE instant from the buffer PTS, not time.time()
                # after inference. t_sent is kept separately so the pipeline latency
                # stays visible instead of being folded into the measurement.
                t_cap = clock.frame_time(ptss[i])
                tx.submit({
                    "cam": i, "seq": seq[i],
                    "ts": round(t_cap, 6),
                    "t_capture": round(t_cap, 6),
                    "t_sent": round(now, 6),
                    "latency_ms": round((now - t_cap) * 1000.0, 2),
                    "readout_ms": round(clock.readout_s * 1000.0, 2),
                    "net_w": net_w, "net_h": net_h,
                    "jpeg_w": pw, "jpeg_h": ph,
                    "crop": list(crops[i]),
                    "full_w": w, "full_h": h,
                    "refined": ys[i] is not None,
                    "fps": round(fps_meas, 2),
                    "dets": items,
                }, jpeg)

            el = time.monotonic() - t_stats
            if el >= args.stats_every:
                fps_meas = frames / el
                print(f"[{time.strftime('%H:%M:%S')}] {fps_meas:5.2f} fps/cam  "
                      f"sent={tx.sent} dropped={tx.dropped} "
                      f"cardinals={cards} "
                      f"stale_full={y_stale[0]}/{y_stale[1]} "
                      f"est_err={est_err[0]} "
                      f"link={'up' if tx.connected else 'DOWN'}", flush=True)
                frames, cards = 0, 0
                t_stats = time.monotonic()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        # Drain via EOS: tearing the pipeline down mid-capture leaves Argus in a
        # latched error state that survives a reboot and needs a power cycle.
        pipe.send_event(Gst.Event.new_eos())
        bus.timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.EOS)
        pipe.set_state(Gst.State.NULL)
        pipe.get_state(3 * Gst.SECOND)
        tx.shutdown()
        det.close()
        if cls is not None:
            cls.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
