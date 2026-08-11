#!/usr/bin/env python3
"""Dual OV5647 -> YOLO26 detector -> cardinal classifier -> TCP to the Pi.

Runs on the Jetson. For each camera it emits a small preview JPEG plus the
detections, and the receiver draws the boxes.

Two destinations, on purpose
----------------------------
  detections + preview  --TCP :3401--> ligmax-pi3.local
      The Pi fuses these with the aft lidar and sends ONE world model up the
      telemetry link, so the operator's map cannot show two disagreeing versions
      of the same buoy. `receiver.py` also still accepts this stream directly,
      which is what you point at during bench work.

  preview JPEG          --HTTPS-----> live.ligmax.no /api/camera
      Straight to shore, small and off by default, so the dashboard can show a
      picture without one existing on the 4G uplink at all times. See
      `cloud_camera.py`; `--no-cloud` disables it outright.

Note the port: :3401, not :3338. The dashboard binds 3338 and `live.ligmax.no` is
forwarded there, so the edge feed moved off it (docs/findings.md item 1).

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

THIS PROCESS OWNS BOTH SENSORS, which is why two features that look unrelated to
a detector live in here. Argus hands a CSI sensor to one consumer, so nothing else
on this board can open cam0 or cam1 while this is running -- there is no "just run
a little capture script alongside it". So:

  * **full-resolution stills** are taken here, on request, and handed to
    `cloud_camera` to upload (`--no-cloud` disables them along with the uplink).
    The whole 2592x1944 frame, not the detector's crop: the crop is a 2:1 band
    swung off each lens's axis and the AR tags on a dock are not in it.
  * **the detector can be turned off**, two independent ways. `--no-detect` never
    loads an engine at all, so this runs as a plain camera on a board where the
    engines have not been rebuilt; and the dashboard can switch inference off and
    on at runtime, without restarting capture -- which matters because tearing
    capture down is what latches Argus into the state only a power cycle clears.
    Either way the pipeline, the previews, the stills and the lidar all keep
    running; only the inference stops.

  ./sender.py                                       # -> ligmax-pi3.local:3401
  ./sender.py --host 192.168.99.135 --port 3401     # -> a viewer, for bench work
  ./sender.py --no-cloud                            # no dashboard uplink at all
  ./sender.py --no-detect                           # cameras only, no engine
  ./sender.py --preview 640x320 --conf 0.3
  ./sender.py --no-rotate --wb auto --no-track
"""
import argparse
import collections
import json
import math
import os
import queue
import signal
import socket
import sys
import threading
import time

import gi
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from cloud_camera import CameraUplink

gi.require_version("Gst", "1.0")
# GstApp must be imported for appsink's try_pull_sample/pull_sample to exist as
# methods; without it parse_launch hands back a plain Gst.Element and the calls
# fail with AttributeError at runtime.
gi.require_version("GstApp", "1.0")
# GstVideo is needed for buffer_get_video_meta: the full-resolution NV12 buffers are
# row-padded to a hardware alignment, and the meta is the only reliable stride.
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstApp, GstVideo  # noqa: E402,F401

import artags
import estimate
import fusion
import lidar as lidar_mod
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
            f"wbmode={WB_MODES.index(args.wb)} saturation={args.saturation} "
            f"do-timestamp=true "
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
            # preview branch: downscale, then hardware JPEG.
            #
            # Off the CROPPED tee by default, so the preview is framed exactly like
            # the detector and a box overlays with one uniform scale.
            #
            # Off the FULL tee under --preview-full, because the docking task needs
            # the opposite thing: the AR tags are ~75 deg off each lens's axis and
            # the 2:1 detector band does not contain them, so a preview of the crop
            # is a live picture of the wrong part of the lens. This branch adds no
            # src-crop of its own -- the segfault described above is two CROPPERS
            # racing on one tee'd surface, and a plain scale is not a second
            # cropper -- but it is a change to a pipeline that is known to be
            # delicate on this board, so it is opt-in and the run falls back by
            # dropping the flag.
            + (f"a{sid}. ! queue max-size-buffers=2 leaky=downstream "
               if args.preview_full else
               f"t{sid}. ! queue max-size-buffers=2 leaky=downstream ")
            + f"! nvvideoconvert ! video/x-raw(memory:NVMM),width={pw},height={ph},format=I420 "
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


def sample_to_nv12(sample, w, h):
    """A whole NV12 buffer as UNPADDED bytes -- Y then interleaved UV -- plus PTS.

    `sample_to_y` above takes the luma plane and is what every measurement uses.
    This takes the colour as well, and is only called when a full-resolution still
    has been asked for, because it copies 7.5 MB rather than 5.

    The padding is stripped here rather than passed on with a stride, so the far
    end needs to know nothing about the VIC's row alignment: what comes out is
    exactly `w*h + w*h/2` bytes. Both plane strides and both plane offsets come
    from the video meta -- assuming the UV plane starts right after the Y plane is
    true for the buffers this pipeline produces and is not true in general, and a
    wrong offset would give a correct-looking picture with wrong colour.
    """
    buf = sample.get_buffer()
    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None, None
    try:
        stride_y = stride_uv = w
        off_y, off_uv = 0, None
        try:
            meta = GstVideo.buffer_get_video_meta(buf)
        except Exception:
            meta = None
        if meta is not None and meta.n_planes >= 2:
            stride_y, stride_uv = int(meta.stride[0]), int(meta.stride[1])
            off_y, off_uv = int(meta.offset[0]), int(meta.offset[1])
        if off_uv is None:
            # No meta. Fall back to the same 64-byte alignment guess sample_to_y
            # makes, with the planes contiguous.
            aligned = (w + 63) // 64 * 64
            if len(mi.data) >= aligned * h * 3 // 2:
                stride_y = stride_uv = aligned
            off_y, off_uv = 0, stride_y * h
        need = max(off_y + stride_y * h, off_uv + stride_uv * (h // 2))
        if len(mi.data) < need or h % 2 or w % 2:
            return None, None
        y = np.frombuffer(
            mi.data[off_y:off_y + stride_y * h], dtype=np.uint8
        ).reshape(h, stride_y)
        uv = np.frombuffer(
            mi.data[off_uv:off_uv + stride_uv * (h // 2)], dtype=np.uint8
        ).reshape(h // 2, stride_uv)
        out = np.empty(w * h * 3 // 2, dtype=np.uint8)
        out[:w * h].reshape(h, w)[:] = y[:, :w]
        out[w * h:].reshape(h // 2, w)[:] = uv[:, :w]
        return out.tobytes(), buf.pts
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


def _cloud_line(cloud):
    """One compact field for the stats line: is video going to shore, and how much.

    Deliberately says `off` rather than nothing when the operator has not asked for
    video, so the log distinguishes "nobody wants it" from "it is broken".

    `off` and `UNREACHABLE` are the two halves of that: the first means the poll
    is being answered and the answer is no, the second means the dashboard has
    never answered one at all, which is what a missing LIGMAX_BOAT_KEY or a
    Cloudflare 1010 looks like from here. They are indistinguishable on the
    dashboard, so this line is where you tell them apart.
    """
    stats = cloud.stats()
    if not stats["config_ok"]:
        return f"UNREACHABLE({stats['last_error'] or 'connecting'})"
    # Stills ride the same uplink and are worth saying even while the video
    # stream is off, which is the normal state for taking one: without this the
    # line reads a bare `off` while a 2 MB upload is in flight.
    still = ""
    if stats["capturing"]:
        still = "+capturing"
    elif stats["stills"] or stats["stills_failed"]:
        still = f"+{stats['stills']}still"
        if stats["stills_failed"]:
            still += f"/{stats['stills_failed']}fail"
    if not stats["enabled"]:
        return f"off{still}"
    parts = [f"{stats['sent']}sent{still}"]
    if stats["dropped"]:
        parts.append(f"{stats['dropped']}drop")
    if stats["errors"]:
        parts.append(f"{stats['errors']}err")
    parts.append(stats["asked"])
    return "/".join(parts)


def _lidar_line(reader, lit, stale, self_n, skew_ms, fuse_ms):
    """Points, how many the cameras could colour, and the frame-to-sweep skew.

    `skew` is the one to watch: it is how far apart in capture time the sweep and
    the frame that coloured it were, and it is the number that goes wrong quietly
    if the sweep buffer is ever too short for the camera pipeline's latency.

    `lit` should now sit near the share of a rotation the two lenses cover, and
    stay there: with a frame buffer the per-point choice no longer rises and falls
    with `skew` the way it did on one frame. The parenthesised count is how many
    of those were coloured from outside --lidar-max-skew. A few is the buffer
    doing its job at the edges of the rotation; most of them means the cameras
    and the scanner have drifted apart far enough to be worth a look.

    `self` is how many returns rig.json's `self_box` took off the boat itself. It
    should be a steady couple of dozen -- the deck does not move. A drop to zero
    on a rig that was reporting some means the mask and the world have parted
    company (the lidar's yaw, or the box), and that is worth more than it looks:
    it is also the shape of the deck being shipped as obstacles again.
    """
    st = reader.stats()
    if not st["healthy"]:
        return f"DOWN({st['last_error'] or 'connecting'})"
    return (f"{st['points']}pts/{st['hz']:.1f}Hz lit={lit}"
            + (f"({stale} stale)" if stale else "")
            + (f" self={self_n}" if self_n else "")
            + f" skew={skew_ms:.0f}ms fuse={fuse_ms:.1f}ms"
            + (f" err={st['errors']}" if st["errors"] else ""))


def _parse_self_box_arg(text):
    """"HALFW,FRONT,BACK" -> what fusion.Rig.self_box wants, or raise.

    Raises rather than falling back to the rig file: a typo'd override that
    silently ran the default would look exactly like the override working.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--lidar-self-box wants HALFW,FRONT,BACK, got {text!r}")
    hw, front, back = parts
    return fusion.parse_self_box({
        "half_width_m": float(hw),
        "front_m": float(front),
        "back_m": None if back.lower() in ("", "none", "null") else float(back),
    })


def main():
    ap = argparse.ArgumentParser()
    # The Pi, which fuses these detections with the aft lidar before anything
    # reaches shore. Port 3401 rather than 3338: the dashboard owns 3338 and
    # live.ligmax.no is forwarded to it (docs/findings.md item 1).
    ap.add_argument("--host", default=os.environ.get("LIGMAX_FUSION_HOST",
                                                     "ligmax-pi3.local"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("LIGMAX_FUSION_PORT", "3401")))
    # The dashboard uplink. On by default only in the sense that it *connects*:
    # the server says video is off until an operator asks, so nothing is sent.
    ap.add_argument("--no-cloud", action="store_true",
                    help="do not offer preview frames to the dashboard at all")
    ap.add_argument("--cloud-url", default=None,
                    help="dashboard root; default LIGMAX_UPLOAD_URL or live.ligmax.no")
    ap.add_argument("--no-cloud-boxes", action="store_true",
                    help="send the dashboard a clean picture, without the "
                         "detector's boxes burned into it. The overlay is drawn "
                         "in the uplink's own thread at the stream's frame rate, "
                         "so this buys no detector headroom -- it is for judging "
                         "the camera itself, or the lens, without the detector's "
                         "opinion drawn on top.")
    ap.add_argument("--mode", type=int, default=0, choices=sorted(SENSOR),
                    help="sensor mode: 0=2592x1944@14 (default, full FOV), "
                         "1=1920x1080@29, 2=1296x972@28. Modes 3 and 4 do not "
                         "stream on this board.")
    ap.add_argument("--engine", default="best_640x1280_b2_fp16.engine")
    ap.add_argument("--cls-engine", default="best-cls_fp16.engine")
    ap.add_argument("--no-detect", action="store_true",
                    help="do not load the YOLO engine at all: capture, previews, "
                         "full-resolution stills, bearings and the lidar all run, "
                         "and nothing is inferred. This is the mode for using the "
                         "cameras as cameras -- photographing the AR tags on the "
                         "dock, checking a lens, running on a board whose engines "
                         "have not been rebuilt after a pull. THIS PROCESS OWNS "
                         "BOTH SENSORS (Argus gives a CSI camera to one consumer), "
                         "so a separate capture script is not an alternative. The "
                         "dashboard can also switch inference off and on at "
                         "runtime; this flag is the stronger form, and no remote "
                         "toggle can turn it back on in this process.")
    ap.add_argument("--net", default="1280x640",
                    help="detector input size to assume when there is no engine "
                         "to ask (--no-detect). Sets the crop's aspect and the "
                         "preview geometry, so leaving it at the real detector's "
                         "2:1 keeps a no-detect run framed exactly like a normal "
                         "one. Ignored when an engine is loaded -- that always "
                         "wins, since a mismatch would misplace every box.")
    ap.add_argument("--preview", default="640x320", help="preview WxH per camera")
    ap.add_argument("--preview-full", action="store_true",
                    help="make the preview JPEG the WHOLE sensor frame instead of "
                         "the detector's 2:1 band. This is what the docking page "
                         "needs: the AR tags sit ~75 deg off each lens's axis, "
                         "outside the crop, so the ordinary preview is a live "
                         "picture of the wrong part of the lens. The height is "
                         "re-derived from --preview's width to keep 4:3. Detector "
                         "boxes still overlay correctly (the header carries `crop` "
                         "and `preview_source`), but this moves the preview onto "
                         "the full-resolution tee, and this pipeline has a known "
                         "segfault involving tees -- see build_pipeline. Opt-in for "
                         "that reason; drop the flag to get the old framing back.")
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
    ap.add_argument("--artags", action="store_true",
                    help="find the dock's AR tags (NJORD §9.3) in the FULL sensor "
                         "frame and ship their boat-relative geometry as `tags`. "
                         "This is the docking task's only sensor now that both "
                         "lidars are down: three 18 cm tags mark the assigned "
                         "berth, and ligmax-pi builds the berth out of them. Needs "
                         "a calibration and rig.json, and needs cv2.aruco in the "
                         "OpenCV on this board -- all three are checked at start-up "
                         "and reported rather than discovered mid-run. Costs one "
                         "detectMarkers over about a third of each frame; runs "
                         "happily with --no-detect and --no-lidar, which is the "
                         "docking configuration.")
    ap.add_argument("--artag-size", type=float, default=artags.TAG_M,
                    help="the tags' BLACK SQUARE edge in metres (default 0.18, per "
                         "the handbook). Every range scales linearly with this, so "
                         "measure the print rather than trusting the printer: A4 at "
                         "'fit to page' comes out a few per cent small.")
    ap.add_argument("--artag-dict", default=artags.DICT_NAME,
                    help="ArUco dictionary. The organisers' own files decode as ids "
                         "0-7 in DICT_4X4_50, which is the smallest that holds them "
                         "and so the most false-positive-resistant.")
    ap.add_argument("--artag-window", type=float, default=artags.WINDOW_DEG,
                    help="half-width in degrees of the forward arc searched for "
                         "tags. The tags are on a dock ahead; cropping to that is "
                         "most of what makes this affordable on a 5 Mpx frame.")
    ap.add_argument("--artag-min-px", type=float, default=artags.MIN_EDGE_PX,
                    help="smallest tag edge in pixels worth reporting. An 18 cm tag "
                         "is ~53 px at 3 m and ~20 px at 8 m on this rig.")
    ap.add_argument("--artag-every", type=int, default=1,
                    help="run the tag search every Nth frame. 1 (the default) is "
                         "every frame, which is what has been measured: ~80 ms per "
                         "camera on a cluttered scene, so ~160 ms a frame pair and "
                         "capture drops to 6 fps or so. That is ample for a 0.3 m/s "
                         "creep, and the pipeline drops frames rather than blocking, "
                         "so it costs rate and not stability. Raise this to 2 or 3 "
                         "if the frame rate matters more than the tag rate -- it is "
                         "the cheapest lever there is.")
    ap.add_argument("--artag-bench", action="store_true",
                    help="print what the two cameras disagree by about every tag "
                         "they can both see. rig.json's camera yaws are hand-"
                         "described and UNVERIFIED, and they are what holds one "
                         "camera's half of a berth in register with the other's; "
                         "the pair overlaps ~24 deg across the bow, so a tag placed "
                         "dead ahead is measured twice and the difference is the "
                         "yaw error. Bench use -- it prints every frame.")
    ap.add_argument("--readout-frac", type=float, default=0.9,
                    help="fraction of the frame period the rolling shutter spends "
                         "reading active rows, used for the per-detection timestamp. "
                         "~0.9 on the OV5647 at full resolution, so the bottom of "
                         "the frame is exposed ~60 ms after the top.")
    ap.add_argument("--no-lidar", action="store_true",
                    help="do not open the RPLidar C1 at all: no point cloud on the "
                         "wire and no lidar range on any detection")
    ap.add_argument("--lidar-port", default=None,
                    help="override config.lidar_port() (the udev symlink)")
    ap.add_argument("--rig", default="rig.json",
                    help="hand-measured lidar/camera mounting geometry. Without it "
                         "there is no way to say where a return lands in an image, "
                         "so the sweep ships uncoloured and detections get no range.")
    ap.add_argument("--lidar-max-skew", type=float, default=40.0,
                    help="milliseconds a return and the frame colouring it may "
                         "differ in capture time and still count as TIMELY. Not a "
                         "switch: points outside it are still coloured, from the "
                         "closest frame there is, but are counted in `stale` and "
                         "carry their own `age_ms` so a consumer can tell.")
    ap.add_argument("--lidar-frame-history", type=int, default=2,
                    help="detector frames buffered per camera, on top of the "
                         "current one, for colouring only. A rotation is 100 ms "
                         "and a frame is an instant, so one frame is never near "
                         "all of a sweep; 2 (three frames total, ~100 ms apart) "
                         "covers a rotation either side. 0 restores the old "
                         "one-frame behaviour. Costs ~2.4 MB per frame per camera.")
    ap.add_argument("--lidar-keep-unseen", action="store_true",
                    help="ship returns no camera could see, uncoloured, instead "
                         "of dropping them. They are real obstacles -- the ~34 deg "
                         "aft wedge outside both lenses on this rig -- so this is "
                         "the safe setting if nothing else watches behind. The "
                         "default drops them: it shortens the box tests, the eight "
                         "columnar arrays and the wire, and keeps grey dots off "
                         "the plot, on the assumption the aft lidar has that arc.")
    ap.add_argument("--no-lidar-self-box", action="store_true",
                    help="ship the returns that came off the boat itself instead "
                         "of masking them. The default discards everything inside "
                         "rig.json's `self_box` -- a corridor 0.70 m either side "
                         "of the centreline running aft from the lidar -- because "
                         "a 360 deg scanner on the bow sees its own deck, and a "
                         "hull return inside a detection box is the nearest one "
                         "in it, so it takes the range. Use this to SEE what the "
                         "mask is eating before changing the numbers.")
    ap.add_argument("--lidar-self-box", default=None, metavar="HALFW,FRONT,BACK",
                    help="override rig.json's self_box for one run, in metres in "
                         "the rig frame: half-width either side of the centreline, "
                         "the forward edge, and the aft edge (empty or 'none' for "
                         "no aft edge). E.g. 0.9,0.0,none")
    ap.add_argument("--lidar-max-age", type=float, default=250.0,
                    help="milliseconds past which a frame is too old to colour "
                         "from at all, and the point ships uncoloured. This is a "
                         "STALL GUARD, not the quality gate -- buffered frames sit "
                         "~100 ms apart so it never bites in normal running, only "
                         "when a camera has stopped producing and the buffer has "
                         "gone stagnant. 0 for no bound beyond the buffer itself.")
    ap.add_argument("--no-rotate", action="store_true",
                    help="upright mount; skip the 180 degree rotation")
    ap.add_argument("--saturation", type=float, default=2.0,
                    help="Argus chroma gain, 0-2, applied in the ISP. Defaults to "
                         "the maximum, and that is deliberate: JetPack ships no "
                         "ISP tuning for the OV5647, so Argus returns near "
                         "sensor-native RGB at about a third of the chroma a "
                         "normal camera gives -- and the detector was trained on "
                         "normal cameras, so washed-out is OFF-DISTRIBUTION for "
                         "it. This is the only colour knob upstream of inference "
                         "and it is free (the ISP is already converting), whereas "
                         "the matrix in fusion.py is affordable for 400 lidar "
                         "points but not for 819k detector pixels. Measured over "
                         "real frames: chroma 30.1 -> 59.9, clipping 0.0 -> 2.8%%. "
                         "Set 1.0 to restore the old sensor-native behaviour; "
                         "fusion picks up the rest of the correction either way.")
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

    # Argus rejects the whole pipeline for an out-of-range saturation rather than
    # clamping, so a typo here would read as "the cameras are dead" 40 s later.
    args.saturation = max(0.0, min(2.0, args.saturation))
    # What is left for the matrix to do once the ISP has boosted chroma. Computed
    # ONCE and carried on the wire, so no consumer has to guess how much
    # correction is already baked into the pixels it was handed.
    ccm_strength = fusion.ccm_strength_for(args.saturation)
    print(f"colour: ISP saturation {args.saturation:.2f}, "
          f"OV5647 matrix at {ccm_strength:.2f} strength "
          f"({'sensor-native' if args.saturation == 1.0 else 'ISP-boosted'} pixels "
          f"into the detector)")

    # SIGTERM has to land in the same place Ctrl-C does. Python's default for it
    # exits the interpreter outright, so the `finally` below never runs, the
    # pipeline is never drained with EOS, and Argus latches into the error state
    # that only a power cycle clears (README). That is not a theoretical path: it
    # is what `systemctl restart`, `systemctl stop`, and the dashboard's Update
    # button all do -- update.py SIGTERMs the process group before it pulls. So
    # turn it into the KeyboardInterrupt the shutdown code already handles.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    w, h, fps = SENSOR[args.mode]
    pw, ph = (int(v) for v in args.preview.split("x"))
    if args.preview_full:
        # --preview lives at the detector's 2:1, and the full sensor frame is 4:3.
        # Keeping the requested WIDTH and re-deriving the height is what stops a
        # full-frame preview arriving squashed; the width is what the 4G uplink and
        # the dashboard's panel actually care about.
        ph = 2 * int(round(0.5 * pw * h / w))
        print(f"preview is the WHOLE sensor frame, {pw}x{ph} "
              f"(--preview-full): the AR tags are outside the detector's band")

    # The detector's input size decides the crop's aspect and the preview's, so it
    # has to be known before the pipeline is built. With an engine it comes from
    # the engine, always -- a mismatch there would misplace every box. Without
    # one, --net stands in, defaulting to the real detector's 2:1 so a no-detect
    # run is framed exactly like a normal one.
    det = None
    try:
        net_w, net_h = (int(v) for v in args.net.lower().split("x"))
    except ValueError:
        print(f"--net wants WxH, got {args.net!r}", file=sys.stderr)
        return 2
    if args.no_detect:
        print(f"detector OFF (--no-detect): no engine loaded, "
              f"net geometry {net_w}x{net_h} from --net")
    else:
        det = Engine(args.engine)
        b, _, net_h, net_w = det.in_shape
        if b != 2:
            print(f"detector engine is batch {b}; this app batches the two cameras "
                  f"together and needs batch 2", file=sys.stderr)
            return 2
        print(f"detector {args.engine}: in {det.in_shape} out {det.out_shape}")

    cls = None
    if det is not None and not args.no_classify:
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
        # A calibration is only valid for the image ORIENTATION it was captured in.
        # Get this wrong and nothing fails: the fit is still good, the projection
        # still lands inside the frame, and every lidar return is simply coloured
        # from the pixel 180 deg opposite the one it should read -- a green light
        # off the starboard bow tints returns off the port quarter. There is no
        # geometric check that catches it, which is why it is asserted here.
        rotated = model.get("rotated_180") if model is not None else None
        if model is not None:
            if rotated is None:
                print(f"[warn] cam{sid} calibration does not record whether it was "
                      f"captured with the 180 deg rotation on; it predates that "
                      f"field. This run has rotation "
                      f"{'ON' if not args.no_rotate else 'OFF'} -- if colour lands "
                      f"opposite where it belongs, that mismatch is the reason "
                      f"(try --no-rotate). Recalibrate to record it.",
                      file=sys.stderr)
            elif bool(rotated) != (not args.no_rotate):
                print(f"[warn] cam{sid} calibration was captured with rotation "
                      f"{'ON' if rotated else 'OFF'} but this run has it "
                      f"{'ON' if not args.no_rotate else 'OFF'}. Colour and bearing "
                      f"will be 180 deg out. Pass "
                      f"{'--no-rotate' if rotated is False else 'no --no-rotate'} "
                      f"or recalibrate.", file=sys.stderr)
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

    # The lidar. Its geometry is a separate, HAND-MEASURED thing from the fisheye
    # intrinsics: rig.json says where the sensors are relative to each other, the
    # cam*.json fits say what each lens does. Both are needed to put a return on a
    # pixel, so a missing rig file disables colour but not capture.
    # Loaded for the tags as well as the lidar, and therefore NOT behind
    # --no-lidar: with both lidars down, the docking configuration is
    # `--artags --no-lidar --no-detect`, and that run needs the camera poses more
    # than any other. rig.json holds both sets of geometry and only the lidar half
    # of it is the known-stale one.
    rig, reader = None, None
    if not args.no_lidar or args.artags:
        try:
            rig = fusion.Rig.load(args.rig)
            if args.lidar_self_box:
                # Parsed here rather than in Rig so the file stays the one place
                # the geometry LIVES, and this stays visibly a one-run override.
                rig.self_box = _parse_self_box_arg(args.lidar_self_box)
            print(f"rig {args.rig}:\n{rig.describe()}")
        except OSError as e:
            print(f"[warn] no rig geometry ({e}); the sweep will ship uncoloured "
                  f"and detections get no lidar range", file=sys.stderr)
    if not args.no_lidar:
        reader = lidar_mod.LidarReader(args.lidar_port)
        reader.start()      # connects on its own thread; never blocks capture

    # The dock's AR tags. Constructed up front and loudly, because every way this
    # can fail is a start-up fact -- no cv2.aruco in this OpenCV, no calibration,
    # no rig -- and the alternative is finding out on the water that the one
    # sensor the docking task has was never running.
    finder = None
    if args.artags:
        try:
            finder = artags.TagFinder(
                cams, rig,
                tag_m=args.artag_size, dict_name=args.artag_dict,
                window_deg=args.artag_window, min_edge_px=args.artag_min_px,
            )
            print(f"AR tags on:\n{finder.describe()}")
        except artags.TagError as e:
            print(f"[warn] AR tags OFF: {e}", file=sys.stderr)
            print("[warn] the docking task has no other sensor -- fix this before "
                  "the run", file=sys.stderr)
    # Tags are measured on the FULL sensor frame, so they need the same
    # full-resolution branch the bearings do -- and they need it even under
    # --no-estimate, which is why this is its own flag and not folded into do_est.
    want_tags = finder is not None

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

    # The dashboard uplink. Constructed even when video is off, because it is what
    # polls to find out whether an operator has switched it on - and because
    # starting it later would mean restarting the detector, which means tearing
    # down capture, which Argus does not forgive.
    cloud = None
    if not args.no_cloud:
        boxes = not args.no_cloud_boxes
        cloud = (CameraUplink(args.cloud_url, os.environ.get("LIGMAX_BOAT_KEY"),
                              draw_boxes=boxes)
                 if args.cloud_url else CameraUplink.from_env(draw_boxes=boxes))
        print(f"dashboard uplink -> {cloud.scheme}://{cloud.host}:{cloud.port}"
              f" ({'boxes burned in' if cloud.draw_boxes else 'clean picture'}, "
              f"off until an operator asks for it)")

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
    last_sweep = 0          # sweep already on the wire; frames outrun sweeps slightly
    lidar_seq = 0
    lidar_skew = 0.0        # ms, |sweep t_mid - frame t_capture|, last fused
    lidar_lit = 0           # points of the last sweep that got a colour
    lidar_stale = 0         # of those, coloured from outside --lidar-max-skew
    lidar_self = 0          # points of the last sweep masked off as our own hull
    # The last few detector frames per camera, newest first, for colouring. A
    # rotation is 100 ms and a frame is an instant, so one frame cannot be close
    # in time to a whole sweep; a couple of older ones let each return be coloured
    # from the frame exposed nearest to it. sample_to_rgb already copies out of
    # the GStreamer buffer, so keeping these across frames is safe.
    frame_hist = [collections.deque(maxlen=max(0, args.lidar_frame_history)),
                  collections.deque(maxlen=max(0, args.lidar_frame_history))]
    fuse_ms = 0.0           # time in the fusion block, summed over the window
    fuse_n = 0
    tag_ms = 0.0            # and in the AR-tag block, the docking task's sensor
    tag_n = 0
    last_tags = [[], []]    # re-sent on a frame the tag search skipped
    stills = 0              # full-resolution frames handed to the uplink
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
            # Has an operator asked for a full-resolution still? Read once per
            # frame, because it decides whether the branch below keeps colour as
            # well as luma. Cheap: a couple of comparisons on a dict the uplink
            # thread owns.
            want_still = cloud.wanted_still() if cloud is not None else None

            ys = [None, None]
            nv12s = [None, None]
            if do_est or want_tags or want_still is not None:
                for i in (0, 1):
                    keep = (want_still is not None
                            and str(i) in want_still["cameras"])
                    newest = None
                    while True:
                        s = fulls[i].try_pull_sample(0)
                        if s is None:
                            break
                        y_arr, y_pts = sample_to_y(s, w, h)
                        if y_arr is not None:
                            y_cache[i][y_pts] = y_arr
                        if keep:
                            # Hold the sample, do not convert yet: the loop is
                            # draining a few buffers and only the last of them is
                            # worth 7.5 MB of copying. A Gst.Sample owns its
                            # buffer, so keeping one across further pulls is safe.
                            newest = s
                    # Full frames are 5 MB each, so keep the window short.
                    if len(y_cache[i]) > 4:
                        for k in sorted(y_cache[i])[:-3]:
                            del y_cache[i][k]
                    if do_est or want_tags:
                        ys[i] = y_cache[i].pop(ptss[i], None)
                        if ys[i] is None:
                            y_stale[i] += 1
                    else:
                        # Nothing measures on these; they were only drained so the
                        # still could have one. Do not let the cache grow.
                        y_cache[i].clear()
                    if newest is not None:
                        nv12s[i], still_pts = sample_to_nv12(newest, w, h)
                        if nv12s[i] is not None and cloud.submit_still(
                            i, nv12s[i], w, h, clock.frame_time(still_pts),
                            want_still,
                            meta={
                                # How the picture was made. Every one of these is
                                # something a later marker or calibration fit has
                                # to know and cannot recover from the pixels:
                                # a calibration does not transfer across sensor
                                # modes, and it is only valid for the ORIENTATION
                                # it was captured in (see the rotated_180 warning
                                # further up) -- a mismatch there fails silently.
                                "mode": args.mode,
                                "rotated_180": not args.no_rotate,
                                "calib": (names[i] if cams[i] is not None else ""),
                                "wb": args.wb,
                                "saturation": args.saturation,
                                # The whole sensor frame. Said explicitly because
                                # every OTHER image this program emits is cropped,
                                # and a consumer that assumed the usual crop would
                                # get a plausible, wrong principal point.
                                "crop": "none",
                                "detect": "on" if det is not None else "off",
                            },
                        ):
                            stills += 1

            # ---- the dock's AR tags, on the FULL frame, before inference.
            #
            # Before, because with both lidars down this is the docking task's only
            # sensor and it must not be the thing that gets skipped when the
            # detector has a slow frame. It reads `ys[i]`, the full-resolution luma
            # plane already matched to this frame by PTS for the bearing
            # refinement, so it costs no extra capture and measures the same
            # instant everything else on this frame does.
            per_cam_tags = [[], []]
            if finder is not None and frames % max(1, args.artag_every) == 0:
                t_tag0 = time.perf_counter()
                for i in (0, 1):
                    if ys[i] is None:
                        continue
                    per_cam_tags[i] = finder.find(i, ys[i])
                last_tags = per_cam_tags
                tag_ms += 1000.0 * (time.perf_counter() - t_tag0)
                tag_n += 1
                if args.artag_bench:
                    both = per_cam_tags[0] + per_cam_tags[1]
                    for row in finder.bench_check(both):
                        print(f"[bench] tag {row['id']}: cam0 "
                              f"{row['cam0_bearing_deg']:+.2f} deg / "
                              f"{row['cam0_range_m']:.3f} m, cam1 "
                              f"{row['cam1_bearing_deg']:+.2f} deg / "
                              f"{row['cam1_range_m']:.3f} m -> yaw error "
                              f"{row['bearing_error_deg']:+.2f} deg, range "
                              f"differs {row['range_error_m']:+.3f} m")

            elif finder is not None:
                # A SKIPPED frame re-sends the last sighting rather than an empty
                # list. `tags: []` on the wire means "looked, and the berth is not in
                # view", and the Pi ages tags out at 1 s - so sending [] here would
                # tell it the berth had vanished on every other frame.
                per_cam_tags = last_tags

            # Two switches, and they are not the same switch. `--no-detect` means
            # no engine was ever loaded in this process; the dashboard's toggle
            # means one was and inference is paused. Both land here, and what
            # stops is only the inference: capture, previews, stills, the lidar
            # and the frames on the wire all carry on, because tearing the
            # pipeline down to stop inferring is what latches Argus.
            detecting = det is not None and (cloud is None or cloud.detect)
            if detecting:
                for i in (0, 1):
                    # Fused multiply into a preallocated buffer: the obvious
                    # astype()/255 form measured 17 ms per camera here, this one
                    # 3.5 ms.
                    np.multiply(rgbs[i].transpose(2, 0, 1), np.float32(1.0 / 255.0),
                                out=blob[i])
                out = det.infer(blob)[0]        # (2, 300, 6)
            else:
                out = None
            now = time.time()
            frames += 1

            per_cam = []
            for i in (0, 1):
                if out is None:
                    # An empty list, not a missing key: a frame with the detector
                    # off is a frame with nothing detected in it, and every
                    # consumer already handles that. The alternative -- omitting
                    # `dets` -- would make receiver.py and cloud_camera's overlay
                    # each need a new case for a state that is not new.
                    per_cam.append([])
                    continue
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
                per_cam.append(items)

            # ---- lidar fusion, between building the detections and sending them,
            # because it writes a range onto detections in BOTH cameras and needs
            # both frames to decide which one colours each return.
            #
            # The sweep is chosen by TIME against this frame's t_capture, not by
            # "the newest one". The camera pipeline runs ~250 ms behind the photons
            # and the lidar does not, so the newest sweep is a quarter-second in
            # front of the frame -- 0.75 m of registration error at 3 m/s, on a
            # sensor whose whole range is 12 m. LidarReader keeps a couple of
            # seconds of sweeps for exactly this lookup.
            cloud_pts = None
            if reader is not None and rig is not None:
                t_fuse0 = time.perf_counter()
                t_caps = [clock.frame_time(ptss[i]) for i in (0, 1)]
                sweep, _ = reader.sweep_near(t_caps[0])
                if sweep is not None:
                    # The boxes belong to THIS frame, so it stays the View's own
                    # `rgb`; the buffered ones are offered for colour only, and
                    # only where they were exposed nearer to a given return.
                    views = [fusion.View(i, cams[i], rgbs[i], per_cam[i],
                                         history=frame_hist[i])
                             for i in (0, 1)]
                    # Fuse on EVERY frame, so detections always carry a lidar
                    # range -- but build the point cloud only for a sweep the Pi
                    # has not had yet. The C1 settles at 10 Hz against 14 fps, so
                    # roughly three frames in ten are nearest to a sweep an
                    # earlier frame already sent; those still get their ranges,
                    # they just do not re-serialise a rotation nobody will read.
                    fresh = sweep.seq != last_sweep
                    try:
                        pts = fusion.fuse(sweep, rig, views,
                                          max_skew=args.lidar_max_skew / 1000.0,
                                          max_age=(args.lidar_max_age / 1000.0
                                                   or None),
                                          drop_unseen=not args.lidar_keep_unseen,
                                          drop_self=not args.no_lidar_self_box,
                                          t_caps=t_caps, build_cloud=fresh,
                                          ccm_strength=ccm_strength)
                    except Exception:           # geometry must never kill a frame
                        est_err[0] += 1
                        pts = None
                    lidar_skew = 1000.0 * abs(sweep.t_mid - t_caps[0])
                    if fresh and pts is not None:
                        last_sweep = sweep.seq
                        lidar_lit = pts["coloured"]
                        lidar_stale = pts["stale"]
                        lidar_self = pts["n_self"]
                        cloud_pts = pts
                # AFTER fusing, so a View is never offered the frame it is already
                # holding as `rgb` -- that would make the nearest-frame choice a
                # tie with itself and waste a buffer slot on a duplicate.
                for i in (0, 1):
                    frame_hist[i].appendleft((rgbs[i], t_caps[i]))
                # Kept on the stats line, not just measured once: this is the
                # only part of the frame budget the lidar can eat, and the budget
                # has ~10 ms of slack in it.
                fuse_ms += 1000.0 * (time.perf_counter() - t_fuse0)
                fuse_n += 1

            for i in (0, 1):
                items = per_cam[i]
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
                    "kind": protocol.KIND_FRAME,
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
                    # WHICH frame the preview JPEG is of, because since
                    # --preview-full there are two answers and a consumer that
                    # assumes the old one draws every box and every tag outline in
                    # the wrong place. "crop" scales net->jpeg directly; "full"
                    # goes net->full through `crop` first. See protocol.py.
                    "preview_source": "full" if args.preview_full else "crop",
                    "refined": ys[i] is not None,
                    # How much chroma the ISP already applied, so a viewer knows
                    # how much of the OV5647 matrix is left to apply and cannot
                    # correct twice. Correcting twice is not cosmetic: measured at
                    # saturation 2.0 with a full matrix on top, 53% of the frame
                    # clips.
                    "saturation": args.saturation,
                    "fps": round(fps_meas, 2),
                    "dets": items,
                    # The dock's AR tags, in the RIG frame, measured on the full
                    # sensor frame. Omitted entirely rather than sent as [] when
                    # tags are off, so the Pi can tell "looked and saw none" from
                    # "this build is not looking" -- the same distinction the
                    # dashboard's `caps` exists for, and the one that otherwise
                    # reads as a broken sensor. See protocol.py.
                    **({"tags": per_cam_tags[i]} if finder is not None else {}),
                }, jpeg)

                # Offer the same preview to the dashboard. It rate-limits and
                # re-encodes on its own thread, and returns immediately when video
                # is off, so this costs nothing in the common case. The picture
                # goes straight to shore; the detections above go to the Pi.
                #
                # `dets` goes with it only so the boxes can be BURNED IN. The
                # dashboard has no channel for detections beside the JPEG - they
                # went to the Pi - so an overlay drawn on shore is not an option
                # the way it is for receiver.py. Passing the list is two references
                # and no work: cloud_camera converts it after its own rate gate,
                # and draws in its worker thread.
                if cloud is not None:
                    cloud.submit(i, jpeg, pw, ph, t_cap,
                                 dets=items, det_size=(net_w, net_h),
                                 # The tags go to shore as OUTLINES ON THE PICTURE,
                                 # and their geometry goes to the Pi. Same split as
                                 # the detections, for the same reason: the
                                 # dashboard has no channel for either beside the
                                 # JPEG, and the /dock page's whole job is showing
                                 # an operator what the boat has hold of.
                                 tags=per_cam_tags[i],
                                 tag_frame=(w, h,
                                            "full" if args.preview_full else "crop",
                                            crops[i]))

            # The sweep goes as its own message rather than riding on a camera
            # frame: it belongs to neither camera (most of a rotation is behind
            # both of them), it arrives at its own rate, and duplicating it onto
            # both frames would double a payload that is already the larger half
            # of the header. Same framing, empty payload -- see protocol.py.
            if cloud_pts is not None:
                lidar_seq += 1
                tx.submit({"kind": protocol.KIND_LIDAR, "seq": lidar_seq,
                           "ts": cloud_pts["t_start"], "t_sent": round(now, 6),
                           "lidar": cloud_pts}, b"")

            el = time.monotonic() - t_stats
            if el >= args.stats_every:
                fps_meas = frames / el
                print(f"[{time.strftime('%H:%M:%S')}] {fps_meas:5.2f} fps/cam  "
                      f"sent={tx.sent} dropped={tx.dropped} "
                      # Said every window rather than only at start-up: the
                      # dashboard can turn this off mid-run, and "why are there
                      # no boxes" is otherwise a long hunt.
                      + ("detect=OFF " if not detecting else "")
                      + f"cardinals={cards} "
                      + (f"stills={stills} " if stills else "")
                      + f"stale_full={y_stale[0]}/{y_stale[1]} "
                      f"est_err={est_err[0]} "
                      f"link={'up' if tx.connected else 'DOWN'}"
                      + (f"  lidar={_lidar_line(reader, lidar_lit, lidar_stale, lidar_self, lidar_skew, fuse_ms / max(fuse_n, 1))}"
                         if reader else "")
                      + (f"  {artags.stats_line(per_cam_tags[0] + per_cam_tags[1], finder)}"
                         f" {tag_ms / max(tag_n, 1):.1f}ms" if finder else "")
                      + (f"  cloud={_cloud_line(cloud)}" if cloud else ""),
                      flush=True)
                frames, cards, stills = 0, 0, 0
                fuse_ms, fuse_n = 0.0, 0
                tag_ms, tag_n = 0.0, 0
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
        if reader is not None:
            # STOP and drain the C1 before the port closes, or it keeps streaming
            # into a dead port and the next run inherits the mess -- see lidar.py.
            reader.shutdown()
            reader.join(timeout=3.0)
        if cloud is not None:
            cloud.close()
        if det is not None:
            det.close()
        if cls is not None:
            cls.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
