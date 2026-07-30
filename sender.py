#!/usr/bin/env python3
"""Dual OV5647 -> YOLO26 detector -> cardinal classifier -> TCP to the viewer.

Runs on the Jetson. For each camera it emits a small preview JPEG plus the
detections, and the receiver draws the boxes.

Geometry, which is the fiddly part:
  sensor mode 0 gives 2592x1944 (4:3, full field of view, 14 fps). The detector
  wants 1280x640 (2:1). Letterboxing 4:3 into 2:1 would waste a third of the input
  on grey bars, and stretching would distort. So we CROP a 2592x1296 band and scale
  it by exactly 2.025 in both axes -- full sensor width, no padding, no distortion.
  Box coordinates therefore map to the preview with a single uniform scale.
  --crop-top picks which horizontal band, since on a boat you want the horizon.

The preview JPEG and the detector frame are branches of the same buffer, so they
share a PTS and are matched by it -- boxes can never be drawn on the wrong frame.

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
from gi.repository import GLib, Gst, GstApp  # noqa: E402,F401

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


def build_pipeline(args, w, h, fps, crop_h, crop_top, net_w, net_h, pw, ph):
    """Two cameras, each tee'd into a detector branch and a preview-JPEG branch."""
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
        # nvjpegenc only accepts video/x-raw(memory:NVMM) -- feeding it system
        # memory corrupts the conversion path and takes CUDA down with it.
        chunks.append(
            f"nvarguscamerasrc name=cam{sid} sensor-id={sid} sensor-mode={args.mode} "
            f"wbmode={WB_MODES.index(args.wb)} do-timestamp=true "
            f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 "
            f"! nvvideoconvert src-crop=0:{crop_top}:{w}:{crop_h}{flip} "
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
            # preview branch: downscale then hardware JPEG
            f"t{sid}. ! queue max-size-buffers=2 leaky=downstream "
            f"! nvvideoconvert ! video/x-raw(memory:NVMM),width={pw},height={ph},format=I420 "
            f"! nvjpegenc quality={args.quality} "
            f"! appsink name=jpg{sid} emit-signals=false max-buffers=3 drop=true sync=false"
        )
    return " ".join(chunks)


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

    # Crop a net_w:net_h-shaped band so the scale is uniform and there is no padding.
    crop_h = min(h, int(round(w * net_h / net_w)))
    crop_top = (h - crop_h) // 2 if args.crop_top is None else max(0, min(h - crop_h, args.crop_top))
    scale = w / net_w
    print(f"sensor {w}x{h}@{fps} -> crop {w}x{crop_h} at y={crop_top} "
          f"-> net {net_w}x{net_h} (uniform {scale:.3f}x, no letterbox)")
    print(f"preview {pw}x{ph} q{args.quality} -> {args.host}:{args.port}")

    Gst.init(None)
    desc = build_pipeline(args, w, h, fps, crop_h, crop_top, net_w, net_h, pw, ph)
    try:
        pipe = Gst.parse_launch(desc)
    except GLib.Error as e:
        print(f"pipeline build failed: {e}\n\n{desc}", file=sys.stderr)
        return 2

    dets = [pipe.get_by_name(f"det{i}") for i in (0, 1)]
    jpgs = [pipe.get_by_name(f"jpg{i}") for i in (0, 1)]

    tx = Sender(args.host, args.port)
    tx.start()

    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("failed to start pipeline", file=sys.stderr)
        return 2

    bus = pipe.get_bus()
    # One tracker per camera: they see different scenes, and ids are per-camera.
    # A lost track coasts for about a second whatever the sensor mode's rate.
    trackers = None if args.no_track else [Tracker(max_age=max(3, int(round(fps))))
                                           for _ in (0, 1)]
    blob = np.empty((2, 3, net_h, net_w), dtype=np.float32)
    jpeg_cache = [{}, {}]
    seq = [0, 0]
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
                    items.append(d)

                jpeg = jpeg_cache[i].pop(ptss[i], None)
                if jpeg is None and jpeg_cache[i]:
                    # Fall back to the newest preview if the exact PTS is missing;
                    # at most one frame of skew, and only when a branch hiccups.
                    jpeg = jpeg_cache[i].pop(max(jpeg_cache[i]), None)
                if jpeg is None:
                    continue

                seq[i] += 1
                tx.submit({
                    "cam": i, "seq": seq[i], "ts": now,
                    "net_w": net_w, "net_h": net_h,
                    "jpeg_w": pw, "jpeg_h": ph,
                    "fps": round(fps_meas, 2),
                    "dets": items,
                }, jpeg)

            el = time.monotonic() - t_stats
            if el >= args.stats_every:
                fps_meas = frames / el
                print(f"[{time.strftime('%H:%M:%S')}] {fps_meas:5.2f} fps/cam  "
                      f"sent={tx.sent} dropped={tx.dropped} "
                      f"cardinals={cards} "
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
