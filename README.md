# Buoy detector — live view

Dual OV5647 → YOLO26m detector → cardinal-mark classifier → boxes and a preview
JPEG streamed to a viewer on the network.

Clean split from the benchmarking work in `../yolo-test`; nothing here depends on
that directory.

## Run it

On the Jetson:

```bash
./run.sh                                    # sends to 192.168.99.135:3338
./.venv/bin/python sender.py --help         # all options
```

On the viewer machine (192.168.99.135) — needs Python 3, `pillow` and `numpy`.
Copy **both** `receiver.py` and `protocol.py` across; the receiver imports the
wire format from it.

```bash
pip install pillow numpy
python3 receiver.py                         # listens on :3338, serves on :8080
```

Then open **http://192.168.99.135:8080/** in a browser. Both camera feeds appear
side by side with boxes drawn. Order does not matter — the sender retries the
connection with backoff, so you can start either first and restart the viewer
without touching the Jetson.

If the feed never appears, check in this order: is `sender.py` actually running on
the Jetson (the receiver listens silently forever, so nothing arriving looks
exactly like a broken network); does the laptop's firewall allow inbound 3338
(Windows Defender prompts once for `python.exe` and blocks it if dismissed); is the
receiver running under WSL rather than native Windows (a WSL-bound listener is not
reachable from the Jetson without a `netsh portproxy`). Ping is not a useful test —
Windows drops inbound ICMP echo by default while still accepting the TCP listener.

## Models

| File | Role |
| --- | --- |
| `best.pt` | YOLO26m detector, 3 classes: `green`, `red`, `cardinal`. NMS-free (`end2end`), output `[batch,300,6]` |
| `best-cls.pt` | yolo26n-cls, 4 classes: `east`, `north`, `south`, `west`, at 96×96 |
| `best_640x1280_b2_fp16.engine` | detector, 1280×640 input, batch 2 (one slot per camera) |
| `best-cls_fp16.engine` | classifier, batch 1 |

The classifier runs **only** on detections of class 2 (`cardinal`), so its cost
scales with how many cardinal marks are actually in view, not with frame rate.

## Measured performance

Sensor mode 0 (2592×1944 @ 14 fps, full field of view), both cameras:

| | |
| --- | --- |
| Frame rate | **13.7 fps/camera, 27.5 aggregate** (98 % of the sensor's own limit) |
| Frame interval | mean 71.4 ms, p99 75.8, max 77.1, stdev 1.7 ms (2.4 %) |
| Dropped frames | **0 of 410** over 30 s |

Per-frame budget against the 71.4 ms period:

| Stage | Cost |
| --- | --- |
| CPU preprocessing, two cameras | 6.9 ms |
| Detector, batch of 2 (incl. host transfers) | 54.0 ms |
| Cardinal classifier | 1.1 ms per crop, capped at `--max-cardinals` (default 6) |
| Preview JPEG ×2 | on the VIC/GPU inside the pipeline, off the critical path |
| **Total typical** | **~61 ms of 71.4** |

Roughly 10 ms of slack, which is why the cardinal count is capped rather than
unbounded. If you need more headroom, an INT8 detector engine takes 54 ms down to
roughly 31 ms — see `../yolo-test/RESULTS.md`.

## Geometry, and why it is a crop rather than a letterbox

The sensor gives 2592×1944 (4:3). The detector wants 1280×640 (2:1). Letterboxing
would spend a third of the network input on grey bars; stretching would distort.
So the sender crops a **2592×1296** band and scales it by exactly **2.025× in both
axes** — full sensor width, no padding, no distortion. Box coordinates therefore
map to the preview with one uniform scale, which is what makes the overlay simple
and exact.

`--crop-top` chooses which horizontal band (default centred). On a boat you want
the band containing the horizon.

Because the full-resolution frame is what the camera captured, you can crop the
original at detection coordinates for a true high-resolution look at any object:
multiply box coordinates by 2.025 and add `crop_top` to the y values.

## Sensor modes

Only modes 0, 1 and 2 work on this board. **Modes 3 and 4 do not stream** — they
time out at the CSI level and poison the camera stack until a power cycle. Mode 4
(2592×1080 @ 27 fps) would be the ideal choice here, being the same full width at
twice the frame rate; see `../yolo-test/RESULTS.md` for how far that was narrowed
down.

| `--mode` | Geometry | fps | Field of view |
| --- | --- | --- | --- |
| 0 (default) | 2592×1944 | 14 | 100 % |
| 1 | 1920×1080 | 29 | 41 % of sensor area |
| 2 | 1296×972 | 28 | 100 %, 2×2 binned |

Mode 1 runs faster but is a hardware crop that throws away most of the frame, and
being binned, mode 2 has no extra resolution to crop into.

## Wire format

Length-prefixed frames over TCP; see `protocol.py`. Each message is a JSON header
plus a JPEG. Boxes are in detector-input pixels; the header carries both that size
and the preview size so the receiver can scale.

Detections look like:

```json
{"id": 7, "cls": 2, "name": "cardinal", "conf": 0.77,
 "box": [400.0, 250.0, 470.0, 340.0],
 "card": "north", "card_conf": 0.88}
```

`card` is null except on cardinal detections that were classified.

`/api/status` on the viewer returns the current detections as JSON, which is the
hook to use if you want to consume them from something other than the browser.

## Track ids

`id` stays with a buoy across frames, so consecutive frames can be related to each
other; the viewer draws it as `#7`. It is unique per camera, not across the pair.

There is no tracker in the engine to reuse. Ultralytics' `model.track()` is a
Python-layer wrapper (ByteTrack / BoT-SORT) around the PyTorch model — it is not
part of the graph and does not survive an ONNX/TensorRT export, so `Tracker` in
`sender.py` does the association. It costs nothing measurable; there are only ever
a handful of boxes.

Boxes are predicted forward one frame at their last measured velocity, then matched
by Hungarian assignment on a cost that is mostly IoU. Two details matter:

* **A distance gate as well as IoU.** Buoys are small at range, so in any swell a
  box can travel further than its own height between frames and IoU for the correct
  pair is then exactly zero. IoU alone hands out a fresh id every few frames.
* **Association ignores class.** A red buoy that reads `green` for one frame is
  still the same buoy. Gating on class would split it into two tracks, and the
  point of the id is to make that flicker visible rather than to hide it.

A lost track coasts for about a second before its id is retired, so a brief miss
does not renumber. Ids are never reused, but nothing guarantees a buoy keeps one
across a long occlusion. `--no-track` turns it off.

## Orientation and colour

The mount is inverted, so the frame is rotated 180° by `flip-method=2` on the
`nvvideoconvert` that already does the crop — before the `tee`, so it is one VIC
pass shared by both branches and the detector sees the same upright frame as the
viewer. Boxes then need no coordinate flip anywhere. `--no-rotate` for an upright
mount.

Not done with the sensor's flip registers: those change the Bayer phase
(BGGR → RGGB) and would silently wreck the demosaic. `src-crop` is applied in
input pixels — verified, not assumed — so `--crop-top` keeps meaning "band measured
down the raw sensor frame" whether the rotation is on or off.

White balance is pinned to `daylight` rather than left on auto. The detector
classifies by colour, and auto WB shifting hue between frames is worse than a
consistent small error. `--wb auto` restores the old behaviour.

JetPack ships no ISP colour tuning for the OV5647, so Argus returns close to
sensor-native RGB. Every Bayer sensor has heavy spectral crosstalk between its
filters, and measured against SMPTE bars these primaries come out at 0.30–0.41
saturation instead of 1.00 — which is why reds arrive brown. The **receiver**
applies the 3×3 matrix that undoes it (fitted in `../camera-test`, which also has
the tooling to refit it for your own lighting).

It runs on the viewer, not the Jetson, for two reasons: the Jetson has ~10 ms of
slack per frame and this needs more than that, and the viewer has already paid for
the JPEG decode in order to draw boxes. Measured on the Jetson's own ARM core, so
a laptop will be several times quicker:

| Viewer work per frame | Cost |
| --- | --- |
| decode + boxes + encode | 8.3 ms |
| the same plus `--ccm 1.0` | 27.2 ms |

`--ccm 0.5` softens it and `--ccm 0` disables it. Worth reaching for in dim scenes:
the off-diagonal terms are large because the crosstalk is, so full strength also
amplifies chroma noise. If the viewer cannot keep up, the sender's `dropped=`
counter is where it shows.

The correction deliberately does **not** touch what the detector sees — that path is
unchanged, so detection behaviour is exactly what it was before the matrix existed.
Whether the model would do better on corrected input depends on what it was trained
on and is worth an experiment rather than an assumption.

## Things that will bite

* **Only modes 0/1/2 stream.** Asking for 3 or 4 disables the cameras until a cold
  power cycle.
* **Pin the caps.** `nvarguscamerasrc` defaults to 1920×1080 when caps leave
  resolution free; a mode/geometry mismatch wedges the capture stack the same way
  an unstreamable mode does.
* **`nvjpegenc` accepts NVMM memory only.** Handing it system memory corrupts the
  conversion path and surfaces as `cudaErrorIllegalAddress` inside inference.
* **Shut down with EOS, not a kill.** Tearing the pipeline down mid-capture leaves
  Argus latched in an error state. `sender.py` drains on Ctrl-C for this reason, so
  stop it with SIGINT (`pkill -INT -f sender.py`) and never SIGKILL.
* **`nvvideoconvert` defaults to the VIC on Jetson**, and the VIC cannot do
  NV12 → RGB. The detector branch needs exactly that, so it carries
  `compute-hw=GPU`; without it the branch fails with *"RGB/BGR Format
  transformation is not supported by VIC"* and takes the whole pipeline down with
  `Internal data stream error` — both cameras, not just the branch.
* **No NVENC on Orin Nano.** There is no hardware H.264/H.265 encoder on this
  module, which is why the preview is MJPEG.
* **Recovering a wedged camera stack:** power-cycle. Restarting `nvargus-daemon`
  and reloading `nv_ov5647` were both tried and neither works.
