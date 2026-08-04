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

Adding the full-resolution branch and per-detection bearing/range estimation costs
essentially nothing measurable: **13.2–13.9 fps/camera, `dropped=0`, `stale_full=0`**.
The extra branch is a VIC pass and the geometry is a handful of Newton iterations per
box.

## Geometry, and why it is a crop rather than a letterbox

The sensor gives 2592×1944 (4:3). The detector wants 1280×640 (2:1). Letterboxing
would spend a third of the network input on grey bars; stretching would distort. So
the sender **crops** a 2:1 band — no padding, no distortion — and box coordinates
map to the preview with one uniform scale, which is what makes the overlay exact.

The band is **2048×1024 scaled by 1.6×** by default, and *not* the full sensor
width. Full width was the original choice, but the calibration says the left and
right edges of a 2592-wide crop fall outside the lens's valid cone (88° off axis),
where a pixel has no bearing at all. Giving those up buys 27 % more pixels per buoy
as well.

`--aim-deg` swings the window toward the pair's **overlap**, which was measured
rather than assumed — the same ceiling lamp sits at cam0 x≈2045 and cam1 x≈280, so
the cameras diverge with the overlap on cam0's right and cam1's left. The default
`15` applies +15° to cam0 and −15° to cam1:

```
cam0: crop 2048x1024 at (544,460), aim +15.0 deg, covers 139.3x69.4 deg, 1.200 mrad/px
cam1: crop 2048x1024 at (119,460), aim -15.0 deg, covers 139.2x69.4 deg, 1.201 mrad/px
```

The aim is clamped to what the crop width allows (a 2048-wide window cannot swing
much past 15°) and the *achieved* value is printed, so a request that could not be
honoured is visible rather than silent. `--crop-w` sets the width; `--crop-top`
picks the band, and on a boat you want the one containing the horizon.

The full-resolution frame is kept as its own pipeline branch, so any detection can
be measured at native resolution: multiply box coordinates by `crop_w / net_w` and
add the crop origin. The header carries `crop`, `full_w` and `full_h` for exactly
this.

## Bearing, range and capture time

With a calibration present (`--calib`, on by default), every detection gains a
bearing, a range and an uncertainty for each — see [estimate.py](estimate.py) and
the field list in [protocol.py](protocol.py). `--no-estimate` turns it all off,
including the full-resolution branch that feeds it.

**Capture time.** The header's `t_capture` comes from the GStreamer buffer PTS, not
from `time.time()` after inference. That matters more than it sounds: measured over
40 frames, capture→send latency was **min 174, median 247, max 471 ms**. The old
timestamp was a quarter-second late *and varied by 300 ms*, so it was not a constant
you could have subtracted out. `t_sent` and `latency_ms` are on the wire separately
so pipeline delay stays visible instead of contaminating the measurement.

This is also a rolling shutter with a **64 ms top-to-bottom sweep**, so a frame does
not have a single capture instant. Each detection carries its own `t_row`; use that
for anything geometric.

**Bearing** is reported as azimuth/elevation and as a unit vector `ray_cam`, in the
**camera** frame. Relating the two cameras to each other or to the boat needs the
mount rotation, which is not in this repo. σ is ~0.26°, and it is split on the wire
because the parts behave differently: `sigma_calib_deg` (0.25°) is the calibration's
own error and is **correlated** — the same on every detection, every frame, so it
does not average out over a track and does not cancel between the cameras. Only
`sigma_centroid_deg` (~0.03–0.07°) is independent noise that averages down.

**Range** comes from apparent size, assuming a sphere of `--buoy-diameter` (0.40 m
for Njord marks): `z = (D/2)/sin(α/2)`, the exact tangent geometry. It measures
**width, not height**, because these marks float and the waterline cuts an unknown
amount off the bottom. Error grows as z²:

| Range | Buoy width | σ | |
| --- | --- | --- | --- |
| 10 m | 33 px | ±0.5 m (5 %) | good |
| 20 m | 17 px | ±1.3 m (7 %) | good |
| 30 m | 11 px | ±2.4 m (8 %) | good |
| 50 m | 6.7 px | ±5.9 m (12 %) | marginal |
| 100 m | 3.3 px | ±22 m (22 %) | check `valid` first |

Validated by planting synthetic buoys of known range into real frames. Subpixel edge
refinement engages down to ~15 full-resolution px (≈25 m) and falls back to the
detector box beyond that, with σ widening honestly — a 35 m buoy read 28 ± 9.8 m,
truth inside the bar. **Weight by `sigma_m` and check `valid`**; do not trust
`range_m` merely because a number came back.

Beyond ~30 m, apparent size is the wrong cue. Depression below the horizon is far
better — a buoy at 100 m with a 2 m camera height sits 20 mrad ≈ 16 px below the
horizon, giving ~6 % instead of 22 % — but it needs camera height and attitude.
`../camera-test/horizon/` has the horizon detection for it.

## Calibration

[calibrate/](calibrate/) fits a Kannala-Brandt fisheye model per camera. The results
live in `calibrate/calib/cam0.json` and `cam1.json` and are what the sender loads;
the stills they were fitted from are gitignored, being hundreds of MB specific to
one lens at one focus position.

```bash
./.venv/bin/python calibrate/calib_server.py          # then browse to :8080, space to shoot
./.venv/bin/python calibrate/calibrate_fisheye.py --help
```

Both cameras came out at **RMS ~1.18 px** over 123/160 views, an implied HFOV of
168.1°/166.7°, and an expected bearing error of **~0.25°**, unbiased. That number is
a split-half disagreement halved (each half-fit sees half the data), and it is
*systematic*, not noise-limited — doubling the data did not shrink the parameter
spread. The likely floor is LCD cover-glass parallax at oblique angles, since the
target was a laptop screen; a matte printed target is the change that could reach
~0.1°.

Two things about this model are worth knowing before touching the code:

* **`cv2.fisheye` cannot represent field angles ≥ 90°.** It computes
  θ = atan(‖(X/Z, Y/Z)‖), so past the wall it silently *folds back* — 110° lands at
  the same radius as 70°. Verified empirically on this build, and it is why the
  fitter computes a **valid cone** (here 88°) and returns NaN outside it rather than
  a plausible wrong answer. `in_valid_cone` on the wire is the same idea.
* **Scale of a planar target is a gauge freedom for the intrinsics** — it moves
  `tvecs` only, not K or D. *Non-uniform* scale is not, and permanently corrupts the
  fx/fy ratio, which is why the target's aspect must be right even when its absolute
  size does not matter.

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
 "card": "north", "card_conf": 0.88,
 "bearing_deg": 12.44, "elevation_deg": -1.87, "field_angle_deg": 27.6,
 "ray_cam": [0.2154, -0.0326, 0.9760], "in_valid_cone": true,
 "sigma_deg": 0.259, "sigma_calib_deg": 0.25, "sigma_centroid_deg": 0.068,
 "mrad_per_px": 1.204,
 "range": {"range_m": 21.7, "sigma_m": 1.5, "rel_sigma": 0.069,
           "alpha_mrad": 18.4, "valid": true, "why": null},
 "width_method": "refined_edges", "edge_sigma_px": 1.0, "width_px_full": 15.3,
 "truncated": false, "t_capture": 1785876440.230088, "t_row": 1785876440.264}
```

`card` is null except on cardinal detections that were classified. The geometry
fields are null without a calibration or outside the valid cone.

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
* **`src-crop` must happen exactly once.** `nvvideoconvert` writes its crop
  rectangle into the shared `NvBufSurface` metadata, and a `tee` hands every branch
  the *same* surface — so two croppers off one tee race, and it segfaults with
  *"Failed in mem copy"* (`nvbufsurftransform_copy.cpp:438`) and a core dump. Each
  branch works perfectly in isolation, which is what makes it a trap. Hence the
  two-level tee in `build_pipeline`.
* **Match branches by PTS, never by "newest".** The full-resolution branch has no
  inference in it, so it runs a frame ahead of the detector; taking the newest sample
  measured buoys against the wrong frame 99 % of the time (`stale_full=944/954`).
  Both the JPEG and the full-res caches are PTS-keyed for this reason, and
  `stale_full=` in the stats line is there to prove it stays at 0.
* **`--crop-w 2592` is not a valid option** even though the sensor is that wide: the
  edges fall outside the calibration's 88° cone, so detections there have no bearing.
* **Recovering a wedged camera stack:** power-cycle. Restarting `nvargus-daemon`
  and reloading `nv_ov5647` were both tried and neither works.
