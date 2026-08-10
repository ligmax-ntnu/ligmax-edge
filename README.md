# Buoy detector — live view

Dual OV5647 → YOLO26m detector → cardinal-mark classifier → boxes and a preview
JPEG streamed to a viewer on the network.

Clean split from the benchmarking work in `../yolo-test`; nothing here depends on
that directory.

## The other repos on this board

Checked out beside this one, because half the questions here are answered on the
other side of a wire and guessing at the far end wastes an afternoon:

| path | what it is |
| --- | --- |
| `../ligmax-server` | the dashboard at `live.ligmax.no`. `git clone https://github.com/andreasviner/ligmax-server.git` if it is missing. `ligmax_gui/camera.py` is the frame relay this repo's `cloud_camera.py` talks to, `ligmax_gui/server.py` holds the routes and the auth, `web/js/camera.js` is the panel. |
| `../camera-test` | where the dual-OV5647 capture was first got working — `capture_both.py`, `crop_planner.py`, `live_server.py`. The Bayer-phase finding behind `flip-method=2` (rather than the sensor's flip registers) is from here. |
| `../yolo-test` | engine builds, precision sweeps and `RESULTS.md`. Nothing here imports it. |

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

## The camera stream to the dashboard

Three things consume the cameras and they all hang off **one crop**, which is the
property that makes the operator's picture trustworthy: the detector branch, the
preview/stream branch and the lidar colouring all read the same 2048×1024 window
out of the same `tee`. Only the scale differs.

```
nvarguscamerasrc 2592x1944  ! rotate 180  ! tee a{sid}
    a. ! full frame, NV12, system memory  -> estimate.py       (range refinement)
    a. ! src-crop 2048x1024 -> 1280x640   -> tee t{sid}
           t. ! RGB          1280x640     -> YOLO26m + classifier
           t. ! nvjpegenc     640x320     -> Pi viewer, and cloud_camera.py
                                              -> re-encoded 480x240 -> live.ligmax.no
```

2048×1024, 1280×640, 640×320 and 480×240 are all exactly 2:1 and all the same
window, so **what shore sees is the detector's field of view, pixel for pixel, at
a quarter of the linear resolution**. There is no second capture path and no
second crop to drift out of sync — changing `--crop-w`, `--aim-deg` or
`--crop-top` moves the detector and the stream together by construction. The
downscale in `cloud_camera._encode` derives the height from the width, so the
server's `max_width` slider cannot break the aspect either.

### Boxes are burned into the frame, and only here

`receiver.py` gets the detections as JSON alongside the JPEG and draws its own
overlay. The dashboard cannot: the detections went to the Pi and reach the
operator as objects on the *map*, so the only thing arriving at `/api/camera` is
pixels. A clean picture there would show what the lens sees and never what the
detector sees — and the interesting case is exactly when those two disagree.

So `cloud_camera.py` draws them in, using the same class colours as
[receiver.py](receiver.py) so a buoy is not green on one screen and yellow on the
other. The label is just the confidence, plus the cardinal direction when there is
one: the box colour already says green/red/cardinal, and *which* cardinal is the
one thing colour cannot encode. At 480 px a longer label covers the buoy it
describes.

None of it touches the frame budget:

* Coordinates are handed over as two references. They are converted to fractions
  **after** the uplink's rate gate, so a frame about to be dropped costs nothing.
* Fractions, not pixels, because `max_width` is a slider on the dashboard and can
  change between the submit and the encode.
* The drawing happens in the uplink's worker thread, at the *stream's* 2 fps
  rather than the detector's 12 — measured **+3.6 ms per frame** on top of the
  6.0 ms the downscale already cost, so about 1 % of one core across both cameras,
  and none of it on the capture loop.
* Drawn **after** the downscale. The other order thins a 2 px outline to under a
  pixel and resamples the labels into mush.

`--no-cloud-boxes` sends a clean picture. It buys no detector headroom — it is for
judging the lens without the detector's opinion drawn over it.

### Two things had to be wrong at once, and both were

This link broke in a way worth writing down, because each half on its own looks
exactly like the other and neither shows up as an error anywhere.

**1. No `User-Agent`.** Cloudflare fronts `live.ligmax.no` and 403s (error 1010)
anything that looks like a stock library client. `http.client` sends no
`User-Agent` at all unless told to. The frame POST set one; the config GET did
not — so the poll was refused at the edge, before Flask ever saw it, `enabled`
never went true, and therefore no frame was ever offered in the first place. The
dashboard's own diagnosis for this is *"the Jetson has never asked for the
config"*, which reads as a dead board. `_headers()` now builds one set of headers
for every request here, which is the point of it existing at all — the same trap
already caught `update.py` once.

**2. `/etc/ligmax/node.env` had only `LIGMAX_NODE_KEY`.** The two keys are not
interchangeable:

* `LIGMAX_NODE_KEY` → `update.py`, the deploy/pull endpoints.
* `LIGMAX_BOAT_KEY` → `cloud_camera.py`, the frames.

The server takes *either* key on `GET /api/camera/config` but **only the boat
key** on `POST /api/camera` (`ligmax-server/ligmax_gui/server.py`). So a board
carrying just the node key updates itself perfectly, reports healthy, and cannot
push a single frame — and with no boat key at all the config poll 403s too.

Both are fixed, and the stats line now tells them apart instead of saying a bare
`cloud=off`:

```
cloud=UNREACHABLE(...)   the poll is not being answered  -> UA, key, or network
cloud=off                the poll IS answered, the answer is no -> nobody has
                         switched video on in the dashboard yet
cloud=998sent/480px q55 2.0fps                           -> streaming
```

Check it end to end from the Jetson. Note the `-H` — curl sends its own agent, so
a bare `curl` will succeed where the daemon fails, which is exactly how the first
half of this hid for so long:

```bash
KEY=$(sudo sed -n 's/^LIGMAX_BOAT_KEY=//p' /etc/ligmax/node.env)
curl -s -H "Authorization: Bearer $KEY" https://live.ligmax.no/api/camera/config
# {"cameras":["0","1"],"enabled":false,"fps":2.0,"jpeg_quality":55,"max_width":480,...}
```

`403 {"error":"boat key required"}` is Flask, so the key is wrong or missing. A
403 with Cloudflare's HTML body instead is the agent. A `200` with
`"enabled":false` is the normal resting state — video is off by default because
it shares the 4G uplink with telemetry and the E-stop ack, and an operator has to
switch it on from the dashboard's camera panel. The server counts refused polls
(`refused` / `last_refusal` in `GET /api/camera/state`), which is the far end of
the same signal.

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

## Lidar, and colouring it from the cameras

An RPLidar C1 on the front, read by [lidar.py](lidar.py), projected into both
cameras by [fusion.py](fusion.py), and sent to the Pi as its own message type
alongside the detections. Two products come out of it:

* **A colourised point cloud.** A 2D scanner returns a range and nothing else, so
  a wall, a buoy and a person are the same measurement. The cameras say which.
* **A true range on any detection the lidar can see.** Range from apparent size
  is ±5 % at 10 m and degrades as z²; the C1 is ±3 cm flat. Inside its 12 m the
  lidar wins by an order of magnitude, so detections gain a `lidar` block —
  *alongside* `range`, never replacing it, because two independent measurements
  that disagree are worth being able to notice.

```bash
./.venv/bin/python lidar.py --seconds 5      # sweep summaries; sensor only
./sender.py --no-lidar                       # pipeline with the lidar switched off
```

Measured on this unit, **once settled: 10.0 Hz, ~400 returns per rev at 0.9°**,
about a third of which come back with distance 0 (no echo) and are dropped in the
driver. Measure it *after* it settles — the C1 comes up nearer 14 Hz and slows
over the first few seconds, so a short run reports a rate it does not hold.

### Time, which is where this goes wrong quietly

A rotation takes **100 ms — longer than a camera frame** — and a camera frame
reaches the fusion code ~250 ms after its photons landed. Both matter:

* The sweep is chosen **by capture time** out of a two-second buffer, not by
  "the newest one". Newest is ~250 ms ahead of the frame, which at 3 m/s is
  0.75 m of registration error on a sensor whose whole range is 12 m.
* Timing is **per point, not per sweep**. Every return carries its own timestamp
  (`dt_ms`, interpolated by angle). Judging the sweep as a whole cannot work: a
  100 ms rotation's midpoint is up to 50 ms from any frame no matter how well the
  two are running, so colour would degrade permanently while nothing was wrong.
* Each point is coloured from **its own nearest frame**. One frame is an instant
  and a rotation is 100 ms, so no single frame is close to all of a sweep: with a
  ±40 ms gate the arithmetic caps colouring at 80 % of a rotation and falls to
  ~45 % as the sweep midpoint drifts. Measured here it sat at a **median 46 %,
  breathing between 13 % and 68 %** as the 10 Hz scanner beat against ~10 fps
  cameras — which is what a viewer sees as the colour pulsing. `sender` keeps the
  last `--lidar-frame-history` frames per camera (default 2, plus the current
  one) and every return takes whichever was exposed nearest to it. Costs
  0.3 ms per frame and ~10 MB, and takes colouring to every return a lens covers.

Past `--lidar-max-skew` a point is coloured **anyway**, from the closest frame
there is — a slightly mistimed colour beats none — but honestly: `age_ms` carries
each point's own frame distance and `stale` counts the ones outside the gate, so
a consumer can down-weight them instead of being unable to tell. Frames land
~100 ms apart, so worst-case age is now bounded near 50 ms rather than a full
rotation. `--lidar-max-age` (default 250 ms) is the hard stop for a camera that
has stalled and left the buffer stagnant; only there do points ship uncoloured.

Returns **no camera could see at all** — the ~34° aft wedge outside both lenses —
are dropped rather than shipped grey. `n` is then the points in the arrays and
`dropped` how many were removed (and see `n_self` below: a rev is
`n + dropped + n_self`).
This is a data decision, not a speed one: it assumes the aft lidar watches that
arc, and it buys ~10 % of the wire but only ~1 % of `fuse`, since the work was
never in the points no camera could see. `--lidar-keep-unseen` ships them.

### The boat is in the way

A 360° planar scanner on the bow sees **the vessel it is bolted to**: the
superstructure, the mast, the aft lidar's housing, anything stowed on deck. To a
range-only sensor those are the same measurement as a mark at 2 m — and worse, a
hull return that lands inside a detection box is the *nearest* return in it, so it
wins the foreground cluster and the buoy 8 m away is reported at 2 m.

So `fuse` discards them geometrically, first thing, before anything is projected
or coloured: **`self_box` in [rig.json](rig.json)**, a box in the rig frame the
hull occupies. Default **0.70 m either side of the centreline, from the lidar's
own plane (`z = 0`) running aft with no rear edge** — a 1.40 m corridor astern,
which at 3 m range is a 27° arc and ~27 of ~360 returns. No rear edge because
nothing measured says where the deck stops, and behind is the aft lidar's arc
anyway. A box in metres, not an angular wedge, because the hull subtends a wide
arc up close and a narrow one further aft; a fixed wedge either keeps stern
returns or eats open water off the bow quarter.

The count ships as **`n_self`** and appears as `self=` on the stats line and
`(N self)` on the receiver's plot, so it is never a silent removal. A
returns-per-rev check therefore wants `n + dropped + n_self`.

```bash
./sender.py --no-lidar-self-box --lidar-keep-unseen   # see what the mask eats
./sender.py --lidar-self-box 0.9,0.0,none            # try other numbers, no edit
```

**These are requested dimensions, not measured against the hull**, and the box's
meaning depends entirely on `lidar.yaw_deg` — which is stale since the 2026-08-08
remount, so "aft" is currently pointing ~90° away from the stern. Fix the yaw
first, then check the box; the two failures look identical from shore, and both
look like a working plot.

`skew=` on the stats line is the sweep-to-current-frame capture-time difference —
a whole-sweep summary, no longer what decides any one point's colour. `in_time`
on the wire still separates the two reasons a return is short of a good colour:
measured at the wrong moment (timing) versus outside every lens (geometry,
expected). `coloured` can now exceed `in_time`; the excess is `stale`.

**If a whole sweep goes gray, suspect the capture clock, not the lidar.** The
Jetson has no RTC, and an NTP step landing after `estimate.CaptureClock` sampled
its offset makes every frame time wrong by that step, so every point looks
untimely for the life of the process — seen here as 58 minutes and 7.2 hours of
`skew=`. `frame_time()` re-baselines past a 1 s step and logs `[capture-clock]`;
a `skew=` in seconds or more is that fault, tens of ms is normal.

### What it costs

The lidar is **not free**, and unlike the bearing/range work it is measurable.
Measured on this board, alternating runs of 34 s, no receiver attached:

| | fps/cam |
| --- | --- |
| `--no-lidar` | **13.0–13.5** |
| reader thread only (no fusion) | 12.5 |
| reader + fusion | **11.0–11.75** |

So roughly **12 ms of the 71.4 ms frame period**, of which ~4.5 ms is `fuse()`
itself (timed directly — it is the `fuse=` field on the stats line) and ~3 ms is
the reader thread competing for the GIL; the rest is scheduling. Run-to-run
spread is ±0.5 fps, so treat these as approximate.

Two rounds of optimisation are already in: the columnar arrays are built with
`np.round(...).tolist()` rather than a per-element comprehension (5.11 → 3.34 ms),
and each camera projects only the points in front of it rather than the whole
rotation (3.34 → 2.97 ms isolated). The remaining cost is many small numpy calls
on ~400-point arrays, where per-call dispatch overhead dominates on an ARM core.

**`--no-lidar` restores the old frame rate exactly** if you ever need it back.
11.4 fps/cam is still comfortably above the C1's own 10 Hz, so the fusion is not
losing sweeps — it is spending camera frames.

### Mounting geometry

[rig.json](rig.json) — hand-measured, and the file to edit when anything is
re-bolted. Everything is in a **rig frame** (`+x` starboard, `+y` **down**, `+z`
forward, origin at the lidar), which is also the frame the point cloud ships in,
so the Pi can merge it with the aft lidar without guessing.

Defaults: cameras horizontal, ±15° either side of forward, 5 cm from the lidar
centre along their own pointing direction and 5 cm above the scan plane. The
lidar body is mounted rotated **45° to port**, so `lidar.yaw_deg` is `-45` to
turn the scan back onto the boat's axes. `cam0` and `cam1` are written out in
full rather than mirrored, so each absorbs its own mounting error independently.

Nothing downstream can tell you these numbers are wrong — a slightly wrong
transform still produces a full, plausible, entirely mis-registered cloud. So
check them:

```bash
./.venv/bin/python test/test_lidar_overlay.py            # both cameras
./.venv/bin/python test/test_lidar_overlay.py --yaw 12.5 # sweep a value, no edit
```

It draws the returns onto a real frame, coloured by range. Put a hard vertical
edge 1–3 m out; the returns must land *on* it. Points consistently left or right
→ yaw; above or below and converging with range → `dy`; the whole world mirrored
→ `angle_dir`; rotated by a constant → `yaw_deg` or `angle_zero_deg`.

The colour lookup goes through the **calibrated Kannala-Brandt model**, not a
pinhole approximation — at 60° off axis a pinhole would land 356 px wrong in a
1280-wide frame. Returns outside the 88° valid cone get no colour rather than a
plausible wrong one.

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
  stop it with SIGINT (`pkill -INT -f sender.py`) and never SIGKILL. **SIGTERM is
  handled too, and has to be**: Python's default for it exits without running
  `finally`, so the pipeline was never drained — and SIGTERM is exactly what
  `systemctl stop`, `systemctl restart` and the dashboard's Update button all
  send (`update.py` SIGTERMs the process group before it pulls). Any of those
  could latch the cameras until a power cycle. `sender.py` now turns SIGTERM into
  the same clean path as Ctrl-C.
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
* **Every request to `live.ligmax.no` needs a `User-Agent`.** Cloudflare 403s
  (error 1010) anything without one, and `http.client` sends none by default. A
  bare `curl` will not reproduce it, because curl sends its own. Bit both
  `cloud_camera.py` and `update.py`.
* **`node.env` needs the boat key too.** `LIGMAX_NODE_KEY` alone gets you a board
  that updates itself, reports healthy and can never send a picture, because the
  server takes either key on the config poll and only the boat key on the frame
  POST. See "The camera stream to the dashboard".
* **Recovering a wedged camera stack:** power-cycle. Restarting `nvargus-daemon`
  and reloading `nv_ov5647` were both tried and neither works.
