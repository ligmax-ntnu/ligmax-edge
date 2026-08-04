#!/usr/bin/env python3
"""Live browser view for capturing fisheye calibration stills from both OV5647s.

    nvarguscamerasrc mode 0 --> BGR appsink --> aruco overlay --> MJPEG to browser
                                     |                                   |
                              full-res frame kept              you see coverage fill
                                     |                          in as you shoot
                              [Capture] --> PNG + coverage

    ./calib_server.py            # then open http://<jetson-ip>:8080/

Same shape as ../camera-test/focus_server.py, which is where the Argus pipeline and
the publish/wait threading pattern come from -- but self-contained, because
buoy-live deliberately does not import from that directory.

WHY A SERVER AND NOT A CAPTURE LOOP. Over SSH there is no preview, and a shot that
detected nothing looks exactly like one that worked. On a 180deg lens the thing that
decides whether the calibration is any good is coverage of the OUTER cone, and you
cannot aim at the outer cone blind. So this draws, live:

  * the valid cone boundary (red) and iso-angle rings -- see calibrate_fisheye.py
    for why anything outside that circle is unusable no matter how well you shoot it
  * every marker the detector currently finds, so a view that will not count is
    obvious before you press the button rather than after the fit
  * the accumulated coverage, tinted over the frame, so you can see which part of
    the cone is still empty and put the board there

Point the browser at the Jetson from the same laptop that is displaying the board:
hold the laptop up, glance at the browser, press space, move on.

The 180 rotation matches sender.py's default, so the fit transfers to what the
detector sees by pure arithmetic (crop, then scale) with no coordinate flip to
remember:

    ./calibrate_fisheye.py --images calib/cam0 --marker <measured> --fov 180 \
        --out calib/cam0.json
    # sender.py adapts the model to its own crop itself, so this is only for
    # inspecting what the detector sees standalone.
    ./calibrate_fisheye.py --adapt calib/cam0.json --crop_left 544 --crop_top 460 \
        --crop_size 2048x1024 --scale_to 1280x640 --out calib/cam0_det.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_fisheye as cf  # noqa: E402

W, H, FPS = 2592, 1944, 14      # sensor mode 0. Modes 3 and 4 do not stream at all.
PREVIEW = (1296, 972)           # exactly half, so overlay coords scale by 2
WB_MODES = ["off", "auto", "incandescent", "fluorescent", "warm-fluorescent",
            "daylight", "cloudy-daylight", "twilight", "shade"]


def exposure_props(us: float, gain: float, dgain: float) -> list:
    """Argus properties that pin brightness to ONE operating point.

    All three have to be pinned together, which is the part that catches people out.
    Constraining exposuretimerange alone does not stop auto-exposure -- it just makes
    AE reach for the other two knobs instead, cranks analog and ISP digital gain to
    hit its target brightness, and blows the white squares out again exactly as
    before, only now with more noise for the same clipping. A range whose min equals
    its max is how Argus is told "no choice available".

    Times are microseconds here and nanoseconds on the wire, because 4000 is a
    number you can reason about and 4000000 is one you mistype.
    """
    ns = int(round(us * 1000))
    return [f'exposuretimerange="{ns} {ns}"',
            f'gainrange="{gain:.3f} {gain:.3f}"',
            f'ispdigitalgainrange="{dgain:.3f} {dgain:.3f}"']


def build_pipeline(sensor_id: int, cfg: dict) -> str:
    flip = "" if cfg["no_rotate"] else " flip-method=2"
    src = [f"nvarguscamerasrc sensor-id={sensor_id}",
           f"wbmode={WB_MODES.index(cfg['wbmode'])}",
           "do-timestamp=true"]
    # A bright screen is the worst case for AE: it blows out the white field, the
    # black markers bloom into it, and corner refinement is then biased on exactly
    # the measurement the whole calibration rests on. Pinning is the fix, which is
    # why it is a first-class control here and settable straight from the URL.
    if cfg.get("exposure_us"):
        src += exposure_props(float(cfg["exposure_us"]), float(cfg.get("gain") or 1.0),
                              float(cfg.get("dgain") or 1.0))
    return (" ".join(src) +
            f" ! video/x-raw(memory:NVMM),width={W},height={H},format=NV12,"
            f"framerate={FPS}/1"
            f" ! nvvidconv{flip} ! video/x-raw,format=BGRx"
            f" ! videoconvert ! video/x-raw,format=BGR"
            f" ! appsink drop=true max-buffers=2 sync=false")


class CalibStream:
    """One Argus pipeline: publishes a preview JPEG and keeps the full-res frame."""

    def __init__(self, sensor_id: int, cfg: dict, outdir: str, cone, params, dic,
                 n_markers: int, quality: int):
        self.sensor_id = sensor_id
        self.cfg = dict(cfg)
        self.outdir = outdir
        self.cone = cone
        self.params, self.dic, self.n_markers = params, dic, n_markers
        self.quality = quality

        self._frame: np.ndarray | None = None      # full res BGR, most recent
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None

        self.coverage = np.zeros((H, W), np.uint8)
        # Resume numbering past whatever is already on disk. Restarting at v000 would
        # silently overwrite an earlier session's shots and, worse, leave its tail
        # behind: the surviving high-numbered files get fitted together with the new
        # ones as though they were one session, so a set captured at a corrected
        # distance is quietly contaminated by the batch it was meant to replace.
        d = os.path.join(outdir, f"cam{sensor_id}")
        prev = [f for f in os.listdir(d) if f.startswith("v") and f.endswith(".png")] \
            if os.path.isdir(d) else []
        self.shots = 0
        for f in prev:
            try:
                self.shots = max(self.shots, int(f[1:4]) + 1)
            except ValueError:
                pass
        if self.shots:
            print(f"[cam{sensor_id}] {len(prev)} existing shots in {d}, "
                  f"continuing at v{self.shots:03d}")
        self.markers_now = 0
        self.blown = 0.0
        self.contrast = 0.0
        self.fps = 0.0
        self.error: str | None = None
        # Preview-scale cone, and its outline as a contour. Precomputed: resizing a
        # 5 MP mask three times per frame is pure waste at 14 fps.
        self._cone_small = cv2.resize(cone, PREVIEW)
        self._outside = self._cone_small == 0
        self._ring = cv2.findContours(self._cone_small, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._restart.set()

    def reconfigure(self, cfg: dict):
        with self._lock:
            self.cfg.update(cfg)
            self._restart.set()

    def _run(self):
        while not self._stop.is_set():
            self._restart.clear()
            with self._lock:
                pipeline = build_pipeline(self.sensor_id, self.cfg)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.error = (f"could not open Argus for sensor {self.sensor_id}. "
                              f"If the stack is wedged, power-cycle -- restarting "
                              f"nvargus-daemon does not clear it.")
                time.sleep(2.0)
                continue
            self.error = None
            ticks, t0 = 0, time.monotonic()
            while not self._restart.is_set() and not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = f"sensor {self.sensor_id} stopped delivering frames"
                    break
                self._publish(frame)
                ticks += 1
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self.fps, ticks, t0 = ticks / dt, 0, time.monotonic()
            cap.release()

    def _publish(self, frame: np.ndarray):
        # Detect on the HALF-res preview, not the full frame: this is feedback, and
        # markers big enough to calibrate from are still found at half scale for a
        # quarter of the cost. The saved PNG is full res and gets re-detected
        # properly at fit time, so nothing here limits the calibration.
        small = cv2.resize(frame, PREVIEW, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cf.detect_markers(gray, self.dic, self.params)
        self.markers_now = 0 if ids is None else len(ids)
        # Saturation measured over the BOARD, not the frame. Over the frame this is
        # meaningless: the target is a few percent of a mostly dim room, so a board
        # that is 37% clipped reads as 2% and no warning ever fires. Clipped white
        # squares bloom into the black borders, lift black off the floor and gut the
        # edge contrast that subpixel corner refinement depends on -- it is the
        # single largest error source when the target is an emissive screen, and it
        # is invisible in the preview because the image still looks perfectly sharp.
        if self.markers_now:
            pts = np.concatenate([c.reshape(-1, 2) for c in corners])
            x0, y0 = np.clip(pts.min(0), 0, None).astype(int)
            x1, y1 = pts.max(0).astype(int)
            roi = gray[y0:y1 + 1, x0:x1 + 1]
            self.blown = float((roi >= 250).mean()) if roi.size else 0.0
            self.contrast = (float(roi[roi >= roi.mean()].mean()
                                   - roi[roi < roi.mean()].mean())
                             if roi.size > 100 else 0.0)
        else:
            self.blown, self.contrast = 0.0, 0.0

        vis = small
        vis[self._outside] = (vis[self._outside] * 0.4).astype(np.uint8)
        cov = cv2.resize(self.coverage, PREVIEW) > 0
        vis[cov] = (vis[cov] * 0.55 + np.array([0, 90, 0])).clip(0, 255) \
            .astype(np.uint8)
        cv2.drawContours(vis, self._ring, -1, (0, 0, 255), 2)
        if self.markers_now:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids, (0, 255, 255))

        ok, buf = cv2.imencode(".jpg", vis,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        with self._cond:
            self._frame = frame
            if ok:
                self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def wait_jpeg(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._jpeg is None or self._seq <= last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def capture(self) -> dict:
        """Save the current full-res frame if it has enough markers to count."""
        with self._cond:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return {"ok": False, "why": "no frame yet"}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cf.detect_markers(gray, self.dic, self.params)
        n = 0 if ids is None else len(ids)
        if n == 0:
            return {"ok": False, "why": "no markers in the full-res frame"}

        pts = np.concatenate([c.reshape(-1, 2) for c in corners])
        # Only count corners INSIDE the cone toward coverage: the ones outside are
        # what calibrate_fisheye will throw away, so counting them here would
        # report progress that the fit is about to discard.
        inside = 0
        for p in pts:
            u, v = int(round(p[0])), int(round(p[1]))
            if 0 <= u < W and 0 <= v < H and self.cone[v, u]:
                cv2.circle(self.coverage, (u, v), 14, 255, -1)
                inside += 1
        d = os.path.join(self.outdir, f"cam{self.sensor_id}")
        os.makedirs(d, exist_ok=True)
        # PNG not JPEG: at Q95 the ringing on a sharp marker edge is small but it is
        # a BIAS on corner position, and corner position is the whole measurement.
        path = os.path.join(d, f"v{self.shots:03d}.png")
        cv2.imwrite(path, frame)
        self.shots += 1
        return {"ok": True, "path": path, "markers": n, "corners_in_cone": inside}

    def reset(self):
        self.coverage[:] = 0
        self.shots = 0

    def stats(self) -> dict:
        return {"sensor_id": self.sensor_id, "shots": self.shots,
                "markers_now": self.markers_now, "n_markers": self.n_markers,
                "coverage": round(float((self.coverage[self.cone > 0] > 0).mean())
                                  * 100, 1),
                "blown": round(self.blown * 100, 1),
                "contrast": round(self.contrast, 0),
                "exposure_us": self.cfg.get("exposure_us") or 0,
                "gain": self.cfg.get("gain") or 0,
                "fps": round(self.fps, 1), "error": self.error}


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>fisheye calibration capture</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{background:#111;color:#ddd;font:14px system-ui,sans-serif;margin:0;padding:12px}
 h1{font-size:15px;font-weight:600;margin:0 0 10px;color:#fff}
 .row{display:flex;gap:10px;flex-wrap:wrap}
 .cam{flex:1 1 460px;background:#1a1a1a;border-radius:8px;padding:8px}
 img{width:100%;display:block;border-radius:5px;background:#000}
 .st{display:flex;gap:14px;margin-top:7px;font-variant-numeric:tabular-nums;
     flex-wrap:wrap}
 .st b{color:#fff;font-weight:600}
 .bar{height:6px;background:#333;border-radius:3px;margin-top:6px;overflow:hidden}
 .bar i{display:block;height:100%;background:#3c9;width:0}
 .ctl{margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;
        padding:8px 14px;font:inherit;cursor:pointer}
 button:hover{background:#333}
 button.big{background:#0a6;border-color:#0c8;font-weight:600;padding:10px 22px}
 input{background:#222;color:#eee;border:1px solid #444;border-radius:5px;
       padding:6px;font:inherit;width:150px}
 .warn{color:#f85}.ok{color:#3c9}.log{margin-top:10px;font-size:12px;color:#999;
   max-height:110px;overflow:auto;white-space:pre-wrap;font-family:ui-monospace}
 .hint{color:#888;font-size:12px;margin-top:4px}
</style></head><body>
<h1>fisheye calibration capture &mdash; SPACE or Capture to shoot both cameras</h1>
<div class="row" id="cams"></div>
<div class="ctl">
  <button class="big" onclick="shoot()">Capture (space)</button>
  <button onclick="post('/api/reset')">Reset coverage</button>
  <label>exposure us <input id="exp" placeholder="e.g. 3000"></label>
  <label>gain <input id="gain" placeholder="1.0"></label>
  <button onclick="applyCfg()">Apply</button>
  <span class="hint" id="pinned"></span>
</div>
<div class="hint">Green = already covered. Red ring = the valid cone; anything
 outside it cannot be calibrated, so do not spend shots there. Aim for &gt;45%
 coverage with plenty of views out near the ring, each tilted differently.</div>
<div class="log" id="log"></div>
<script>
const cams=[0,1];
document.getElementById('cams').innerHTML=cams.map(i=>`
 <div class="cam"><img src="/stream?cam=${i}" alt="cam${i}">
  <div class="st"><span>cam${i}</span>
   <span>markers <b id="m${i}">-</b></span>
   <span>shots <b id="s${i}">-</b></span>
   <span>cone coverage <b id="c${i}">-</b>%</span>
   <span id="b${i}"></span><span>fps <b id="f${i}">-</b></span></div>
  <div class="bar"><i id="bar${i}"></i></div></div>`).join('');
function log(t){const l=document.getElementById('log');
 l.textContent=t+"\\n"+l.textContent;}
async function post(u){const r=await fetch(u,{method:'POST'});log(await r.text());}
async function shoot(){
 const r=await fetch('/api/capture',{method:'POST'});const j=await r.json();
 log(j.map(x=>`cam${x.sensor_id??'?'}: `+(x.ok?
   `saved ${x.path} (${x.markers} markers, ${x.corners_in_cone} corners in cone)`
   :`SKIPPED - ${x.why}`)).join('   |   '));}
async function applyCfg(){
 const q=new URLSearchParams();
 const e=document.getElementById('exp').value.trim();
 const g=document.getElementById('gain').value.trim();
 if(e)q.set('exposuretimerange',e); if(g)q.set('gainrange',g);
 log(await (await fetch('/api/config?'+q,{method:'POST'})).text());}
{const q=new URLSearchParams(location.search);
 if(q.get('exposure'))document.getElementById('exp').value=q.get('exposure');
 if(q.get('gain'))document.getElementById('gain').value=q.get('gain');}
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();shoot();}});
setInterval(async()=>{
 const j=await (await fetch('/api/status')).json();
 for(const s of j){const i=s.sensor_id;
  document.getElementById('m'+i).textContent=s.markers_now+'/'+s.n_markers;
  document.getElementById('s'+i).textContent=s.shots;
  document.getElementById('c'+i).textContent=s.coverage;
  document.getElementById('f'+i).textContent=s.fps;
  document.getElementById('bar'+i).style.width=Math.min(100,s.coverage/45*100)+'%';
  if(i===0){const pn=document.getElementById('pinned');
   pn.textContent=s.exposure_us? ('pinned: '+s.exposure_us+' us, gain '+s.gain)
                               : 'exposure AUTO - will clip the screen';
   pn.className=s.exposure_us?'hint':'warn';}
  const b=document.getElementById('b'+i);
  b.className=s.error?'warn':(s.blown>2?'warn':'ok');
  b.textContent=s.error?s.error:(s.markers_now?
    (s.blown>2?'BOARD '+s.blown+'% clipped - shorten exposure':
     'board clean, contrast '+s.contrast):'');
 }},700);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    streams: list = []
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/api/capture":
            out = []
            for s in self.streams:
                r = s.capture()
                r["sensor_id"] = s.sensor_id
                out.append(r)
            self._send(200, "application/json", json.dumps(out).encode())
        elif u.path == "/api/reset":
            for s in self.streams:
                s.reset()
            self._send(200, "text/plain", b"coverage reset")
        elif u.path == "/api/config":
            q = parse_qs(u.query)
            cfg = {}
            for key, name in (("exposure", "exposure_us"), ("gain", "gain"),
                              ("dgain", "dgain")):
                if key in q:
                    try:
                        cfg[name] = float(q[key][0])
                    except ValueError:
                        pass
            if "wbmode" in q and q["wbmode"][0] in WB_MODES:
                cfg["wbmode"] = q["wbmode"][0]
            for st in self.streams:
                st.reconfigure(cfg)
            self._send(200, "text/plain",
                       f"reconfigured: {cfg} (pipelines restarting)".encode())
        else:
            self._send(404, "text/plain", b"no")

    def do_GET(self):                                   # noqa: N802
        u = urlparse(self.path)
        if u.path == "/":
            # ?exposure=<us>&gain=<x>&dgain=<x> pins brightness. Applied server-side
            # rather than by page javascript so the URL alone is the whole setting --
            # you can bookmark an exposure, and reloading re-asserts it. Guarded on
            # change because a reconfigure tears both Argus pipelines down and back
            # up, and a browser that reloads on a whim would otherwise keep the
            # cameras permanently restarting.
            q = parse_qs(u.query)
            want = {}
            for key, name in (("exposure", "exposure_us"), ("gain", "gain"),
                              ("dgain", "dgain")):
                if key in q:
                    try:
                        want[name] = float(q[key][0])
                    except ValueError:
                        pass
            if want:
                cur = self.streams[0].cfg
                if any(float(cur.get(k) or 0) != v for k, v in want.items()):
                    for s in self.streams:
                        s.reconfigure(want)
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/api/status":
            self._send(200, "application/json",
                       json.dumps([s.stats() for s in self.streams]).encode())
        elif u.path == "/coverage":
            cam = int(parse_qs(u.query).get("cam", ["0"])[0])
            s = self.streams[cam]
            img = np.where(s.cone > 0, s.coverage, 60).astype(np.uint8)
            ok, buf = cv2.imencode(".png", img)
            self._send(200, "image/png", buf.tobytes() if ok else b"")
        elif u.path == "/stream":
            cam = int(parse_qs(u.query).get("cam", ["0"])[0])
            self._stream(self.streams[cam])
        else:
            self._send(404, "text/plain", b"no")

    def _stream(self, stream: CalibStream):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpeg, seq = stream.wait_jpeg(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " +
                                 str(len(jpeg)).encode() + b"\r\n\r\n" +
                                 jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="calib")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fov", type=float, default=180.0,
                    help="nominal horizontal FOV, used only to draw the cone and to "
                         "measure coverage against the same region the fit will use")
    ap.add_argument("--theta_max", type=float, default=88.0)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--wb", default="daylight", choices=WB_MODES)
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="pin exposure to this many MICROSECONDS (0 = leave auto). "
                         "Overridable live with ?exposure=<us> in the URL.")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="pinned analog gain, only used when --exposure is set")
    ap.add_argument("--dgain", type=float, default=1.0,
                    help="pinned ISP digital gain, ditto")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--no-rotate", action="store_true",
                    help="skip the 180 rotation. sender.py rotates by default and "
                         "the two models must agree.")
    args = ap.parse_args()

    f_nom = (W / 2.0) / np.radians(args.fov / 2.0)
    K_nom = np.array([[f_nom, 0, W / 2.0], [0, f_nom, H / 2.0], [0, 0, 1.0]])
    cone = cf.valid_mask(K_nom, np.zeros(4), (W, H), np.radians(args.theta_max))
    print(f"[cone] nominal HFOV {args.fov:.0f} deg -> f {f_nom:.1f} px/rad, valid to "
          f"{args.theta_max:.0f} deg = {f_nom * np.radians(args.theta_max):.0f} px "
          f"radius, {float((cone > 0).mean()) * 100:.1f}% of the frame")

    cfg = {"wbmode": args.wb, "no_rotate": args.no_rotate,
           "exposure_us": args.exposure, "gain": args.gain, "dgain": args.dgain}
    dic, params = cf.make_dictionary(), cf.detector_params("aruco")
    streams = [CalibStream(sid, cfg, args.outdir, cone, params, dic,
                           args.nx * args.ny, args.quality) for sid in (0, 1)]
    for s in streams:
        s.start()
    Handler.streams = streams

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print(f"[serving] http://<jetson-ip>:{args.port}/   Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop]")
    finally:
        for s in streams:
            s.stop()
        srv.server_close()
    for s in streams:
        print(f"  cam{s.sensor_id}: {s.shots} shots, "
              f"{s.stats()['coverage']}% of the cone covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
