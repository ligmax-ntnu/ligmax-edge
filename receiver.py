#!/usr/bin/env python3
"""Viewer: accepts frames from the Jetson and serves them as a web page.

Run this on the machine the Jetson sends to (192.168.99.135 by default), then
open http://<that machine>:8080/ in a browser.

Two ports:
  3338  TCP, the Jetson connects here and streams frames (protocol.py)
  8080  HTTP, browsers connect here

Boxes are drawn server-side rather than by JavaScript in the browser. That costs a
JPEG decode/re-encode per frame, but this runs on a laptop with CPU to spare rather
than the Jetson, and it means the overlay can never be a frame out of step with the
image -- which client-side drawing would risk. Browsers get a plain
multipart/x-mixed-replace MJPEG stream, so no JS is needed to view it.

Colour correction happens here for the same reason: it is a per-pixel pass the
Jetson has no spare frame budget for, and this end has already paid for the decode.

Requires: pillow, numpy  (pip install pillow numpy)
numpy is only needed for --ccm; without it the correction is skipped with a warning.

  ./receiver.py                        # listen on 0.0.0.0:3338, serve on :8080
  ./receiver.py --http-port 8000 --no-draw
  ./receiver.py --ccm 0.5              # softer colour correction; 0 disables
"""
import argparse
import io
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw

try:
    import numpy as np
    # One sensor, one matrix. fusion.py owns it because that is where sensor
    # pixels are sampled; a frame and a lidar point taken from that same frame
    # must not be corrected differently, or the top-down plot disagrees with the
    # picture it was sampled from. fusion imports numpy, so a box without numpy
    # fails this import too and lands in the same no-correction path as before.
    from fusion import OV5647_CCM as CCM, ccm_strength_for
except ImportError:
    np = None
    CCM = None
    ccm_strength_for = None

import protocol

# Colours per detector class, plus a fallback.
COLOURS = {0: (60, 220, 90), 1: (240, 70, 70), 2: (250, 200, 40)}
FALLBACK = (200, 200, 200)

_TABLES = {}


def ccm_tables(strength):
    """(linearise LUT, matrix, re-encode LUT), built once per strength.

    The linearise table comes out pre-scaled to the re-encode table's index
    range, so the per-pixel path is a gather, a 3x3 matmul and a gather with
    nothing in between -- folding the scale in here and letting np.take clamp
    measured 2x faster than scaling and clipping the pixels separately, for
    bit-identical output.
    """
    if strength not in _TABLES:
        c = np.arange(256, dtype=np.float32) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        m = np.asarray(CCM, dtype=np.float32)
        if strength < 1.0:
            # Blend back toward identity. The off-diagonal terms are large because
            # the crosstalk is, so full strength also amplifies chroma noise.
            m = np.eye(3, dtype=np.float32) * (1.0 - strength) + m * strength
        # 4096 steps, not 256: sRGB encoding expands the shadows, so a table with
        # one entry per output level would band visibly in the dark half.
        g = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
        srgb = np.where(g <= 0.0031308, g * 12.92,
                        1.055 * np.power(g, 1.0 / 2.4) - 0.055)
        out_lut = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
        _TABLES[strength] = ((lin * (out_lut.shape[0] - 1)).astype(np.float32),
                             m, out_lut)
    return _TABLES[strength]


def apply_ccm(im, strength):
    """Colour-correct an RGB PIL image.

    The matrix is defined in linear light, so the frame is linearised first and
    re-encoded after -- applying it straight to gamma-encoded values shifts hues
    instead of just restoring saturation.
    """
    if strength <= 0.0:
        return im          # identity still costs a lossy LUT round-trip
    lin, m, out_lut = ccm_tables(strength)
    # PIL gives RGB and the matrix is in RGB order, so unlike the cv2 version in
    # ../camera-test there is no channel order to undo here.
    a = lin[np.asarray(im, dtype=np.uint8)] @ m.T
    # mode="clip" folds the out-of-gamut clamp into the gather: the off-diagonal
    # terms are large enough to push saturated colours past both ends.
    return Image.fromarray(np.take(out_lut, a.astype(np.int32), mode="clip"))


class Latest:
    """Newest frame per camera, plus a condition so viewers can wait for one."""

    def __init__(self):
        self.cv = threading.Condition()
        self.frames = {}        # cam -> (version, jpeg_bytes, header)
        self.version = 0
        self.stats = {}         # cam -> dict
        self.lidar = None       # newest sweep payload (protocol.KIND_LIDAR)
        self.lidar_n = 0

    def put_lidar(self, sweep):
        with self.cv:
            self.lidar = sweep
            self.lidar_n += 1
            self.version += 1
            self.cv.notify_all()

    def lidar_snapshot(self):
        with self.cv:
            return self.lidar, self.lidar_n

    def put(self, cam, jpeg, header):
        with self.cv:
            self.version += 1
            self.frames[cam] = (self.version, jpeg, header)
            st = self.stats.setdefault(cam, {"count": 0, "t0": time.monotonic(),
                                             "fps": 0.0, "last": 0.0})
            st["count"] += 1
            st["last"] = time.time()
            el = time.monotonic() - st["t0"]
            if el >= 3.0:
                st["fps"] = st["count"] / el
                st["count"] = 0
                st["t0"] = time.monotonic()
            self.cv.notify_all()

    def wait_newer(self, cam, since, timeout=5.0):
        with self.cv:
            end = time.monotonic() + timeout
            while True:
                item = self.frames.get(cam)
                if item and item[0] > since:
                    return item
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cv.wait(left)

    def cameras(self):
        with self.cv:
            return sorted(self.frames)

    def snapshot(self):
        with self.cv:
            return {c: (v[2], dict(self.stats.get(c, {}))) for c, v in self.frames.items()}


LATEST = Latest()


def render(jpeg, header, draw, ccm):
    """Decode, colour-correct, draw boxes and labels, re-encode.

    Returns the frame untouched if there is nothing to do, or if it will not
    decode -- a corrupt frame should cost an overlay, not the stream.

    `ccm` of None means AUTO, which is the default and the one to use: the sender
    boosts chroma in the ISP before the frame ever leaves the Jetson and puts the
    amount in the header, so the strength that belongs here is only whatever that
    did not already do. Pinning --ccm 1.0 against an ISP-boosted frame corrects
    twice and clips 53% of it.
    """
    if ccm is None:
        ccm = (ccm_strength_for(header.get("saturation", 1.0))
               if ccm_strength_for is not None else 0.0)
    if not draw and ccm <= 0.0:
        return jpeg
    try:
        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return jpeg
    # Correct before drawing: the box colours below are display constants and
    # putting them through the matrix would shift them along with the image.
    if ccm > 0.0:
        im = apply_ccm(im, ccm)
    if draw:
        draw_overlay(im, header)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=80)
    return out.getvalue()


def render_lidar(sweep, size=520, max_range=12.0):
    """Top-down plot of one sweep, each return in the colour a camera gave it.

    Top-down rather than an overlay on a camera frame because most of a rotation
    is behind both cameras -- an image overlay can only ever show the third of the
    sweep that a lens covers. Use test/test_lidar_overlay.py on the Jetson for
    that view; it is the one for checking rig.json, and it needs the geometry.

    Rig frame: +x starboard, +z forward, so the plot is +z up and +x right, which
    is the view from above with the bow at the top.

    The point colours arrive ALREADY corrected -- `fusion._correct` runs the
    OV5647 matrix over the sweep on the Jetson, where it costs 0.06 ms for ~400
    points. Do not put `apply_ccm` over them again: correcting twice past-boosts
    saturation and pushes anything already near the gamut edge out of it.
    """
    im = Image.new("RGB", (size, size), (14, 16, 18))
    d = ImageDraw.Draw(im)
    cx = cy = size / 2.0
    scale = (size / 2.0 - 18) / max_range

    for ring in range(2, int(max_range) + 1, 2):
        r = ring * scale
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(44, 50, 56))
        d.text((cx + 3, cy - r - 11), f"{ring} m", fill=(90, 100, 110))
    d.line([cx, 0, cx, size], fill=(44, 50, 56))
    d.line([0, cy, size, cy], fill=(44, 50, 56))
    d.text((cx + 4, 4), "bow", fill=(90, 100, 110))

    if not sweep:
        d.text((10, size - 16), "no sweep yet", fill=(160, 140, 100))
        return im

    xs, zs = sweep.get("x") or [], sweep.get("z") or []
    rgb = sweep.get("rgb") or []
    cam = sweep.get("cam") or []
    det = sweep.get("det") or []
    for i in range(min(len(xs), len(zs))):
        px = cx + xs[i] * scale
        py = cy - zs[i] * scale
        if not (0 <= px < size and 0 <= py < size):
            continue
        c = int(cam[i]) if i < len(cam) else -1
        if c < 0:
            # No camera saw it. Still a real obstacle, so it is drawn -- just in
            # the colour of "the lidar alone knows this is here".
            col = (105, 115, 125)
            rad = 1.4
        else:
            col = tuple(rgb[3 * i:3 * i + 3]) if 3 * i + 2 < len(rgb) else (200, 200, 200)
            rad = 1.8
        if i < len(det) and det[i] >= 0:
            rad = 3.2      # attributed to a detection: this return is a known buoy
            d.ellipse([px - rad - 2, py - rad - 2, px + rad + 2, py + rad + 2],
                      outline=(250, 220, 60))
        d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col)

    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(240, 240, 240))
    skew = sweep.get("skew_ms") or []
    tag = (f"{sweep.get('n', 0)} pts  {sweep.get('coloured', 0)} coloured  "
           f"{sweep.get('hz') or 0:.1f} Hz")
    if skew and skew[0] is not None:
        tag += f"  skew {skew[0]:+.0f} ms"
    d.rectangle([0, size - 15, d.textlength(tag) + 6, size], fill=(0, 0, 0))
    d.text((3, size - 14), tag, fill=(230, 230, 230))
    return im


def draw_overlay(im, header):
    """Draw boxes, labels and the per-camera tag onto an RGB image, in place."""
    net_w = header.get("net_w") or im.width
    net_h = header.get("net_h") or im.height
    # Sender crops to the network aspect and scales uniformly, so one factor per axis.
    sx = im.width / float(net_w)
    sy = im.height / float(net_h)
    d = ImageDraw.Draw(im)

    for det in header.get("dets", []):
        box = det.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        col = COLOURS.get(det.get("cls"), FALLBACK)
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)

        label = f"{det.get('name', '?')} {det.get('conf', 0):.2f}"
        if det.get("id") is not None:
            label = f"#{det['id']} {label}"
        if det.get("card"):
            label += f" [{det['card']} {det.get('card_conf', 0):.2f}]"
        tw = d.textlength(label)
        ty = max(0, y1 - 12)
        d.rectangle([x1, ty, x1 + tw + 4, ty + 12], fill=col)
        d.text((x1 + 2, ty), label, fill=(0, 0, 0))

    tag = (f"cam{header.get('cam')}  {header.get('fps', 0):.1f} fps  "
           f"{len(header.get('dets', []))} det")
    d.rectangle([0, 0, d.textlength(tag) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), tag, fill=(255, 255, 255))


def ingest(conn, addr, draw, ccm):
    print(f"[recv] jetson connected from {addr[0]}:{addr[1]}", flush=True)
    n = 0
    try:
        while True:
            msg = protocol.read_message(conn)
            if msg is None:
                break
            header, jpeg = msg
            # Dispatch on kind BEFORE reading `cam`. A lidar message has no `cam`,
            # and defaulting it to 0 would file an empty payload as camera 0 and
            # blank that feed at the sweep rate.
            if header.get("kind") == protocol.KIND_LIDAR:
                LATEST.put_lidar(header.get("lidar"))
                continue
            cam = int(header.get("cam", 0))
            jpeg = render(jpeg, header, draw, ccm)
            LATEST.put(cam, jpeg, header)
            n += 1
    except (OSError, ValueError) as e:
        print(f"[recv] {addr[0]} error: {e}", flush=True)
    finally:
        try:
            conn.close()
        except OSError:
            pass
        print(f"[recv] {addr[0]} disconnected after {n} frames", flush=True)


def ingest_server(host, port, draw, ccm):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    print(f"[recv] listening for the Jetson on {host}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=ingest, args=(conn, addr, draw, ccm),
                         daemon=True).start()


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Buoy detector live view</title>
<style>
  :root { color-scheme: dark light; }
  body { margin:0; background:#111; color:#eee;
         font:14px/1.4 system-ui,-apple-system,sans-serif; }
  header { padding:10px 14px; background:#000; display:flex; gap:16px;
           align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  #status { color:#9a9; font-variant-numeric:tabular-nums; }
  .wrap { display:flex; flex-wrap:wrap; gap:10px; padding:10px; }
  figure { margin:0; flex:1 1 480px; min-width:320px; }
  figure.lidar { flex:0 1 380px; }
  figcaption { padding:4px 2px; color:#9a9; font-size:12px; }
  img { width:100%; height:auto; display:block; background:#000; border-radius:4px; }
</style>
<header>
  <h1>Buoy detector &mdash; live</h1>
  <span id="status">connecting&hellip;</span>
</header>
<div class="wrap" id="wrap"></div>
<script>
function build(cams) {
  const wrap = document.getElementById('wrap');
  wrap.innerHTML = '';
  if (!cams.length) {
    wrap.innerHTML = '<p style="padding:12px;color:#a88">'
      + 'No cameras yet. Waiting for the Jetson to connect&hellip;</p>';
    return;
  }
  for (const c of cams) {
    const f = document.createElement('figure');
    f.innerHTML = '<img src="/cam' + c + '/stream?t=' + Date.now() + '" alt="camera ' + c + '">'
                + '<figcaption>camera ' + c + '</figcaption>';
    wrap.appendChild(f);
  }
}
// The lidar tile is polled rather than streamed: one sweep a frame is plenty to
// look at, and it keeps the MJPEG connections for the cameras alone.
let lidarTile = null;
function showLidar(on) {
  if (on && !lidarTile) {
    lidarTile = document.createElement('figure');
    lidarTile.className = 'lidar';
    lidarTile.innerHTML = '<img id="lidarimg" alt="lidar, top down">'
      + '<figcaption>lidar &mdash; top down, bow up, colour from the cameras</figcaption>';
    document.getElementById('wrap').appendChild(lidarTile);
    setInterval(() => {
      const el = document.getElementById('lidarimg');
      if (el) el.src = '/lidar.jpg?t=' + Date.now();
    }, 400);
  }
}
let shown = '';
async function poll() {
  try {
    const r = await fetch('/api/status', {cache:'no-store'});
    const j = await r.json();
    const key = j.cameras.join(',');
    if (key !== shown) { shown = key; build(j.cameras); lidarTile = null; }
    if (j.lidar) showLidar(true);
    let s = j.cameras.length
      ? j.cameras.map(c => 'cam'+c+': '+(j.fps[c]||0).toFixed(1)+' fps, '
          + (j.dets[c]||0) + ' det').join('   |   ')
      : 'waiting for the Jetson';
    if (j.lidar) s += '   |   lidar: ' + j.lidar.n + ' pts, '
      + j.lidar.coloured + ' coloured, ' + (j.lidar.hz||0).toFixed(1) + ' Hz';
    document.getElementById('status').textContent = s;
  } catch (e) {
    document.getElementById('status').textContent = 'viewer unreachable';
  }
  setTimeout(poll, 1000);
}
poll();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # one line per MJPEG frame would be unreadable

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"))
            return
        if path == "/api/status":
            import json
            snap = LATEST.snapshot()
            sweep, seen = LATEST.lidar_snapshot()
            body = json.dumps({
                "cameras": sorted(snap),
                "fps": {c: round(st.get("fps", 0.0), 2) for c, (_, st) in snap.items()},
                "dets": {c: len(hdr.get("dets", [])) for c, (hdr, _) in snap.items()},
                "detail": {c: hdr.get("dets", []) for c, (hdr, _) in snap.items()},
                "lidar": None if not sweep else {
                    "sweeps": seen, "n": sweep.get("n"),
                    "coloured": sweep.get("coloured"), "hz": sweep.get("hz"),
                    "skew_ms": sweep.get("skew_ms"),
                },
            }).encode()
            self._send(body, "application/json")
            return
        if path == "/api/lidar":
            import json
            sweep, _ = LATEST.lidar_snapshot()
            self._send(json.dumps(sweep).encode(), "application/json")
            return
        if path == "/lidar.jpg":
            sweep, _ = LATEST.lidar_snapshot()
            out = io.BytesIO()
            render_lidar(sweep).save(out, format="JPEG", quality=85)
            self._send(out.getvalue(), "image/jpeg")
            return
        if path.startswith("/cam") and path.endswith("/stream"):
            try:
                cam = int(path[4:path.index("/", 4)])
            except (ValueError, IndexError):
                self._send(b"bad camera", "text/plain", 400)
                return
            self.stream(cam)
            return
        if path.startswith("/cam") and path.endswith(".jpg"):
            try:
                cam = int(path[4:-4])
            except ValueError:
                self._send(b"bad camera", "text/plain", 400)
                return
            item = LATEST.frames.get(cam)
            if not item:
                self._send(b"no frame yet", "text/plain", 404)
                return
            self._send(item[1], "image/jpeg")
            return
        self._send(b"not found", "text/plain", 404)

    def stream(self, cam):
        boundary = "buoyframe"
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = 0
        try:
            while True:
                item = LATEST.wait_newer(cam, seen, timeout=10.0)
                if item is None:
                    continue        # keep the connection open through a quiet spell
                seen, jpeg, _ = item
                self.wfile.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0", help="interface for the Jetson feed")
    ap.add_argument("--port", type=int, default=3338, help="TCP port the Jetson sends to")
    ap.add_argument("--http-port", type=int, default=8080, help="port browsers use")
    ap.add_argument("--no-draw", action="store_true",
                    help="serve frames unannotated (boxes still on /api/status)")
    ap.add_argument("--ccm", type=float, default=None,
                    help="OV5647 colour-correction strength. The default is AUTO: "
                         "each frame says how much chroma the Jetson's ISP already "
                         "applied and only the remainder is done here, which is "
                         "what stops a boosted frame being corrected twice. Give a "
                         "number to override -- 0 off, 1.0 full -- but 1.0 against "
                         "an ISP-boosted frame clips about half of it.")
    args = ap.parse_args()

    ccm = None if args.ccm is None else max(0.0, args.ccm)
    if ccm != 0.0 and np is None:
        print("[warn] numpy not installed, so colour correction is off and colours "
              "will look washed out; pip install numpy", flush=True)
        ccm = 0.0

    threading.Thread(target=ingest_server,
                     args=(args.bind, args.port, not args.no_draw, ccm),
                     daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
    httpd.daemon_threads = True
    print(f"[http] open http://<this-machine>:{args.http_port}/ in a browser",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
