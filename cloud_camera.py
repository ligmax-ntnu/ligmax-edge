"""Push a small camera picture to the operator dashboard, when asked to.

This is the *only* thing the Jetson sends straight to `live.ligmax.no`. The split
matters and it is not the obvious one:

    detections + front lidar  --TCP--> ligmax-pi3.local
                                       the Pi merges them with the aft lidar and
                                       sends one fused world model up the telemetry
                                       link, so the map shows one coherent picture
                                       instead of two that disagree.

    preview JPEG              --HTTPS--> live.ligmax.no /api/camera
                                       straight to shore, because a picture is not
                                       part of the world model and routing it
                                       through the Pi would cost a hop for nothing.

Why it is off by default
------------------------
The vessel is on 4G. The uplink is shared with the telemetry frames, the log
stream and the channel operator commands come *back* down - including the E-stop
ack. Video is the only payload big enough to crowd those out, so nothing is sent
until an operator switches it on from the dashboard, and what gets sent is
deliberately small: the server hands out the size, quality and frame rate
(`ligmax-server/ligmax_gui/camera.py` DEFAULT_STREAM, currently 480 px wide, q55,
2 fps - roughly 0.3 Mbit/s).

Nothing here ever accepts an inbound connection, which is what makes it work
behind carrier NAT. The config is *polled*: an outbound GET every few seconds,
and the reply to each frame POST carries the config too, so a change to fps takes
effect on the next frame rather than the next poll.

Design rules, because this runs in the capture loop's process:

  * `submit()` never blocks and never raises. It drops the frame into a
    latest-wins slot and returns; one worker thread does the encoding and the
    POST. A stalled 4G link cannot back-pressure the detector.
  * Latest frame wins, always. A preview superseded before it went out is of no
    interest - unlike a log line, which queues.
  * While the stream is off, `submit()` is a couple of comparisons. It stays
    wired in at all times so switching video on needs no restart of the detector.

Usage, from sender.py:

    uplink = CameraUplink.from_env()          # or CameraUplink(url, key)
    ...
    uplink.submit(cam_index, jpeg_bytes, width, height, t_capture)
    ...
    uplink.close()
"""

from __future__ import annotations

import http.client
import io
import json
import os
import ssl
import threading
import time
import urllib.parse

DEFAULT_URL = "https://live.ligmax.no"

# Sent on EVERY request, not just the frame POST. Cloudflare fronts the
# dashboard and refuses anything that looks like a stock library client - see
# `_headers()` and the same note in `update.py`.
UA = "ligmax-edge/cloud_camera"

# How often to ask the dashboard what it wants, while nothing is being sent. The
# reply to every frame POST also carries the config, so once video is running this
# poll is only a keepalive that tells the panel the Jetson is listening at all.
POLL_PERIOD = 5.0
# Backoff after a network error. A 4G dropout is measured in seconds and there is
# nothing useful to do but wait.
ERROR_BACKOFF = 3.0
REQUEST_TIMEOUT = 8.0
IDLE_TICK = 0.25

# Refuse to encode above this regardless of what the server asks for. The server
# clamps too; this is the second half of the same guard, on the side that would
# actually spend the bandwidth.
MAX_WIDTH = 1280
MAX_FPS = 10.0

try:
    from PIL import Image
except ImportError:  # the viewer host may have it and the Jetson may not
    Image = None


class CameraUplink:
    """Latest-frame-wins JPEG uplink to the dashboard, with a polled config.

    One worker thread, one in-flight POST, one pending frame per camera.
    """

    def __init__(self, target: str = DEFAULT_URL, key: str | None = None,
                 verify_tls: bool = True) -> None:
        parsed = urllib.parse.urlparse(
            target if "://" in target else f"https://{target}")
        self.scheme = (parsed.scheme or "https").lower()
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.key = (key or "").strip() or None
        self.verify_tls = verify_tls

        # What the dashboard has asked for. Assume off until it says otherwise -
        # the expensive default is never the one we start with.
        self.stream = {
            "enabled": False,
            "max_width": 480,
            "jpeg_quality": 55,
            "fps": 2.0,
            "cameras": ["0", "1"],
        }

        self.sent = 0
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_config_at = 0.0

        self._pending: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._connection = None
        self._next_send: dict[str, float] = {}
        self._last_poll = 0.0
        self._last_config_status: int | None = None
        self._warned_no_pil = False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="camera-uplink")
        self._thread.start()

    @classmethod
    def from_env(cls, **kwargs) -> "CameraUplink":
        """Build from the same variables the rest of the fleet already uses.

        ``LIGMAX_UPLOAD_URL``, else ``LIGMAX_DEPLOY_URL`` (already in
        /etc/ligmax/node.env), else ``https://live.ligmax.no``. The secret is
        ``LIGMAX_BOAT_KEY`` - the same one the Pi's telemetry uses, because this is
        the same trust boundary: whoever can push frames could push telemetry.
        ``LIGMAX_UPLOAD_INSECURE=1`` skips certificate verification, for a LAN test
        straight at Caddy whose origin certificate only Cloudflare trusts.
        """
        target = (os.environ.get("LIGMAX_CAMERA_URL")
                  or os.environ.get("LIGMAX_UPLOAD_URL")
                  or os.environ.get("LIGMAX_DEPLOY_URL")
                  or DEFAULT_URL)
        insecure = os.environ.get("LIGMAX_UPLOAD_INSECURE", "").strip().lower()
        kwargs.setdefault("verify_tls", insecure not in ("1", "true", "yes", "on"))
        return cls(target, os.environ.get("LIGMAX_BOAT_KEY"), **kwargs)

    # -- public API ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.stream.get("enabled"))

    def submit(self, cam, jpeg: bytes, width: int, height: int,
               t_capture: float | None = None) -> None:
        """Offer a preview JPEG. Never blocks, never raises.

        Cheap to call while the stream is off, which is why it can sit
        unconditionally in the capture loop.
        """
        if self._closed or not jpeg:
            return
        stream = self.stream
        if not stream.get("enabled"):
            return

        camera = str(cam)
        if camera not in (stream.get("cameras") or ()):
            return

        # Rate-limit here rather than in the worker: dropping a frame before it is
        # copied and re-encoded is the cheap place to drop it.
        now = time.monotonic()
        fps = min(float(stream.get("fps") or 2.0), MAX_FPS)
        interval = 1.0 / max(fps, 0.05)
        if now < self._next_send.get(camera, 0.0):
            return
        self._next_send[camera] = now + interval

        with self._lock:
            if camera in self._pending:
                # Superseded before it went out. Counted, because a high drop rate
                # against a low fps means the uplink cannot keep up and the
                # operator should ask for less.
                self.dropped += 1
            self._pending[camera] = (jpeg, width, height, t_capture or time.time())
        self._wake.set()

    def stats(self) -> dict:
        """One line for sender.py's periodic stats print."""
        return {
            "enabled": self.enabled,
            # Has the dashboard ever answered a config poll? "Off" with this
            # False means we are being refused, not that nobody wants video -
            # the one distinction worth having in the journal.
            "config_ok": self.last_config_at > 0.0,
            "sent": self.sent,
            "dropped": self.dropped,
            "errors": self.errors,
            "asked": (
                f"{self.stream.get('max_width')}px "
                f"q{self.stream.get('jpeg_quality')} "
                f"{self.stream.get('fps')}fps"
            ),
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        self._thread.join(2.0)
        self._drop_connection()

    # -- worker -------------------------------------------------------------

    def _run(self) -> None:
        while True:
            self._wake.clear()
            if self._closed:
                self._drop_connection()
                return

            now = time.monotonic()
            # Poll for the config when nothing is going out. Once frames are
            # flowing their replies carry it, so this stops costing a request.
            if now - self._last_poll >= POLL_PERIOD:
                self._last_poll = now
                self._poll_config()

            with self._lock:
                pending = self._pending
                self._pending = {}
            for camera, frame in pending.items():
                if self._closed:
                    break
                self._post(camera, *frame)

            if not pending:
                self._wake.wait(IDLE_TICK)

    def _encode(self, jpeg: bytes, width: int, height: int):
        """Downscale and re-encode to what the server asked for.

        The Jetson's preview branch already produced a JPEG on the hardware
        encoder, but at the pipeline's own size and quality (`--preview`,
        `--quality`). Re-encoding on the CPU at this size costs a couple of
        milliseconds and is worth it, because the alternative is either sending
        several times the bytes or letting a remote config reach into the
        GStreamer pipeline - which would mean tearing down capture to change a
        slider, and Argus does not survive that gracefully (`run.sh`).

        Without Pillow, the original JPEG is sent unchanged. Honest, and bigger
        than asked for, so it says so once in the log.
        """
        target = min(int(self.stream.get("max_width") or 480), MAX_WIDTH)
        quality = int(self.stream.get("jpeg_quality") or 55)

        if Image is None:
            if not self._warned_no_pil:
                self._warned_no_pil = True
                print("cloud_camera: Pillow not installed, sending the preview "
                      "JPEG unscaled - install pillow to honour the size the "
                      "dashboard asks for", flush=True)
            return jpeg, width, height

        if width and width <= target:
            # Already small enough. Re-encoding it would only lose quality.
            return jpeg, width, height

        try:
            image = Image.open(io.BytesIO(jpeg))
            image.load()
            scale = target / float(image.width)
            size = (target, max(1, int(round(image.height * scale))))
            # BILINEAR, not LANCZOS: this is a 480 px preview on a phone-sized
            # panel and the sharper filter costs more than it shows.
            small = image.resize(size, Image.BILINEAR)
            out = io.BytesIO()
            small.save(out, format="JPEG", quality=quality, optimize=False)
            return out.getvalue(), size[0], size[1]
        except Exception as exc:  # noqa: BLE001 - a bad frame is not fatal
            self.last_error = f"re-encode failed: {exc}"
            self.errors += 1
            return jpeg, width, height

    def _headers(self, **extra: str) -> dict:
        """Headers every request here must carry.

        The User-Agent is not decoration. Cloudflare fronts live.ligmax.no and
        403s requests with a default-looking or absent agent (error 1010) - the
        same trap that left `update.py` reading "Never polled" on the dashboard
        while it had been polling all along, and `http.client` sends no
        User-Agent at all unless told to. This used to be set on the frame POST
        only, so the config GET was refused, `enabled` never went true, and no
        frame was ever offered: no picture, and the panel blamed the Jetson for
        not asking.
        """
        headers = {
            "Connection": "keep-alive",
            "User-Agent": UA,
            **extra,
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def _post(self, camera: str, jpeg: bytes, width: int, height: int,
              t_capture: float) -> None:
        payload, out_w, out_h = self._encode(jpeg, width, height)
        query = urllib.parse.urlencode({
            "cam": camera,
            "t": f"{t_capture:.3f}",
            "width": out_w,
            "height": out_h,
            "label": f"cam{camera}",
        })
        path = f"{self.base_path}/api/camera?{query}"
        headers = self._headers(**{"Content-Type": "image/jpeg"})

        # Two attempts: a kept-alive socket the far end has since closed fails on
        # the write, and that failure says nothing about the next try.
        for attempt in (1, 2):
            connection = self._get_connection()
            if connection is None:
                return
            try:
                connection.request("POST", path, body=payload, headers=headers)
                response = connection.getresponse()
                body = response.read()  # must drain before the socket is reused
            except (http.client.HTTPException, OSError, ssl.SSLError) as exc:
                self._drop_connection()
                if attempt == 2:
                    self.errors += 1
                    self.last_error = str(exc) or exc.__class__.__name__
                    time.sleep(ERROR_BACKOFF)
                continue

            if response.status == 200:
                self.sent += 1
                self.last_error = None
                self._absorb(body)
            else:
                self.errors += 1
                self.last_error = f"HTTP {response.status} {body[:120]!r}"
                if response.status in (401, 403):
                    # A wrong LIGMAX_BOAT_KEY will not become right by retrying.
                    self.stream["enabled"] = False
                    print("cloud_camera: rejected by the dashboard - check "
                          "LIGMAX_BOAT_KEY. Video stays off.", flush=True)
                    time.sleep(ERROR_BACKOFF)
            return

    def _poll_config(self) -> None:
        connection = self._get_connection()
        if connection is None:
            return
        try:
            connection.request(
                "GET", f"{self.base_path}/api/camera/config",
                headers=self._headers())
            response = connection.getresponse()
            body = response.read()
        except (http.client.HTTPException, OSError, ssl.SSLError) as exc:
            self._drop_connection()
            self.last_error = str(exc) or exc.__class__.__name__
            return
        if response.status == 200:
            self._absorb(body)
        else:
            # Say which refusal it is. 401/403 here is the poll being turned
            # away - almost always no LIGMAX_BOAT_KEY in /etc/ligmax/node.env,
            # or Cloudflare - and from the dashboard it looks identical to a
            # Jetson that is not running at all, so the log is the only place
            # the difference shows.
            self.errors += 1
            self.last_error = f"config HTTP {response.status}"
            if response.status in (401, 403):
                self.last_error += (
                    " - the dashboard refused the config poll; check "
                    "LIGMAX_BOAT_KEY is set on this board"
                )
            if response.status != self._last_config_status:
                print(f"cloud_camera: {self.last_error}", flush=True)
        self._last_config_status = response.status

    def _absorb(self, body: bytes) -> None:
        """Take the config out of a reply. Every reply carries it."""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        changed = False
        for key in ("enabled", "max_width", "jpeg_quality", "fps", "cameras"):
            if key in payload and payload[key] != self.stream.get(key):
                self.stream[key] = payload[key]
                changed = True
        self.last_config_at = time.monotonic()
        if changed:
            print(f"cloud_camera: dashboard asks for {self.stats()['asked']}, "
                  f"{'streaming' if self.enabled else 'off'}", flush=True)

    def _get_connection(self):
        if self._connection is None:
            try:
                if self.scheme == "https":
                    context = ssl.create_default_context()
                    if not self.verify_tls:
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                    self._connection = http.client.HTTPSConnection(
                        self.host, self.port, timeout=REQUEST_TIMEOUT,
                        context=context)
                else:
                    self._connection = http.client.HTTPConnection(
                        self.host, self.port, timeout=REQUEST_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                return None
        return self._connection

    def _drop_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


if __name__ == "__main__":
    # Smoke test:  python cloud_camera.py
    # Sends one tiny frame if - and only if - an operator has switched the stream
    # on from the dashboard. Prints what the server is asking for either way.
    uplink = CameraUplink.from_env()
    print(f"target {uplink.scheme}://{uplink.host}:{uplink.port}, "
          f"authenticated={uplink.key is not None}")
    time.sleep(1.5)
    print("dashboard config:", json.dumps(uplink.stream))
    if uplink.enabled:
        red = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
            "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
            "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
            "ffffffc00011080001000101011100ffc40014000100000000000"
            "000000000000000000009ffda0008010100003f00d2cf20ffd9")
        uplink.submit("0", red, 1, 1)
        time.sleep(1.5)
    print("stats:", json.dumps(uplink.stats()))
    uplink.close()
