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

The boxes are burned into the picture here, unlike everywhere else. `receiver.py`
gets the detections as JSON beside the frame and draws them itself; the dashboard
gets a JPEG and nothing else, because the detections took the other route entirely
and reach the operator as objects on the map. So a panel with no overlay is a panel
that cannot show what the detector is seeing, only what the lens is - and the two
disagreeing is exactly the case worth looking at. `--no-cloud-boxes` turns it off.

Full-resolution stills, which are the opposite of everything above
------------------------------------------------------------------
The stream is small, continuous, latest-wins and lossy. A **still** is one whole
2592x1944 sensor frame per camera, on request, and every rule above is inverted
for it:

  * it is **queued, not latest-wins** - somebody pressed a button and is waiting;
  * it is **2-3 MB**, i.e. ten seconds of the stream's entire budget, so it only
    ever happens when asked for;
  * it works **while the stream is off**, which is the normal case for it: the
    reason to want one is to go and photograph the AR tags on the dock, and
    nobody wants video running for that;
  * it is **uncropped**, because the detector's 2:1 band is aimed off the bow and
    the markers are not there (`rig.json`: the cameras look port and starboard).

The request arrives the only way anything can reach this board - as a field on
the config poll's reply. `wanted_still()` is what `sender.py` checks, and it
answers with the cameras still outstanding, so a second camera that is slow does
not make the first one send twice. See `ligmax-server/ligmax_gui/stills.py` for
the shore half.

Usage, from sender.py:

    uplink = CameraUplink.from_env()          # or CameraUplink(url, key)
    ...
    uplink.submit(cam_index, jpeg_bytes, width, height, t_capture,
                  dets=items, det_size=(net_w, net_h))

    if (still := uplink.wanted_still()) is not None:
        # ... grab the full-res NV12 for the cameras in still["cameras"] ...
        uplink.submit_still(cam_index, nv12_bytes, w, h, t_capture, still,
                            meta={"mode": 0, "rotated_180": True})
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

# A still is 2-3 MB and 4G upstream is a couple of Mbit/s on a good day, so one
# can take ten seconds. Its own timeout, well clear of the frame one: giving up
# halfway would waste the whole upload and leave the operator pressing again.
STILL_TIMEOUT = 90.0
# How many stills may be waiting to go out. Two is one press of the button; four
# leaves room for a second press while the first pair is still climbing the link.
STILL_QUEUE = 4

# Refuse to encode above this regardless of what the server asks for. The server
# clamps too; this is the second half of the same guard, on the side that would
# actually spend the bandwidth.
MAX_WIDTH = 1280
MAX_FPS = 10.0

# Deliberately the same values as receiver.py's COLOURS, keyed by detector class.
# The bench viewer and the dashboard show the same scene, and a buoy that is green
# on one screen and yellow on the other costs more confusion than the duplication
# costs here. Not imported from receiver.py: that module is what runs on the
# *viewer* host and importing it would drag its HTTP server onto the Jetson.
BOX_COLOURS = {0: (60, 220, 90), 1: (240, 70, 70), 2: (250, 200, 40)}
BOX_FALLBACK = (200, 200, 200)

try:
    from PIL import Image, ImageDraw
except ImportError:  # the viewer host may have it and the Jetson may not
    Image = ImageDraw = None

try:
    import numpy as np
except ImportError:  # only the still path needs it; the stream does not
    np = None


class CameraUplink:
    """Latest-frame-wins JPEG uplink to the dashboard, with a polled config.

    One worker thread, one in-flight POST, one pending frame per camera.
    """

    def __init__(self, target: str = DEFAULT_URL, key: str | None = None,
                 verify_tls: bool = True, draw_boxes: bool = True) -> None:
        parsed = urllib.parse.urlparse(
            target if "://" in target else f"https://{target}")
        self.scheme = (parsed.scheme or "https").lower()
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.key = (key or "").strip() or None
        self.verify_tls = verify_tls
        # Burn the detector's boxes into the frame. The dashboard gets a picture
        # and nothing else - the detections themselves go to the Pi and reach the
        # operator as map objects - so without this the one thing the panel cannot
        # show is what the detector is actually seeing. Drawn in the worker thread,
        # after the downscale, at 2 fps: see `_overlay`.
        self.draw_boxes = draw_boxes and ImageDraw is not None

        # What the dashboard has asked for. Assume off until it says otherwise -
        # the expensive default is never the one we start with.
        self.stream = {
            "enabled": False,
            "max_width": 480,
            "jpeg_quality": 55,
            "fps": 2.0,
            "cameras": ["0", "1"],
            # Whether the YOLO detector should run. Unlike everything else here
            # this is not about the uplink at all - it is the dashboard's switch
            # for the *inference* in `sender.py`, and it lives on this poll
            # because this poll is the only channel that reaches this board.
            #
            # Default True so a board that never hears from the dashboard behaves
            # exactly as it did before this existed. `sender.py --no-detect` is
            # the other half: that one never loads an engine at all, and no
            # dashboard toggle can turn inference back on in that process.
            "detect": True,
        }

        self.sent = 0
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_config_at = 0.0

        # The outstanding full-resolution capture, as the dashboard describes it,
        # and which cameras have already been handed over for it. Kept out of
        # `self.stream` on purpose: `stats()['asked']` is about the *stream*, and
        # a still is not a stream setting.
        self.capture: dict | None = None
        self.stills_sent = 0
        self.stills_failed = 0

        self._pending: dict[str, tuple] = {}
        # Stills queue rather than overwrite - see the module docstring.
        self._stills: list[tuple] = []
        self._still_id = 0
        self._still_taken: set[str] = set()
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

    @property
    def detect(self) -> bool:
        """Whether the dashboard wants the detector running. See `stream`.

        True when it has never said - a board that cannot reach shore keeps
        detecting, because that is the mode the boat races in.
        """
        return bool(self.stream.get("detect", True))

    def submit(self, cam, jpeg: bytes, width: int, height: int,
               t_capture: float | None = None, dets=None, det_size=None) -> None:
        """Offer a preview JPEG, and the boxes to burn into it. Never blocks.

        Cheap to call while the stream is off, which is why it can sit
        unconditionally in the capture loop.

        `dets` is `per_cam[i]` straight out of the detector and `det_size` is
        `(net_w, net_h)`, the space its `box` coordinates live in. They are turned
        into fractions of the frame here and drawn in the worker thread - see
        `_overlay`. Nothing is copied or converted until after the rate gate below,
        so a frame that is about to be dropped costs nothing extra.
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

        # Past the gate, so this runs at the stream's rate (2 fps) rather than the
        # detector's, and on a handful of detections. Done here rather than in the
        # worker because `dets` belongs to the caller's frame: it is rebuilt every
        # iteration, and holding a reference across the queue would mean reading it
        # while the next frame is being built.
        boxes = self._normalise(dets, det_size) if self.draw_boxes else ()

        with self._lock:
            if camera in self._pending:
                # Superseded before it went out. Counted, because a high drop rate
                # against a low fps means the uplink cannot keep up and the
                # operator should ask for less.
                self.dropped += 1
            self._pending[camera] = (jpeg, width, height,
                                     t_capture or time.time(), boxes)
        self._wake.set()

    @staticmethod
    def _normalise(dets, det_size) -> tuple:
        """Detections -> `(x1, y1, x2, y2, colour, label)` in 0..1 fractions.

        Fractions, not pixels, because the server owns the output size: `max_width`
        is a slider on the dashboard and can change between this call and the
        encode. A fraction survives that; a pixel coordinate would silently scale
        the boxes off the buoys the first time somebody moved it.
        """
        if not dets or not det_size:
            return ()
        net_w, net_h = det_size
        if not net_w or not net_h:
            return ()
        out = []
        for det in dets:
            box = det.get("box") or ()
            if len(box) != 4:
                continue
            colour = BOX_COLOURS.get(det.get("cls"), BOX_FALLBACK)
            # The box colour already says green/red/cardinal, so the name would be
            # redundant on a 480 px tile. What it cannot say is *which* cardinal,
            # and that is the one thing an operator has to read off the picture.
            label = f"{det.get('conf', 0.0):.2f}"
            if det.get("card"):
                label = f"{det['card']} {label}"
            out.append((box[0] / net_w, box[1] / net_h,
                        box[2] / net_w, box[3] / net_h, colour, label))
        return tuple(out)

    # -- full-resolution stills --------------------------------------------

    def wanted_still(self) -> dict | None:
        """The outstanding capture, or None. Cheap enough for the capture loop.

        Returns `{"id", "cameras", "quality"}` where `cameras` is the ids this
        board has **not yet handed over** for that request - so a caller can
        submit them one at a time as frames become available, and a camera whose
        upload is still climbing the link is not asked for again.

        The dashboard also filters by what it has *received*, and the two
        together are what make this safe against either end restarting: shore
        forgets a completed request, and this forgets a superseded one.
        """
        capture = self.capture
        if not capture:
            return None
        cameras = [
            camera for camera in capture.get("cameras") or ()
            if str(camera) not in self._still_taken
        ]
        if not cameras:
            return None
        return {
            "id": capture.get("id"),
            "cameras": cameras,
            "quality": int(capture.get("quality") or 92),
        }

    def submit_still(self, cam, nv12: bytes, width: int, height: int,
                     t_capture: float | None = None, request: dict | None = None,
                     meta: dict | None = None) -> bool:
        """Hand over one full-resolution NV12 frame. Never blocks, never raises.

        **NV12 in, JPEG out, and the conversion happens in the worker thread.**
        A 5 megapixel colour conversion plus a JPEG encode is a third of a second
        on this board's CPU - four whole frame periods at mode 0 - so doing it
        where `submit()` is called would stall capture visibly and, worse,
        intermittently. What the caller pays is a memcpy of the buffer it already
        has.

        Returns whether the frame was taken. False means the queue is full or
        this camera is already in it for this request, which is not an error:
        the request stays outstanding and the next frame will do.
        """
        if self._closed or not nv12:
            return False
        camera = str(cam)
        wanted = request or self.capture or {}
        request_id = int(wanted.get("id") or 0)
        quality = int(wanted.get("quality") or 92)

        with self._lock:
            if request_id != self._still_id:
                # A new request supersedes an unfinished one outright. The old
                # frames are of a moment nobody is waiting for any more.
                self._still_id = request_id
                self._still_taken = set()
                self._stills.clear()
            if camera in self._still_taken:
                return False
            if len(self._stills) >= STILL_QUEUE:
                return False
            self._still_taken.add(camera)
            self._stills.append((camera, nv12, int(width), int(height),
                                 t_capture or time.time(), request_id, quality,
                                 dict(meta or {})))
        self._wake.set()
        return True

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
            "stills": self.stills_sent,
            "stills_failed": self.stills_failed,
            # Truthy only while a capture is outstanding, so the stats line can
            # say "capturing" for the few seconds it takes and nothing after.
            "capturing": bool(self.wanted_still()),
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

            # Stills first, and by a wide margin: somebody is waiting for one,
            # while a preview frame is one of several a second and the next will
            # do. A still also takes seconds, so anything queued behind it waits
            # anyway - better that it be a frame that was going to be superseded.
            with self._lock:
                stills, self._stills = self._stills, []
            for entry in stills:
                if self._closed:
                    break
                self._post_still(*entry)

            with self._lock:
                pending = self._pending
                self._pending = {}
            for camera, frame in pending.items():
                if self._closed:
                    break
                self._post(camera, *frame)

            if not pending and not stills:
                self._wake.wait(IDLE_TICK)

    def _encode(self, jpeg: bytes, width: int, height: int, boxes=()):
        """Downscale, draw the boxes, and re-encode to what the server asked for.

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

        if width and width <= target and not boxes:
            # Already small enough, and nothing to draw on it. Re-encoding it
            # would only lose quality. With boxes there is no such shortcut - they
            # have to be burned in, so the decode happens either way.
            return jpeg, width, height

        try:
            image = Image.open(io.BytesIO(jpeg))
            image.load()
            if image.width > target:
                scale = target / float(image.width)
                size = (target, max(1, int(round(image.height * scale))))
                # BILINEAR, not LANCZOS: this is a 480 px preview on a phone-sized
                # panel and the sharper filter costs more than it shows.
                image = image.resize(size, Image.BILINEAR)
            if boxes:
                # After the resize, never before: drawing at 640x320 and then
                # scaling down would thin the 2 px outlines to something under a
                # pixel and resample the labels into mush. This way every stroke is
                # laid down at the size it will be viewed at, and the drawing cost
                # is set by the output tile rather than the source frame.
                image = self._overlay(image, boxes)
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=quality, optimize=False)
            return out.getvalue(), image.width, image.height
        except Exception as exc:  # noqa: BLE001 - a bad frame is not fatal
            self.last_error = f"re-encode failed: {exc}"
            self.errors += 1
            return jpeg, width, height

    # -- the still encoder --------------------------------------------------

    @staticmethod
    def _nv12_to_rgb(nv12: bytes, width: int, height: int):
        """One NV12 buffer -> an (h, w, 3) uint8 RGB array, or None.

        The buffer must be **unpadded**: Y as h*w bytes followed by interleaved
        UV as (h/2)*(w/2)*2. `sender.sample_to_nv12` strips the hardware row
        alignment before handing it over, because honouring a stride here would
        mean this function needed to know about the VIC.

        **Colour is BT.601 limited range**, which is what this pipeline's
        `video/x-raw,format=NV12` carries, and it is an assumption rather than a
        measurement - the caps do not state colorimetry, so a small cast is
        possible. That matters for looking at the picture and not at all for
        measuring anything on it: ArUco, and every corner refinement that follows
        it, works on the **Y plane**, which is carried here exactly as the sensor
        pipeline produced it. So a still is a faithful measurement image with a
        colour rendering that may be a shade off, and `colour` in the metadata
        says which matrix was used rather than leaving it to be guessed at.
        """
        if np is None:
            return None
        expected = width * height * 3 // 2
        if width <= 0 or height <= 0 or height % 2 or width % 2:
            return None
        if len(nv12) < expected:
            return None
        flat = np.frombuffer(nv12[:expected], dtype=np.uint8)
        y = flat[: width * height].reshape(height, width).astype(np.int32)
        uv = flat[width * height:].reshape(height // 2, width // 2, 2)
        # Nearest-neighbour chroma upsample. Bilinear would be marginally nicer
        # to look at and is not worth 5 megapixels of interpolation for an image
        # whose luma is the part that gets measured.
        u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1).astype(np.int32)
        v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1).astype(np.int32)
        u = u[:height, :width] - 128
        v = v[:height, :width] - 128
        c = (y - 16) * 298
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip((c + 409 * v + 128) >> 8, 0, 255)
        rgb[:, :, 1] = np.clip((c - 100 * u - 208 * v + 128) >> 8, 0, 255)
        rgb[:, :, 2] = np.clip((c + 516 * u + 128) >> 8, 0, 255)
        return rgb

    def _encode_still(self, nv12: bytes, width: int, height: int, quality: int):
        """NV12 -> (jpeg, w, h, note), or (None, 0, 0, why).

        Falls back to a **grayscale** JPEG of the Y plane when numpy is missing,
        rather than to nothing: the luma plane is what a marker is measured on, so
        a colourless still is a fully usable one and an absent still is not.
        """
        if Image is None:
            return None, 0, 0, "Pillow is not installed on this board"
        out = io.BytesIO()
        rgb = self._nv12_to_rgb(nv12, width, height)
        if rgb is not None:
            image = Image.fromarray(rgb, mode="RGB")
            note = "bt601-limited"
        else:
            need = width * height
            if len(nv12) < need:
                return None, 0, 0, f"{len(nv12)} B is short of a {width}x{height} frame"
            # frombytes rather than the numpy path: this branch exists precisely
            # for the case where numpy is what is missing.
            image = Image.frombytes("L", (width, height), nv12[:need])
            note = "luma-only"
        image.save(out, format="JPEG", quality=int(quality), optimize=False,
                   subsampling=0 if note == "bt601-limited" else -1)
        return out.getvalue(), image.width, image.height, note

    def _post_still(self, camera: str, nv12: bytes, width: int, height: int,
                    t_capture: float, request_id: int, quality: int,
                    meta: dict) -> None:
        """Encode and upload one full-resolution frame. One attempt, then admit it.

        No retry loop, unlike `_post`: a still is 2-3 MB, a failed upload has
        already spent most of a minute of uplink, and the operator would rather
        press the button again on a link that is working than have this quietly
        spend another minute on one that is not. The failure is counted and
        printed, and the dashboard's request stays outstanding, so the next poll
        offers it again anyway.
        """
        t0 = time.monotonic()
        jpeg, out_w, out_h, note = self._encode_still(nv12, width, height, quality)
        if jpeg is None:
            self.stills_failed += 1
            self.last_error = f"still encode failed: {note}"
            print(f"cloud_camera: cam{camera} still not sent - {note}", flush=True)
            return
        encode_ms = 1000.0 * (time.monotonic() - t0)

        query = {
            "cam": camera,
            "id": request_id,
            "t": f"{t_capture:.3f}",
            "width": out_w,
            "height": out_h,
            "quality": int(quality),
            "label": f"cam{camera}",
            "colour": note,
            **{k: str(v) for k, v in meta.items()},
        }
        path = f"{self.base_path}/api/camera/capture/upload?" \
               f"{urllib.parse.urlencode(query)}"
        headers = self._headers(**{"Content-Type": "image/jpeg"})

        # Its own connection, closed afterwards, rather than the keep-alive
        # socket the stream uses: this write occupies the link for seconds, and
        # a preview POST that queued behind it would time out and be blamed on
        # the link. Cheap - a still is a once-in-a-while thing.
        self._drop_connection()
        connection = self._get_connection()
        if connection is None:
            self.stills_failed += 1
            return
        previous, connection.timeout = connection.timeout, STILL_TIMEOUT
        try:
            connection.request("POST", path, body=jpeg, headers=headers)
            response = connection.getresponse()
            body = response.read()
        except (http.client.HTTPException, OSError, ssl.SSLError) as exc:
            self.stills_failed += 1
            self.last_error = f"still upload failed: {exc or exc.__class__.__name__}"
            print(f"cloud_camera: cam{camera} still upload failed ({exc})", flush=True)
            self._drop_connection()
            return
        finally:
            connection.timeout = previous
            self._drop_connection()

        if response.status == 200:
            self.stills_sent += 1
            self.last_error = None
            print(f"cloud_camera: cam{camera} still {out_w}x{out_h} "
                  f"{len(jpeg) / 1048576.0:.2f} MB {note} sent "
                  f"(encode {encode_ms:.0f} ms, "
                  f"{1000.0 * (time.monotonic() - t0) - encode_ms:.0f} ms upload)",
                  flush=True)
            self._absorb(body)
        else:
            self.stills_failed += 1
            self.last_error = f"still HTTP {response.status} {body[:120]!r}"
            print(f"cloud_camera: cam{camera} still refused - {self.last_error}",
                  flush=True)
            if response.status in (401, 403):
                # Same key as everything else here, so this is not going to
                # start working. Forget the request rather than re-uploading
                # 2 MB every poll into a 403.
                self.capture = None

    @staticmethod
    def _overlay(image, boxes):
        """Draw the fraction-space boxes onto `image`. Returns the image to encode.

        Costs ~1 ms for a handful of boxes on a 480x240 tile, in the uplink's own
        worker thread and at the stream's 2 fps - so it is off the capture loop
        twice over and cannot show up in the frame budget. That placement is the
        whole reason this is drawn here and not where the detections are produced.

        A JPEG decodes to a mode the draw may not accept (grayscale, or CMYK from a
        stray file), so convert rather than assume.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.width, image.height
        # 2 px at the 480 px default, 1 px if an operator asks for something tiny.
        # A hairline outline disappears against a choppy sea.
        width = max(1, int(round(w / 320.0)))
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2, colour, label in boxes:
            px1, py1 = x1 * w, y1 * h
            px2, py2 = x2 * w, y2 * h
            draw.rectangle([px1, py1, px2, py2], outline=colour, width=width)
            if not label:
                continue
            tw = draw.textlength(label)
            # Above the box, unless the box is against the top edge - a label drawn
            # off-frame is worse than one inside it.
            ty = py1 - 11 if py1 >= 11 else min(py2, h - 11)
            tx = min(px1, max(0.0, w - tw - 4))
            draw.rectangle([tx, ty, tx + tw + 4, ty + 11], fill=colour)
            draw.text((tx + 2, ty), label, fill=(0, 0, 0))
        return image

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
              t_capture: float, boxes=()) -> None:
        payload, out_w, out_h = self._encode(jpeg, width, height, boxes)
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
        for key in ("enabled", "max_width", "jpeg_quality", "fps", "cameras",
                    "detect"):
            if key in payload and payload[key] != self.stream.get(key):
                if key == "detect":
                    print(f"cloud_camera: dashboard turns the detector "
                          f"{'ON' if payload[key] else 'OFF'}", flush=True)
                self.stream[key] = payload[key]
                changed = True

        # The capture request is separate from the stream config and is allowed
        # to be absent, which is the normal case - `"capture": null` on every
        # reply is what clears a finished one. A *new* id resets which cameras
        # have been handed over, since those belong to the old request.
        if "capture" in payload:
            capture = payload["capture"]
            capture = capture if isinstance(capture, dict) else None
            before = (self.capture or {}).get("id")
            self.capture = capture
            if capture and capture.get("id") != before:
                with self._lock:
                    self._still_id = int(capture.get("id") or 0)
                    self._still_taken = set()
                print(f"cloud_camera: full-res capture #{capture.get('id')} "
                      f"asked for cam{'/cam'.join(str(c) for c in capture.get('cameras') or ())}"
                      f" at q{capture.get('quality')}", flush=True)

        self.last_config_at = time.monotonic()
        if changed:
            print(f"cloud_camera: dashboard asks for {self.stats()['asked']}, "
                  f"{'streaming' if self.enabled else 'off'}"
                  f"{'' if self.detect else ', detector off'}", flush=True)

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
