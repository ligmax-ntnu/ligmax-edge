"""Wire format shared by sender and receiver.

One message per camera frame, over a plain TCP stream:

    magic   4 bytes   b'BUOY'
    hdrlen  4 bytes   big-endian uint32
    header  hdrlen    UTF-8 JSON
    jpeg    N bytes   N = header['jpeg_bytes']

Length-prefixed rather than newline-delimited because the payload is binary JPEG.
Magic first so a receiver that loses sync can hunt for the next frame boundary
instead of giving up.

Header fields:
    cam          int    0 or 1
    seq          int    monotonic per camera
    ts           float  == t_capture; kept under the old name for compatibility
    t_capture    float  epoch seconds when the frame's FIRST ROW was exposed,
                        derived from the GStreamer buffer PTS. Not when inference
                        ran and not when the message was built -- those are
                        t_sent and latency_ms, kept separate so pipeline delay
                        stays visible instead of contaminating the measurement.
    t_sent       float  time.time() after inference, for latency bookkeeping only
    latency_ms   float  t_sent - t_capture
    readout_ms   float  rolling-shutter sweep top-to-bottom (~64 ms at mode 0).
                        Each detection carries its own t_row within this window;
                        a frame does NOT have one capture instant.
    net_w/net_h  int    detector input size the boxes are expressed in
    jpeg_w/jpeg_h int   preview size, so the receiver can scale boxes to it
    crop         list   [left, top, w, h] of the sensor window the detector saw
    full_w/full_h int   full sensor size, the frame `crop` is measured in
    refined      bool   whether the full-resolution frame was available, i.e.
                        whether width_method could be "refined_edges"
    jpeg_bytes   int    payload length
    fps          float  sender's measured rate, for display
    dets         list   see below

Each detection carries, beyond {id, cls, name, conf, box, card, card_conf}:

    bearing_deg       azimuth right of the optical axis, CAMERA frame
    elevation_deg     positive up, CAMERA frame
    field_angle_deg   angle off the optical axis; how far into the fisheye it is
    ray_cam           unit vector [x,y,z], +x right +y down +z forward. This is
                      what a triangulator wants; the two angles are for humans.
    in_valid_cone     false => bearing is null, the pixel is past the calibrated
                      88 deg limit and no bearing exists there
    sigma_deg         total bearing uncertainty, and split into:
    sigma_calib_deg     the calibration's own error, ~0.25 deg. CORRELATED across
                        every detection and every frame from this camera -- it does
                        not average out over a track and does not cancel between
                        the two cameras.
    sigma_centroid_deg  independent per detection; this part does average down.
    mrad_per_px       local angular scale, which varies ~2x across the frame

    range: {range_m, sigma_m, rel_sigma, alpha_mrad, alpha_sigma_mrad,
            valid, why}
                      Range from apparent size, assuming a sphere of
                      --buoy-diameter (0.40 m for Njord marks):
                      z = (D/2)/sin(alpha/2). Accuracy degrades as z^2 --
                      roughly 6 % at 20 m, 12 % at 50 m, 22 % at 100 m -- so
                      ALWAYS check `valid` and weight by sigma_m rather than
                      trusting range_m because a number came back.
    width_method      "refined_edges" (subpixel edges measured on the full-res Y
                      plane) or "detector_box" (fallback; several times worse)
    edge_sigma_px     the per-edge uncertainty that fed sigma_m
    width_px_full     measured silhouette width in full sensor pixels
    truncated         box touches the crop edge, so the width is a lower bound
                      and range is forced invalid
    t_capture/t_row   frame time, and the time this detection's own sensor rows
                      were exposed. Use t_row for anything geometric.

Range uses the buoy's WIDTH, not its height: these marks float, so the waterline
cuts an unknown amount off the bottom while the horizontal extent through the
centre is the full diameter.

`id` is a track id that persists while the sender keeps seeing the same buoy, so
consecutive frames can be related to each other. It is unique per camera, not
across the pair, and is null when the sender runs with --no-track. Ids are not
reused, but nothing guarantees a buoy keeps one across an occlusion.

`box` is in detector-input pixels (net_w x net_h), letterboxed. The receiver
scales to the preview size; both come from the same source frame, so a single
uniform scale is correct.

`card`/`card_conf` are present only for cardinal detections that went through the
second-stage classifier; otherwise null.
"""

import json
import struct

MAGIC = b"BUOY"
_HDR = struct.Struct(">4sI")

CLASS_NAMES = {0: "green", 1: "red", 2: "cardinal"}
CARDINAL_NAMES = {0: "east", 1: "north", 2: "south", 3: "west"}

# The detector class index that gets a second-stage cardinal classification.
CARDINAL_CLASS_ID = 2


def encode(header: dict, jpeg: bytes) -> bytes:
    header = dict(header, jpeg_bytes=len(jpeg))
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HDR.pack(MAGIC, len(blob)) + blob + jpeg


def _recv_exact(sock, n):
    """Read exactly n bytes, or return None if the peer closed."""
    chunks = []
    got = 0
    while got < n:
        b = sock.recv(min(65536, n - got))
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_message(sock):
    """Read one (header, jpeg) pair. Returns None on clean disconnect.

    Resynchronises on the magic if the stream is ever misaligned, so one bad
    frame does not kill a long-running viewer.
    """
    head = _recv_exact(sock, _HDR.size)
    if head is None:
        return None
    magic, hdrlen = _HDR.unpack(head)
    while magic != MAGIC:
        nxt = sock.recv(1)
        if not nxt:
            return None
        head = head[1:] + nxt
        magic, hdrlen = _HDR.unpack(head)
    if hdrlen > 1 << 20:          # a sane header is well under 1 MB
        return None
    blob = _recv_exact(sock, hdrlen)
    if blob is None:
        return None
    header = json.loads(blob)
    n = int(header.get("jpeg_bytes", 0))
    if n < 0 or n > 64 << 20:
        return None
    jpeg = _recv_exact(sock, n) if n else b""
    if jpeg is None:
        return None
    return header, jpeg
