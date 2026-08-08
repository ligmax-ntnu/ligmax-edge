"""Wire format shared by sender and receiver.

One message per camera frame, plus one per lidar sweep, over a plain TCP stream:

    magic   4 bytes   b'BUOY'
    hdrlen  4 bytes   big-endian uint32
    header  hdrlen    UTF-8 JSON
    jpeg    N bytes   N = header['jpeg_bytes']

Length-prefixed rather than newline-delimited because the payload is binary JPEG.
Magic first so a receiver that loses sync can hunt for the next frame boundary
instead of giving up.

`kind` says which message this is: KIND_FRAME (a camera frame, with a JPEG) or
KIND_LIDAR (a sweep, with an empty payload). A header without `kind` is a camera
frame, which is what every sender before the lidar existed emitted -- read it that
way rather than rejecting it. DISPATCH ON `kind` BEFORE TOUCHING `cam`: a lidar
message has no `cam`, because most of a rotation is behind both cameras.

Header fields (KIND_FRAME):
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
    saturation   float  chroma gain Argus's ISP already applied, 0-2. Default 2.0,
                        because JetPack ships no ISP tuning for the OV5647 and its
                        untouched output is ~1/3 the chroma of a normal camera.
                        A viewer that colour-corrects MUST read this and apply only
                        the remainder (fusion.ccm_strength_for): a full matrix on
                        top of a boosted frame clips about half of it. Absent on
                        senders older than this field -- treat that as 1.0.
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

A detection that the lidar also saw gains a `lidar` block, which does NOT replace
`range` -- the two are independent measurements and a consumer that can see both
can tell when they disagree. Inside the C1's 12 m the lidar is better by an order
of magnitude (+-3 cm flat, against +-5 % at 10 m rising as z**2):

    n            returns inside the detection's box
    n_used       of those, the foreground cluster -- see `mixed`
    range_m      median range of that cluster, metres
    sigma_m      its uncertainty, floored at the sensor's 3 cm accuracy
    nearest_m    closest single return; the one that matters for not hitting it
    spread_m     depth spread across the cluster
    mixed        true when the box also held returns from well behind the target,
                 i.e. sea showing through around the buoy. The range is the near
                 cluster's, but a large gap is also how a box drawn around the
                 wrong thing announces itself.
    bearing_deg  azimuth in the RIG frame, not the camera frame
    cam          which camera's box this was matched against

Lidar messages (KIND_LIDAR)
---------------------------
One per rotation, ~10 Hz (the C1's settled rate), `jpeg_bytes` 0. Beyond
{kind, seq, ts, t_sent} the header carries `lidar`, holding a COLUMNAR point
cloud -- parallel arrays, not a list of objects, because at ~400 points a sweep
the repeated keys would be most of the bytes:

    seq, n, coloured    sweep number, points IN THESE ARRAYS, how many the
                        cameras could colour
    dropped             returns removed before sending because no camera could
                        see them -- the arc outside both lenses, ~34 deg aft on
                        this rig. `n + dropped` is the size of the rotation, and
                        a returns-per-rev or Hz check wants that sum, not `n`.
                        They were real obstacles: this is the sender saying
                        something else watches that arc (--lidar-keep-unseen
                        ships them uncoloured instead, and then `dropped` is 0
                        and `cam` can be -1).
    stale               of `coloured`, how many were sampled from a frame further
                        away in time than --lidar-max-skew. They are coloured
                        anyway -- a slightly mistimed colour beats none -- but
                        they are the ones to distrust first, and `age_ms` says by
                        how much for each. A handful is the frame buffer working
                        at the edges of a rotation; most of the sweep is a drift
                        between the cameras and the scanner worth investigating.
    in_time             points measured close enough in time to SOME frame to be
                        timely. `n - in_time` is a TIMING shortfall; `in_time -
                        coloured` is simply the part of the rotation no lens
                        covers. Since each point is coloured from the nearest
                        buffered frame rather than one frame per sweep, `coloured`
                        can now exceed `in_time`: the excess is `stale`.
    frame               "rig" -- +x starboard, +y DOWN, +z forward, origin at the
                        lidar unless rig.json moves it. NOT a camera frame.
    t_start, t_end      epoch seconds. A rotation takes ~100 ms -- LONGER than a
                        camera frame -- and is emphatically not an instant, which
                        is what dt_ms below is for
    hz                  measured rotation rate
    skew_ms             [cam0, cam1] sweep-midpoint-to-CURRENT-frame capture-time
                        difference. A whole-sweep summary, and no longer what
                        decides any individual point's colour: each return is
                        coloured from the nearest of a few buffered frames, so
                        `age_ms` is the per-point truth and this is the one
                        readable number per camera for a stats line.
    x, y, z             metres in the rig frame, one entry per point
    dt_ms               ms after t_start that each point was measured, from its
                        angle around the rotation. Use it for anything geometric
                        on a moving boat, exactly as t_row is used per detection.
    age_ms              how far the frame each point was coloured FROM sat from
                        that point's own measurement, or -1 where nothing coloured
                        it (never 0, which would read as a perfect match). This is
                        what makes a mistimed colour safe to ship: weight a point
                        by it, or drop points past your own threshold, but do not
                        assume every colour in a sweep came from the same instant
                        -- they deliberately do not.
    q                   the C1's own return-strength figure
    cam                 0 or 1 -- which camera coloured the point. -1 (no camera
                        could see it) only appears with --lidar-keep-unseen;
                        by default those points are dropped, see `dropped`.
    rgb                 FLAT r,g,b,r,g,b... so it is 3*n long, not n. Sampled off
                        the detector frame -- the same pixels the boxes were drawn
                        from -- and COLOUR-CORRECTED before sending (the OV5647
                        matrix, `fusion._correct`). Display them as they arrive:
                        they are comparable to a corrected preview, and a consumer
                        that boosts saturation to compensate for raw sensor values
                        is now correcting twice. (0,0,0) where cam is -1.
    det                 track id of the detection this return belongs to, or -1.
                        Only foreground returns are tagged, so the sea visible
                        through a box is not attributed to the buoy.
                        KEY IT WITH `cam`, not on its own: track ids are unique
                        per camera and not across the pair (see `id` above), so
                        cam0's #7 and cam1's #7 are different buoys. The pair
                        (cam[i], det[i]) is what identifies a target.

Returns with distance 0 -- about a third of every rotation, the sensor saying it
got no usable echo -- are dropped at the driver and never reach the wire.
"""

import json
import struct

MAGIC = b"BUOY"
_HDR = struct.Struct(">4sI")

KIND_FRAME = "frame"
KIND_LIDAR = "lidar"

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
