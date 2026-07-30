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
    ts           float  sender's time.time() when the frame was captured
    net_w/net_h  int    detector input size the boxes are expressed in
    jpeg_w/jpeg_h int   preview size, so the receiver can scale boxes to it
    jpeg_bytes   int    payload length
    fps          float  sender's measured rate, for display
    dets         list   [{id, cls, name, conf, box:[x1,y1,x2,y2], card, card_conf}]

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
