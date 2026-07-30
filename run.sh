#!/usr/bin/env bash
# Start the detector and stream to the viewer.
#
# Checks the cameras actually deliver a frame first: on this board the capture
# stack can sit in a latched Argus error state where every mode enumerates fine
# and streams nothing, and only a power cycle clears it. Better to say so up front
# than to hand back an empty feed.
set -uo pipefail

HOST=${HOST:-192.168.99.135}
PORT=${PORT:-3338}
MODE=${MODE:-0}
PREVIEW=${PREVIEW:-640x320}

cd "$(dirname "$0")"

case "$MODE" in
  0) W=2592; H=1944; F=14 ;;
  1) W=1920; H=1080; F=29 ;;
  2) W=1296; H=972;  F=28 ;;
  *) echo "sensor mode $MODE does not stream on this board (only 0, 1, 2 do)" >&2
     exit 2 ;;
esac

if [ ! -e /dev/video0 ] || [ ! -e /dev/video1 ]; then
  echo "/dev/video0 or /dev/video1 missing -- the boot probe failed. Reboot." >&2
  exit 1
fi

echo "checking the cameras actually stream (mode $MODE, ${W}x${H}@${F}) ..."
tmp=$(mktemp -d)
for id in 0 1; do
  # Caps pinned to the mode's exact geometry: leaving them open makes
  # nvarguscamerasrc default to 1920x1080 and the mismatch wedges the stack.
  timeout 40 gst-launch-1.0 nvarguscamerasrc sensor-id="$id" sensor-mode="$MODE" \
      num-buffers=8 \
    ! "video/x-raw(memory:NVMM),width=$W,height=$H,framerate=$F/1" \
    ! nvvidconv ! video/x-raw,format=I420 ! jpegenc \
    ! filesink location="$tmp/c$id.jpg" >/dev/null 2>&1
  if [ -s "$tmp/c$id.jpg" ]; then
    echo "  cam$id ok ($(stat -c%s "$tmp/c$id.jpg") bytes)"
  else
    echo "  cam$id delivered NO frames." >&2
    echo "  Argus is latched in its error state. Power-cycle the board;" >&2
    echo "  restarting nvargus-daemon and reloading nv_ov5647 do not help." >&2
    rm -rf "$tmp"; exit 1
  fi
done
rm -rf "$tmp"

echo
exec ./.venv/bin/python sender.py \
  --host "$HOST" --port "$PORT" --mode "$MODE" --preview "$PREVIEW" "$@"
