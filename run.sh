#!/usr/bin/env bash
# Start the detector and stream detections to the Pi, which fuses them.
#
# Checks the cameras actually deliver a frame first: on this board the capture
# stack can sit in a latched Argus error state where every mode enumerates fine
# and streams nothing, and only a power cycle clears it. Better to say so up front
# than to hand back an empty feed.
#
# Two destinations, and they are not the same:
#   HOST:PORT              detections + preview, TCP, to ligmax-pi3.local. The Pi
#                          merges them with the aft lidar and sends one world model
#                          to the dashboard. Point this at a laptop running
#                          receiver.py instead when you are bench testing.
#   the dashboard uplink   preview JPEG only, HTTPS, to live.ligmax.no, and only
#                          when an operator switches video on. cloud_camera.py
#                          takes its target from LIGMAX_UPLOAD_URL /
#                          LIGMAX_DEPLOY_URL and its secret from LIGMAX_BOAT_KEY,
#                          all of which are already in /etc/ligmax/node.env.
#                          NO_CLOUD=1 disables it.
#
# PORT is 3401, not 3338: the dashboard binds 3338 and live.ligmax.no is forwarded
# there, so this feed moved off it (docs/findings.md item 1).
set -uo pipefail

HOST=${HOST:-ligmax-pi3.local}
PORT=${PORT:-3401}
MODE=${MODE:-0}
PREVIEW=${PREVIEW:-640x320}
NO_CLOUD=${NO_CLOUD:-0}

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

CLOUD_ARG=()
if [ "$NO_CLOUD" = "1" ]; then
  CLOUD_ARG=(--no-cloud)
fi

# Lidar is OPT-IN here, deliberately, even though the sender defaults it on.
#
# Switching it on puts a new message type on the link to the Pi (protocol.py
# KIND_LIDAR: no `cam` field, empty JPEG payload). A consumer that reads
# `header["cam"]` without checking `kind` first will file every sweep as camera 0
# and blank that feed -- receiver.py in this repo had exactly that bug until the
# lidar was added. So the Pi has to learn the message before this is turned on,
# and turning it on is a decision rather than a side effect of deploying.
#
#   LIDAR=1 ./run.sh          once ligmax-pi3 dispatches on `kind`
LIDAR_ARG=(--no-lidar)
if [ "${LIDAR:-0}" = "1" ]; then
  LIDAR_ARG=()
fi

echo
exec ./.venv/bin/python sender.py \
  --host "$HOST" --port "$PORT" --mode "$MODE" --preview "$PREVIEW" \
  "${CLOUD_ARG[@]}" "${LIDAR_ARG[@]}" "$@"
