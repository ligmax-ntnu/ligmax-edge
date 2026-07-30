#!/usr/bin/env bash
# End-to-end smoke test: real cameras -> detector -> classifier -> viewer, all on
# this machine. Proves the whole chain without needing the remote viewer host up.
#
# Tracks PIDs explicitly rather than using pkill: a pattern like 'receiver.py'
# also matches the very shell running this script, so pkill kills its own parent.
set -uo pipefail

cd "$(dirname "$0")"
SECS=${SECS:-45}
PORT=${PORT:-3338}
HTTP=${HTTP:-8080}
LOGS=$(mktemp -d)
RX=""; TX=""

cleanup() {
  [ -n "$TX" ] && kill "$TX" 2>/dev/null
  sleep 3                      # let the sender drain via EOS, not a hard kill
  [ -n "$TX" ] && kill -9 "$TX" 2>/dev/null
  [ -n "$RX" ] && kill "$RX" 2>/dev/null
  rm -rf "$LOGS"
}
trap cleanup EXIT

./.venv/bin/python receiver.py --port "$PORT" --http-port "$HTTP" >"$LOGS/rx" 2>&1 &
RX=$!
sleep 2

./.venv/bin/python sender.py --host 127.0.0.1 --port "$PORT" \
  --mode "${MODE:-0}" --stats-every 5 >"$LOGS/tx" 2>&1 &
TX=$!

echo "running for ${SECS}s ..."
sleep "$SECS"

echo
echo "=== sender ==="
grep -E "detector |classifier |sensor |preview |connected|fps/cam|Traceback|error|Error" "$LOGS/tx" \
  | grep -viE "GST_ARGUS|CONSUMER|nvbuf_utils" | tail -14

echo
echo "=== receiver ==="
tail -4 "$LOGS/rx"

echo
echo "=== viewer API ==="
curl -s --max-time 5 "http://127.0.0.1:$HTTP/api/status" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('  cameras :', d['cameras'])
print('  fps     :', d['fps'])
print('  dets    :', d['dets'])
for cam, items in d.get('detail',{}).items():
    for it in items[:4]:
        extra = f\"  cardinal={it['card']} ({it['card_conf']})\" if it.get('card') else ''
        print(f\"    cam{cam}: {it['name']:9} {it['conf']:.2f} box={it['box']}{extra}\")
" 2>&1 | head -20

echo
echo "=== annotated frames over HTTP ==="
for c in 0 1; do
  curl -s --max-time 5 -o "$LOGS/c$c.jpg" \
    -w "  /cam$c.jpg HTTP %{http_code}  %{size_download} bytes\n" \
    "http://127.0.0.1:$HTTP/cam$c.jpg"
done
./.venv/bin/python - "$LOGS" <<'PY'
import sys, os
from PIL import Image
for c in (0, 1):
    p = os.path.join(sys.argv[1], f"c{c}.jpg")
    if os.path.getsize(p) > 1000:
        im = Image.open(p)
        print(f"  cam{c} decodes: {im.size} {im.mode}")
    else:
        print(f"  cam{c}: no usable frame")
PY
