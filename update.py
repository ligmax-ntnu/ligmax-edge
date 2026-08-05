"""Run run.sh, and pull + restart it when the dashboard's Update button is pressed.

Started at boot by ligmax-edge.service, so the Jetson comes up on its own after a
power cycle with no SSH needed. Runs as the repo owner: it owns the child process
and restarts it itself, so no sudo and no systemctl.

Note: a pull does NOT rebuild the TensorRT engines. Those are built on the board and
gitignored, so new detector code that needs a new engine stays inert until someone
rebuilds it by hand.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
NAME = "ligmax-edge"
START = ["./run.sh"]
DASH = os.environ.get("LIGMAX_DEPLOY_URL", "https://live.ligmax.no").rstrip("/")
KEY = os.environ.get("LIGMAX_NODE_KEY", "")
POLL = 30


def say(msg):
    # Goes to the journal: journalctl -u ligmax-edge -f
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ask(path, body=None):
    req = urllib.request.Request(
        f"{DASH}/api/deploy/{NAME}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def head():
    return subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


while True:
    # start_new_session so we signal run.sh AND the sender.py it spawns
    child = subprocess.Popen(START, cwd=REPO, start_new_session=True)
    say(f"started {START[0]} as pid {child.pid} at {head()[:8]}")

    nonce = None
    while child.poll() is None:
        time.sleep(POLL)
        try:
            pending = ask("/pending")
            if pending.get("requested"):
                nonce = pending.get("nonce")
                break
        except Exception:
            pass  # dashboard unreachable is normal in the field; keep running

    if child.poll() is None:
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        try:
            child.wait(15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)

        before = head()
        pull = subprocess.run(
            ["git", "-C", REPO, "pull", "--ff-only"], capture_output=True, text=True
        )
        note = (pull.stdout + pull.stderr).strip().replace("\n", " ")
        say(f"pull: {note}")
        # Report, or /pending keeps saying "requested" and we restart in a loop.
        try:
            ok = pull.returncode == 0
            ask(
                "/report",
                {
                    "nonce": nonce,
                    "result": "ok" if ok else "failed",
                    "message": note[:300],
                    "head": head(),
                },
            )
        except Exception as exc:
            say(f"could not report: {exc}")
        if pull.returncode != 0:
            say("pull failed - restarting the old code")
        elif before == head():
            say("nothing new")
    else:
        # run.sh exits non-zero when the cameras are in a latched Argus error state,
        # which only a power cycle clears. Retrying is still right: it logs the
        # reason every 5 s, which is what tells you it is the cameras and not the code.
        say(f"{START[0]} exited with {child.returncode}; restarting in 5s")
        time.sleep(5)
