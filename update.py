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
POLL = 30  # seconds between /pending checks
TICK = 1  # how often we look at the child, so a restart is not a poll behind
UA = "ligmax-edge/updater"  # deliberately not urllib's default; see ask()
QUIET = 120  # failed polls between journal lines, i.e. one an hour at POLL=30

poll_fails = 0  # consecutive failed polls, for the throttling in poll_if_due

# Wall-clock, and shared across every child this process ever starts -- NOT
# reset each time a new one is spawned. run.sh's own preflight can fail in
# under a second when the cameras are latched in their Argus error state (see
# run.sh), and that state only clears on a power cycle. A child that keeps
# dying faster than POLL seconds must not be able to keep re-arming a 30 s
# countdown forever: that starves /pending completely, and the dashboard sees
# no poll ever, even though this process has been running the whole time.
_next_poll = 0.0


def say(msg):
    # Goes to the journal: journalctl -u ligmax-edge -f
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ask(path, body=None):
    req = urllib.request.Request(
        f"{DASH}/api/deploy/{NAME}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            # Cloudflare fronts live.ligmax.no and bans urllib's default
            # "Python-urllib/3.x" signature outright: every poll came back 403
            # with Cloudflare error 1010, so the node sat at "Never polled" on
            # the dashboard while this process had been running the whole time.
            # That is the same symptom the _next_poll starvation above causes,
            # and it had a second, independent cause -- fixing the timer alone
            # does not help when the request never reaches the app. Any
            # non-default agent is accepted; cloud_camera.py sets one too,
            # which is why the video uplink was never affected.
            "User-Agent": UA,
        },
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def head():
    return subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def poll_if_due():
    """Check /pending, but only once every POLL seconds of wall clock.

    Timer is module-global on purpose -- see the comment on `_next_poll`.
    """
    global _next_poll, poll_fails
    if time.time() < _next_poll:
        return None
    _next_poll = time.time() + POLL
    try:
        pending = ask("/pending")
        if poll_fails:
            say(f"dashboard reachable again after {poll_fails} failed polls")
            poll_fails = 0
        if pending.get("requested"):
            return pending.get("nonce")
    except Exception as exc:
        # An unreachable dashboard IS normal in the field, so this must not fill
        # the journal - but swallowing it in silence is exactly how a 403 on
        # every single poll went unnoticed until someone happened to read the
        # dashboard. Say it the first time, then hourly.
        if poll_fails % QUIET == 0:
            say(f"poll failed ({poll_fails + 1} in a row): {type(exc).__name__}: {exc}")
        poll_fails += 1
    return None


def wait(child=None, seconds=None):
    """Tick once a second, polling on schedule, until the child exits or
    `seconds` run out -- whichever bound is given. Used both while the child
    is up and during the gap before restarting a dead one, so a poll is never
    contingent on the child staying alive.
    """
    deadline = None if seconds is None else time.time() + seconds
    while True:
        if child is not None and child.poll() is not None:
            return None
        if deadline is not None and time.time() >= deadline:
            return None
        time.sleep(TICK)
        nonce = poll_if_due()
        if nonce is not None:
            return nonce


while True:
    # start_new_session so we signal run.sh AND the sender.py it spawns
    child = subprocess.Popen(START, cwd=REPO, start_new_session=True)
    say(f"started {START[0]} as pid {child.pid} at {head()[:8]}")

    nonce = wait(child=child)
    if nonce is None:
        # run.sh exits non-zero when the cameras are in a latched Argus error state,
        # which only a power cycle clears. Retrying is still right: it logs the
        # reason every 5 s, which is what tells you it is the cameras and not the code.
        say(f"{START[0]} exited with {child.returncode}; restarting in 5s")
        # Still polling here, not a bare sleep(5): a child that never lives
        # longer than the retry gap must not be able to hide every poll inside it.
        nonce = wait(seconds=5)

    # Keyed off the request, not off whether the child is still up. Gating the
    # pull on `child.poll() is None` meant a run.sh that had died during the poll
    # interval swallowed the request entirely: nothing was pulled, nothing was
    # reported, and the operator's row sat at "Waiting" for 30 minutes. The
    # cameras latching into an Argus error makes that exit the normal case here.
    if nonce is not None:
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
