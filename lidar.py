#!/usr/bin/env python3
"""RPLidar C1 driver: a background thread that keeps a buffer of recent sweeps.

    from lidar import LidarReader
    rd = LidarReader(); rd.start()
    sweep, skew = rd.sweep_near(t_capture)      # the sweep concurrent with a frame

Standalone, to see what the sensor is actually returning:

    ./.venv/bin/python lidar.py                 # sweep summaries until Ctrl+C
    ./.venv/bin/python lidar.py --seconds 5

Why a buffer and not "the newest sweep"
---------------------------------------
Because the newest sweep is the wrong one. A camera frame reaches the fusion code
about 250 ms after its photons landed (measured: min 174, median 247, max 471 ms
-- see the README), while a lidar sweep reaches this process within a few ms of
the mirror passing. Colorising a frame with whatever sweep is newest therefore
pairs measurements a quarter-second apart, which at 3 m/s is 0.75 m of pure
registration error, in a sensor whose whole range is 12 m.

So sweeps are kept for a couple of seconds and selected by TIME against the
frame's `t_capture`, and `sweep_near` returns the achieved skew so the caller can
report it rather than assume it. This is the same discipline the camera branches
already use with PTS matching, for the same reason.

Three things about the C1 that the `rplidar` library on PyPI gets wrong, all of
which look like a broken sensor:

  * **SET_PWM is not a C1 command.** `RPLidar.__init__` calls `start_motor()`,
    which sends SET_PWM (0xF0) for the A2's motor driver. The C1 spins its own
    motor and answers 0xF0 with nothing at all - and, worse, swallows the
    *next* command too, so the GET_INFO that follows times out and the library
    reports `Descriptor length mismatch`. `C1Lidar` below overrides the motor
    methods to touch DTR only.
  * **Scan mode survives the process.** The C1 keeps streaming after the port
    closes, so a run that was killed with SIGKILL - or any run that did not
    reach its `finally` - leaves the next one reading measurement bytes where
    it expects a response descriptor. That surfaces as `Wrong body size`, or as
    a scan that yields nothing while the raw port is clearly busy. Hence the
    STOP-and-flush before the first command, and again on the way out.
  * **The first measurement is ~2 s behind the SCAN command.** The C1 answers
    SCAN with its descriptor immediately and then sends nothing at all while
    the motor spins up. The library reads measurements with the port's own
    timeout, which defaults to one second, so the very first read times out and
    raises `Wrong body size` again - on a sensor that is about to work fine.
    READ_TIMEOUT below is the fix and is why it is not 1.

Measured on this unit, once settled: **10.0 Hz**, 100 ms/rev, ~400 returns at a
0.9 deg step, of which ~100 come back with distance 0 and are dropped here rather
than downstream.

Measure it AFTER it settles. The C1 comes up nearer 14 Hz and slows to 10 over
the first few seconds, so a short run reads a spin-up transient and reports a
rate the sensor does not hold. Two things downstream depend on the real figure:
a rotation (100 ms) is LONGER than a camera frame (71.4 ms), so a sweep is
emphatically not an instant; and sweeps are SLOWER than frames, so roughly three
frames in ten find no new sweep and must reuse the one they have rather than wait.
"""

import argparse
import sys
import threading
import time
from collections import deque

import numpy as np
from rplidar import RPLidar, RPLidarException

from config import lidar_port

# Fixed for the C1. The A-series' 115200 gets you a silent port, not an error.
BAUDRATE = 460800

STOP_SETTLE = 0.1  # the C1 finishes the rotation in flight before going idle
BOOT_SETTLE = 2.0  # after RESET, before it answers again
# Every read the library makes uses this, so it has to cover the worst one: the
# motor spin-up between SCAN and the first measurement, measured at ~2s. Once
# the stream is running each read returns in microseconds, so a generous value
# costs nothing and only ever applies to a link that has actually gone quiet.
READ_TIMEOUT = 5.0

# Datasheet limits. Returns outside these are the sensor reporting that it did
# not get a usable echo, not a real object, and are dropped.
MIN_RANGE_M = 0.05
MAX_RANGE_M = 12.0

# A sweep with fewer usable returns than this is a partial rotation -- the first
# one after SCAN always is, and so is whatever was in flight when the buffer
# overflowed. Passing it on would look like the world emptied out for one frame.
MIN_SWEEP_POINTS = 30


class C1Lidar(RPLidar):
    """`RPLidar` with the A2 motor commands taken out. See the module docstring."""

    def start_motor(self):
        # DTR low is the A1's motor enable and harmless here; the C1 ignores it.
        # No SET_PWM - that is the command that wedges the C1.
        self._serial_port.dtr = False
        self.motor_running = True

    def stop_motor(self):
        self._serial_port.dtr = True
        self.motor_running = False

    def quiesce(self):
        """Leave the sensor idle with an empty buffer, from any prior state.

        Safe to call on a device that is mid-scan, that has just booted, or
        that a previous crashed run left streaming.
        """
        self.stop()  # sends STOP, then clear_input()
        time.sleep(STOP_SETTLE)
        self._serial_port.reset_input_buffer()


def open_lidar(port=None, baudrate=BAUDRATE, timeout=READ_TIMEOUT):
    """Connected, idle, verified sensor - or an exception naming the port.

    One retry through RESET: a C1 that was left in a state STOP alone does not
    clear answers the first GET_INFO with garbage and the second one properly.
    """
    port = port or lidar_port()
    lidar = C1Lidar(port, baudrate=baudrate, timeout=timeout)
    for attempt in (1, 2):
        try:
            lidar.quiesce()
            info = lidar.get_info()
            health = lidar.get_health()
        except RPLidarException as exc:
            if attempt == 2:
                lidar.disconnect()
                raise
            print(f"[lidar] {exc} - resetting the sensor and retrying", flush=True)
            lidar.reset()
            time.sleep(BOOT_SETTLE)
            continue
        return lidar, info, health
    raise AssertionError("unreachable")


def close_lidar(lidar):
    """STOP, drain, then close. Order matters -- see the module docstring."""
    try:
        lidar.quiesce()
        lidar.stop_motor()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
        print(f"[lidar] cleanup: {exc}", flush=True)
    try:
        lidar.disconnect()
    except Exception:  # noqa: BLE001
        pass


class Sweep:
    """One rotation: parallel arrays, plus the wall-clock window it spans.

    Arrays rather than a list of points because every consumer is vectorised --
    a whole sweep projects into a camera in one numpy pass.
    """

    __slots__ = ("seq", "angle_deg", "dist_m", "quality", "t_start", "t_end")

    def __init__(self, seq, angle_deg, dist_m, quality, t_start, t_end):
        self.seq = seq
        self.angle_deg = angle_deg
        self.dist_m = dist_m
        self.quality = quality
        self.t_start = t_start
        self.t_end = t_end

    def __len__(self):
        return int(self.angle_deg.size)

    @property
    def t_mid(self):
        return 0.5 * (self.t_start + self.t_end)

    @property
    def period(self):
        return self.t_end - self.t_start

    def times(self):
        """Epoch seconds per point, interpolated across the rotation by ANGLE.

        By angle rather than by arrival time on purpose. The mirror turns at a
        near-constant rate, but the USB serial link delivers measurements in
        bursts, so per-point arrival timestamps are quantised by whenever the
        kernel happened to hand over a chunk. Angle is the physical clock here.

        Good to a few ms, which is what matters: the sweep spans ~70 ms, so a
        point at the back of the rotation is most of a camera frame older than
        one at the front, and treating a sweep as instantaneous smears a moving
        target across the scene.
        """
        if self.angle_deg.size == 0:
            return np.empty(0, dtype=np.float64)
        frac = np.mod(self.angle_deg - self.angle_deg[0], 360.0) / 360.0
        return self.t_start + frac * self.period


class LidarReader(threading.Thread):
    """Reads the C1 forever on its own thread; never blocks the caller.

    Reconnects with backoff rather than dying: on a boat a USB serial link that
    drops for a second should cost a second of points, not the run. `healthy`
    and `errors` are there so the caller can say so in its stats line instead of
    the fusion silently reporting zero points forever.
    """

    daemon = True

    def __init__(self, port=None, history=2.0, on_sweep=None):
        super().__init__(name="lidar")
        self.port = port
        # Seconds of sweeps to keep. Has to cover the camera pipeline's
        # capture->fuse latency (median 247 ms, max 471) with room to spare, or
        # the sweep that matches a frame will already have been dropped.
        self.history = history
        self.on_sweep = on_sweep
        self._lock = threading.Lock()
        self._sweeps = deque()
        self._stopping = threading.Event()
        self.seq = 0
        self.errors = 0
        self.dropped_short = 0
        self.healthy = False
        self.info = None
        self.last_error = None

    # ------------------------------------------------------------------ access
    def sweep_near(self, t, max_skew=0.25):
        """The buffered sweep closest in time to `t` -> (sweep, skew_seconds).

        Returns (None, None) if nothing is buffered, and (sweep, skew) with skew
        over `max_skew` rather than silently refusing -- the caller decides what
        to do with a stale pairing, but it always gets told how stale it is.
        """
        with self._lock:
            if not self._sweeps:
                return None, None
            best = min(self._sweeps, key=lambda s: abs(s.t_mid - t))
        return best, best.t_mid - t

    def latest(self):
        with self._lock:
            return self._sweeps[-1] if self._sweeps else None

    def stats(self):
        with self._lock:
            n = len(self._sweeps)
            newest = self._sweeps[-1] if self._sweeps else None
        return {"healthy": self.healthy, "buffered": n, "seq": self.seq,
                "errors": self.errors, "short": self.dropped_short,
                "hz": (1.0 / newest.period if newest and newest.period > 0 else 0.0),
                "points": len(newest) if newest else 0,
                "last_error": self.last_error}

    # -------------------------------------------------------------------- loop
    def run(self):
        backoff = 0.5
        while not self._stopping.is_set():
            lidar = None
            try:
                lidar, info, health = open_lidar(self.port)
                self.info = info
                self.healthy = True
                backoff = 0.5
                print(f"[lidar] {info.get('model')} fw{info.get('firmware')} "
                      f"health={health[0]} on {self.port or lidar_port()}", flush=True)
                self._pump(lidar)
            except Exception as exc:  # noqa: BLE001 - any failure is a reconnect
                self.errors += 1
                self.last_error = str(exc)[:160]
                self.healthy = False
                if not self._stopping.is_set():
                    print(f"[lidar] {exc}; reconnecting in {backoff:.1f}s", flush=True)
            finally:
                if lidar is not None:
                    close_lidar(lidar)
                self.healthy = False
            if self._stopping.wait(backoff):
                break
            backoff = min(backoff * 2, 10.0)

    def _pump(self, lidar):
        """Group the measurement stream into sweeps until stopped or it fails."""
        ang, dist, qual = [], [], []
        t_start = None
        # max_buf_meas caps how much the driver lets accumulate before flushing.
        # Generous because a flush silently discards points mid-rotation; the
        # consumer here is a list append, so it is not the slow part.
        for new_scan, quality, angle, distance in lidar.iter_measurments(max_buf_meas=3000):
            if self._stopping.is_set():
                return
            now = time.time()
            if new_scan:
                if t_start is not None:
                    self._emit(ang, dist, qual, t_start, now)
                ang, dist, qual = [], [], []
                t_start = now
            if t_start is None:
                continue        # mid-rotation when we attached; wait for a boundary
            ang.append(angle)
            dist.append(distance)
            qual.append(quality)

    def _emit(self, ang, dist, qual, t_start, t_end):
        a = np.asarray(ang, dtype=np.float64)
        d = np.asarray(dist, dtype=np.float64) * 1e-3      # mm -> m
        q = np.asarray(qual, dtype=np.int16)
        # distance 0 means "no usable echo", not "an object at the origin", and
        # it is about a third of every rotation on this unit.
        ok = (d >= MIN_RANGE_M) & (d <= MAX_RANGE_M)
        a, d, q = a[ok], d[ok], q[ok]
        if a.size < MIN_SWEEP_POINTS:
            self.dropped_short += 1
            return
        self.seq += 1
        sweep = Sweep(self.seq, a, d, q, t_start, t_end)
        with self._lock:
            self._sweeps.append(sweep)
            cutoff = t_end - self.history
            while self._sweeps and self._sweeps[0].t_end < cutoff:
                self._sweeps.popleft()
        if self.on_sweep is not None:
            try:
                self.on_sweep(sweep)
            except Exception:  # noqa: BLE001 - a callback must not kill capture
                self.errors += 1

    def shutdown(self):
        self._stopping.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None, help="override config.lidar_port()")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long (default: until Ctrl+C)")
    args = ap.parse_args()

    rd = LidarReader(args.port)
    rd.start()
    t0 = time.monotonic()
    last = 0
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            time.sleep(0.25)
            s = rd.latest()
            if s is None or s.seq == last:
                continue
            last = s.seq
            near = s.dist_m.min()
            i = int(np.argmin(s.dist_m))
            print(f"sweep {s.seq:5d}  {len(s):3d} pts  {1.0 / s.period:5.2f} Hz  "
                  f"median {np.median(s.dist_m):5.2f} m  "
                  f"nearest {near:5.2f} m at {s.angle_deg[i]:6.1f} deg  "
                  f"span {1000 * s.period:5.1f} ms", flush=True)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        rd.shutdown()
        rd.join(timeout=3.0)
        print(f"stats: {rd.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
