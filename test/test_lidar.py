"""Capture RPLidar C1 scans to CSV.

    ./.venv/bin/python test/test_lidar.py            # until Ctrl+C
    ./.venv/bin/python test/test_lidar.py 20         # 20 rotations, then stop

The C1's quirks -- the SET_PWM command that wedges it, the scan mode that
survives the process, the ~2 s of silence between SCAN and the first measurement
-- all live in `lidar.py` now, which is what the sender uses too. This script is
the bench tool: raw rotations to a file, nothing projected and nothing fused.

For the fused, colourised cloud see `fusion.py`; for checking that rig.json
actually describes your mounting, see test_lidar_overlay.py.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar import close_lidar, open_lidar  # noqa: E402 - needs the path above


def capture_data(max_rotations=None):
    lidar, info, health = open_lidar()

    print(f"Device Info: {info}")
    print(f"Health Status: {health}")
    if health[0] != "Good":
        print(f"Warning: LiDAR health is {health[0]!r} (error code {health[1]}).")

    filename = os.path.abspath(f"lidar_c1_data_{int(time.time())}.csv")
    print(f"\nSaving to {filename}")
    print("Press Ctrl+C to stop.\n")

    scan_index = 0
    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Timestamp", "Scan_Index", "Quality", "Angle_Deg", "Distance_mm"]
            )

            # One list of measurements per 360-degree rotation.
            for scan in lidar.iter_scans():
                current_time = time.time()
                for quality, angle, distance in scan:
                    writer.writerow(
                        [current_time, scan_index, quality, angle, distance]
                    )

                scan_index += 1
                if scan_index % 10 == 0:
                    # Flush, so a Ctrl+C or a pulled plug still leaves a
                    # readable file rather than a truncated last buffer.
                    csvfile.flush()
                    print(f"Captured {scan_index} full rotations...")
                if max_rotations and scan_index >= max_rotations:
                    break
    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
    finally:
        # STOP and drain BEFORE closing, or the C1 keeps streaming into a closed
        # port and the next run inherits the mess.
        print("Stopping motor and disconnecting...")
        close_lidar(lidar)
        print(f"Disconnected. {scan_index} rotations in {filename}")


if __name__ == "__main__":
    rotations = int(sys.argv[1]) if len(sys.argv) > 1 else None
    capture_data(rotations)
