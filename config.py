"""Board-specific device paths, defined once so nothing hardcodes a raw /dev/*.

    from config import lidar_port
"""

import os

# Created by 99-ligmax-serial.rules, keyed to the RPLidar C1's USB-serial
# adapter by its own serial number rather than by enumeration order. A bare
# /dev/ttyUSB0 is whichever USB-serial device the kernel probed first, and on
# a board with more than one such adapter that's not a promise - it's a coin
# flip that has happened to land the same way so far.
LIDAR_PORT = "/dev/ligmax-lidar"


def lidar_port():
    """Path to the RPLidar C1's serial port.

    Raises rather than falling back to /dev/ttyUSB0: a stale or missing udev
    rule should fail loudly, not silently hand back whatever port happened to
    enumerate first.
    """
    if not os.path.exists(LIDAR_PORT):
        raise FileNotFoundError(
            f"{LIDAR_PORT} does not exist. Install 99-ligmax-serial.rules "
            "(see that file for the two-line install) and reconnect the sensor."
        )
    return LIDAR_PORT
