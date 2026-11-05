#!/usr/bin/env bash
# Start the XFCE session once the X server accepts connections, in its own D-Bus session.
# (Starting it as soon as Xvfb's process exists races the server, and the session dies on boot.)
for _ in $(seq 100); do xset -display "$DISPLAY" q >/dev/null 2>&1 && break; sleep 0.1; done
exec dbus-run-session -- startxfce4
