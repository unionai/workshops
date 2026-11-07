#!/usr/bin/env bash
# Provision the RPA target machine on Debian 13 (trixie). The Dockerfile runs it; it works the same
# on a real Debian VM if you want the target somewhere other than your laptop:
#
#   sudo SRC=/path/to/target_vm bash provision.sh
#
# Result (all under supervisord):
#   :1      Xvfb 1280x800 + XFCE desktop, user "rpa"
#   5900    x11vnc (localhost only)      6080  noVNC (watch the desktop live)
#   8000    CUA computer-server          (computer use: screenshot / mouse / keyboard)
#   8001    Acme ERP REST API            8002  BuildSupply supplier portal
#   desktop Acme ERP Price Maintenance (Tk), drop-folder watcher
# plus OpenSSH on 22 (key auth, user "rpa") for scripts, shell and files.
set -euxo pipefail

SRC="${SRC:-$(cd "$(dirname "$0")" && pwd)}"
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  xvfb xfce4 xfce4-terminal dbus-x11 x11-xserver-utils xdotool x11vnc novnc websockify \
  openssh-server supervisor ffmpeg chromium fonts-dejavu fonts-liberation \
  python3 python3-venv python3-tk python3-dev build-essential \
  ca-certificates curl sudo procps less
apt-get clean

id rpa >/dev/null 2>&1 || useradd -m -s /bin/bash rpa
install -d -o rpa -g rpa /home/rpa/.ssh /home/rpa/Downloads /home/rpa/Desktop
install -d -o rpa -g rpa /srv/acme /srv/drop /srv/share /srv/import /srv/inbox /var/log/acme

# Toy apps and their venv (CUA needs Python >= 3.12; trixie ships 3.13).
install -d /opt/acme
cp -r "$SRC/apps/." /opt/acme/
chmod +x /opt/acme/scripts/*.sh
python3 -m venv /opt/acme/venv
/opt/acme/venv/bin/pip install --no-cache-dir --upgrade pip
# Pinned: cua-computer-server 0.3.46 passes FastAPI an argument that FastAPI 0.145+ rejects.
/opt/acme/venv/bin/pip install --no-cache-dir \
  "fastapi==0.143.0" "uvicorn[standard]==0.54.0" python-multipart openpyxl \
  "cua-computer-server==0.3.46"
chown -R rpa:rpa /opt/acme
sudo -u rpa ACME_DB=/srv/acme/acme.db /opt/acme/venv/bin/python /opt/acme/acme_db.py

# Chromium as the desktop browser: no first-run dialogs, downloads to ~/Downloads without asking.
install -d /etc/chromium/policies/managed
cat >/etc/chromium/policies/managed/rpa.json <<'EOF'
{"DefaultBrowserSettingEnabled": false, "PromptForDownloadLocation": false,
 "DownloadDirectory": "/home/rpa/Downloads", "PasswordManagerEnabled": false,
 "BrowserSignin": 0, "MetricsReportingEnabled": false}
EOF
cat >/home/rpa/Desktop/supplier-portal.desktop <<'EOF'
[Desktop Entry]
Type=Application
Name=Supplier Portal
Exec=chromium --no-first-run --no-sandbox --password-store=basic http://localhost:8002/
Icon=chromium
EOF
cat >/home/rpa/Desktop/acme-erp.desktop <<'EOF'
[Desktop Entry]
Type=Application
Name=Acme ERP
Exec=/opt/acme/venv/bin/python /opt/acme/erp_desktop.py
Path=/opt/acme
Icon=accessories-calculator
EOF
chmod +x /home/rpa/Desktop/*.desktop
chown -R rpa:rpa /home/rpa

# SSH: key auth only. The public key comes from $RPA_SSH_PUBKEY or /tmp/rpa_ssh.pub.
if [[ -n "${RPA_SSH_PUBKEY:-}" ]]; then echo "$RPA_SSH_PUBKEY" > /home/rpa/.ssh/authorized_keys; fi
if [[ -f /tmp/rpa_ssh.pub ]]; then cat /tmp/rpa_ssh.pub > /home/rpa/.ssh/authorized_keys; fi
[[ -f /home/rpa/.ssh/authorized_keys ]] && chown rpa:rpa /home/rpa/.ssh/authorized_keys && chmod 600 /home/rpa/.ssh/authorized_keys
cat >/etc/ssh/sshd_config.d/rpa.conf <<'EOF'
PasswordAuthentication no
PermitRootLogin no
EOF
mkdir -p /run/sshd

install -m 755 "$SRC/start-xfce.sh" /opt/acme/start-xfce.sh
install -m 644 "$SRC/supervisord.conf" /etc/supervisor/conf.d/rpa.conf
# Let the rpa user run supervisorctl (status / restart of the toy apps) without sudo.
sed -i 's/^chmod=0700/chmod=0770\nchown=root:rpa/' /etc/supervisor/supervisord.conf || true
