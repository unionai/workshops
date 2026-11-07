#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${RPA_SSH_PUBKEY:-}" ]]; then
  echo "$RPA_SSH_PUBKEY" > /home/rpa/.ssh/authorized_keys
  chown rpa:rpa /home/rpa/.ssh/authorized_keys && chmod 600 /home/rpa/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null
mkdir -p /run/sshd
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
