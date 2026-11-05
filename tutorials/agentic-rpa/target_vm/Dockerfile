# The RPA target machine as a container: a Debian 13 XFCE desktop with the toy Acme apps, CUA's
# computer server, noVNC and sshd. Build from this directory:
#
#   docker build -t agentic-rpa-vm target_vm
#   docker run -d --name agentic-rpa-vm -e RPA_SSH_PUBKEY="$(cat .secrets/rpa_ssh.pub)" \
#     -p 2222:22 -p 6080:6080 -p 8000:8000 -p 8001:8001 -p 8002:8002 agentic-rpa-vm
FROM debian:trixie-slim
COPY apps /src/apps
COPY provision.sh supervisord.conf start-xfce.sh /src/
RUN SRC=/src bash /src/provision.sh
RUN printf '[program:sshd]\ncommand=/usr/sbin/sshd -D -e\npriority=5\nautorestart=true\n' \
      > /etc/supervisor/conf.d/sshd.conf
COPY docker-entrypoint.sh /entrypoint.sh
EXPOSE 22 6080 8000 8001 8002
ENTRYPOINT ["bash", "/entrypoint.sh"]
