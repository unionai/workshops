# Agentic RPA: a bot that climbs a ladder of mechanisms

A daily back-office bot on Flyte 2. It does the job the way an engineer would: APIs first, then
system state, shell scripts and files, then browser automation, and it falls back to **computer use**
(Claude looking at the screen and driving the mouse and keyboard) only when nothing cheaper works.
An agent picks the mechanism for each step. Every choice, tool call, screenshot and screen recording
ends up in the Flyte run, so when the bot misbehaves you can see what it did and why.

**What it shows:**
- An agent (`flyte.ai.agents.Agent`) whose tools are Flyte tasks. Each tool call is its own action
  in the run, so the run graph is a record of the agent's decisions.
- Computer use with Claude's `computer_toolset_20260801`, driving a real desktop through
  [CUA](https://github.com/trycua/cua)'s computer server. Every model turn is recorded as a
  `flyte.trace` step, and the report tab shows a live screenshot timeline and the mp4 recording.
- Separate task environments per mechanism: a Playwright image for the browser rung, a light one
  for everything else.
- A deterministic twin of the same workflow, to show that "agentic" is a choice per workflow, not
  a rewrite.
- An approval gate (`flyte.new_condition`) and a daily `flyte.Trigger`.

## The scenario: daily supplier price update

Every morning, purchasing brings the ERP's prices in line with the supplier's latest price list:

1. Sign in to the supplier's partner portal and download today's price file (CSV).
2. Compare it with the ERP's current prices to find what changed, as an Excel workbook.
3. Put the workbook on the shared folder and run the import-staging script.
4. Type each new price into the ERP's desktop client. The ERP's API refuses price changes: policy
   says they go through the desktop client's Price Maintenance screen.
5. Check that the ERP now shows the new prices, and report.

The "BuildSupply" supplier portal and the "Acme ERP" (REST API plus a Tk desktop client) are toy
apps that run on a **target VM**: a Docker container with an XFCE desktop, built from
[`target_vm/`](target_vm/). Flyte reaches *out* to that machine over SSH, HTTP and CUA, the way it
would reach the Windows or Linux boxes an existing RPA estate runs on.

**The twist:** run with `--chaos` and the supplier "redesigns" its portal overnight. A cookie banner
covers the page and the export button moves into a Reports menu under a new label. The Playwright
script breaks. A recorded-click bot would page someone; this one switches to computer use for that
one step, then goes back to the cheaper tools.

## The mechanism ladder

| Rung | Mechanism | Tools in this example |
|---|---|---|
| 1 | **API** | `erp_list_skus`, `erp_set_price_via_api`, `erp_audit` |
| 2 | **System state and services** | `vm_system_state`, `vm_restart_service` |
| 3 | **Scripts and shell** (SSH) | `vm_stage_import`, `vm_shell` |
| 4 | **Files** (SFTP) | `vm_list_files`, `vm_read_file`, `vm_write_file`, `build_price_changes`, `read_price_changes` |
| 5 | **Browser, DOM-driven** (Playwright) | `portal_download_prices` |
| 6 | **Computer use** (screen, mouse, keyboard) | `computer_use` |

The agent's instructions ([`instructions.md`](instructions.md)) say: use the lowest rung that can do
the step, and go up only after a lower rung fails or can't do it.

## Setup

You need Docker, Python 3.10+, `uv`, and a [Claude API key](https://platform.claude.com/)
(or Claude on Vertex AI, see [Models](#models)).

```bash
cd tutorials/agentic-rpa
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
playwright install chromium            # for the browser rung in local runs

# An SSH key the tasks use to reach the target VM
mkdir -p .secrets && ssh-keygen -t ed25519 -N "" -f .secrets/rpa_ssh

# Build and start the target VM (the first build takes a few minutes)
docker build -t agentic-rpa-vm target_vm
docker run -d --name agentic-rpa-vm -e RPA_SSH_PUBKEY="$(cat .secrets/rpa_ssh.pub)" \
  -p 127.0.0.1:2222:22 -p 127.0.0.1:6080:6080 -p 127.0.0.1:8000:8000 \
  -p 127.0.0.1:8001:8001 -p 127.0.0.1:8002:8002 agentic-rpa-vm
```

The ports are bound to `127.0.0.1` on purpose. noVNC (6080) and CUA's computer server (8000) have
no authentication: anyone who can reach them can watch and drive the desktop. Don't publish them on
a network interface.

Open http://localhost:6080/vnc.html to watch the VM's desktop live: you should see the Acme ERP
Price Maintenance window. The portal is at http://localhost:8002 (user `acme-buyer`, password
`demo-password`).

Add your API key to `.env`:

```
ANTHROPIC_API_KEY=your-key-here
```

## Run locally

Local runs execute every task in-process on your laptop, against the VM container.

```bash
export RPA_SSH_KEY_PATH=.secrets/rpa_ssh

# The deterministic twin: fixed order, computer use only for the desktop step
flyte run --local --tui scripted.py price_update_scripted

# Reset the VM, then the agentic version with the portal redesign switched on
flyte run --local tools.py reset_demo
flyte run --local --tui agentic.py price_update --chaos
```

The scripted run takes about a minute and a half. The agentic run with `--chaos` takes 3 to 6
minutes and usually makes two computer-use sessions: one to find the moved download in the portal,
one to enter the prices in the desktop client. Watch them on the noVNC tab while it runs. Run
`reset_demo` between runs, so the ERP starts from its seed prices again.

## Run on a Flyte devbox

`flyte start devbox` runs a single-node Flyte 2 cluster in Docker, with the UI, so you get the run
graph, the report tabs and the recordings. Tasks there run in pods inside the devbox container. Both
containers are on Docker's default bridge network, so the pods reach the VM container at its own
address on that network, with SSH on port 22. Nothing needs to be published on your machine's
network interfaces.

```bash
flyte start devbox && flyte create config --devbox

flyte create secret rpa-vm-ssh-key --from-file .secrets/rpa_ssh
flyte create secret ANTHROPIC_API_KEY            # paste the key when asked

export RPA_VM_HOST=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' agentic-rpa-vm)
export RPA_SSH_PORT=22
flyte run agentic.py price_update --chaos
flyte run scripted.py price_update_scripted
flyte run tools.py reset_demo
```

`flyte run` prints a link to the run in the devbox UI. The first run builds the two task images
(the Playwright one takes a few minutes). The VM's address on the bridge network can change when
the container restarts, so set `RPA_VM_HOST` again after a restart.

To put the bot on its 06:00 America/Chicago schedule, deploy it instead of running it:

```bash
flyte deploy agentic.py env
flyte get trigger                                                       # daily-price-update, active
flyte delete trigger daily-price-update agentic-rpa.price_update        # when you're done
```

## What to look at in a run

- **The run graph:** each tool the agent called, in order, with its inputs and outputs. That includes
  the failed Playwright attempt, the ERP API's 403 refusal, and every computer-use turn.
- **The "Browser" tab** on `portal_download_prices`: a screenshot of the page where the script broke.
- **The "Computer use" tab** on each `computer_use` action: the goal, then a timeline of screenshots
  with what the model said and the actions it took, then the screen recording.
- **The output** of `price_update`: the agent's report (how it got the price file, how many changes,
  which mechanism entered them, the verification, and what a person should look at).

**Retries and fresh runs.** Within one run, if a task is retried (give it `retries=` to allow that),
the child actions it already completed are replayed from their recorded outputs instead of running
again. That makes the scripted twin cheap to recover. The agent is different: its model calls aren't
recorded, so on a retry it plans again and may choose different tools. A new `flyte run` is always a
fresh run and repeats every step, including the ones with side effects, so run `reset_demo` first.

## How it works

```
price_update (agentic.py)                price_update_scripted (scripted.py)
  └─ Agent: picks tools at run time        └─ the same tools in a fixed order
      │
      ├─ erp_*                      HTTP ──────┐
      ├─ vm_*                       SSH  ──────┤
      ├─ build/read_price_changes              │     target VM (Docker, target_vm/)
      ├─ portal_download_prices     HTTP ──────┤       • Acme ERP REST API       :8001
      │    (Playwright, own image)             ├──►    • BuildSupply portal      :8002
      ├─ computer_use               CUA, SSH ──┤       • Acme ERP desktop app (Tk)
      │    (each model turn traced)            │       • CUA computer server     :8000
      └─ request_approval                      │       • sshd :22, noVNC :6080
           (flyte.new_condition)
```

| File | What's in it |
|---|---|
| [`config.py`](config.py) | Settings (all from env vars) and the three task environments |
| [`tools.py`](tools.py) | One task per capability, grouped by rung, plus `request_approval`, `reset_demo`, `set_portal_chaos` |
| [`computer_use.py`](computer_use.py) | The computer-use loop: the toolset mapped onto CUA commands, xdotool for keys, the ffmpeg recorder, the report timeline |
| [`llm.py`](llm.py) | The Claude client, and a `call_llm` adapter so `flyte.ai.agents.Agent` talks to the Messages API directly |
| [`agentic.py`](agentic.py) | `price_update`: the agent over all the tools, and the daily trigger |
| [`scripted.py`](scripted.py) | `price_update_scripted`: the deterministic twin |
| [`instructions.md`](instructions.md) | The agent's instructions: the job, the ladder rule, the guardrails |
| [`target_vm/`](target_vm/) | The target machine: Dockerfile, provisioning, supervisord config, and the toy apps |

Some details worth knowing:

- **Keyboard input goes through `xdotool`**, run on the VM through CUA. pynput's keyboard injection
  does nothing under Xvfb, and xdotool key names are what Claude's computer toolset emits anyway.
- **Recording** is ffmpeg `x11grab` on the VM, started over SSH and stopped with SIGINT so the
  mp4 is finalized. On success the mp4 is a `flyte.io.File` output of the `computer_use` task, so it
  sits in lineage next to the inputs that caused it. If the session fails, the recording is still
  uploaded and embedded in the report tab before the error is raised.
- **The computer-use traces are a record, not a checkpoint.** Each model turn's actions are logged
  as a `flyte.trace` step after they run. If a `computer_use` task is retried, the session starts
  over from whatever is on the screen now. Write goals that are safe to repeat: the price-entry goal
  sets exact prices, so entering one twice leaves the same price (and an extra audit entry).
- **What's enforced in code, and what's only an instruction:**
  - Code: the scripted twin pauses for a person on more than `RPA_APPROVAL_THRESHOLD` (10) changes.
    `build_price_changes` refuses a supplier file whose effective date isn't the run date. Prices
    are compared in whole cents. Both workflows end with a check through the ERP's own data, and the
    agentic run fails if the agent reports an error.
  - Instructions only: in the agentic version, the approval threshold and "prices only come from the
    supplier file" are rules in [`instructions.md`](instructions.md). The agent has `computer_use`
    and an unrestricted `vm_shell`, so it could break them. For production, gate the price-writing
    tools in code: for example `tool(..., requires_approval=True)` with an `approval_callback` that
    checks the staged workbook, or a dedicated price-entry tool that takes the validated workbook
    instead of a free-form goal.
- **Pointing it at a real machine:** everything that touches the VM is in `vm.py` (SSH and SFTP),
  `tools.py` (the shell commands) and `computer_use.py` (CUA and the recorder). Run
  `target_vm/provision.sh` on a Debian VM and set `RPA_VM_HOST` and `RPA_SSH_PORT=22`, or swap those
  modules for a Windows host (OpenSSH with PowerShell, `gdigrab` for recording).

## Models

Both the agent and the computer-use loop default to `claude-opus-5-5`
(`RPA_ORCHESTRATOR_MODEL`, `RPA_COMPUTER_USE_MODEL`).

To use Claude on Vertex AI instead of the Claude API, authenticate with
`gcloud auth application-default login` and set:

```bash
export RPA_LLM_PROVIDER=vertex VERTEX_PROJECT=your-gcp-project VERTEX_REGION=global
```

That works as-is for local runs. In a cluster, the task pods need Google credentials of their own
(for example GKE workload identity), and no `ANTHROPIC_API_KEY` secret is needed.

## Caveats

- The portal and ERP are toys built for the example, and the redesign is switched on by a flag.
- The portal password is in the agent's instructions, so the model sees it. A real deployment would
  keep it out of the model's view: a sign-in tool that types the secret itself, or a vault.
- SSH host key checking is off, because the VM's host key changes on every rebuild. Pin
  `known_hosts` for a real machine.
- The model drives a real desktop. Run the target VM somewhere it can't do harm, as here.
