#!/usr/bin/env bash
# Bring up everything the pill hand-over demo needs, then print the one command left to run.
#
#   ./demo.sh            start the voice agent and the policy server, check both
#   ./demo.sh --stop     stop them again
#
# The arm is deliberately not started here. run_policy.py asks you to type 'go' before it energizes the
# motors, and that prompt is the point: someone should be looking at the workspace with a hand near the
# 24 V switch when the arm comes alive. This script prints the command; you run it.
set -uo pipefail
cd "$(dirname "$0")"

DIMOS=~/Documents/GitHub/dimos/.venv/bin/python
LEROBOT=/private/tmp/claude-501/-Users-praneethsamineni-Documents-GitHub-hackmit/8d7a55c1-8aed-48c5-a57d-c0e86f902661/scratchpad/lrtrain/bin/python
CKPT=${CKPT:-act_final}

if [ "${1:-}" = "--stop" ]; then
    pkill -f "server/app.py"; pkill -f "policy_server.py"; pkill -f "run_policy.py"
    pkill -f "memory_pipeline.py"
    echo "stopped the voice agent, the policy server, the perception pipeline and the arm"
    exit 0
fi

up() { curl -s -o /dev/null --max-time 5 "$1" 2>/dev/null; }

# 1. the voice agent: find_object, fetch_object, and the /api/push channel the arm speaks through
if up http://127.0.0.1:8000/api/health; then
    echo "voice agent   already up"
else
    nohup "$DIMOS" server/app.py > /tmp/pam.log 2>&1 &
    for _ in $(seq 30); do up http://127.0.0.1:8000/api/health && break; sleep 2; done
    up http://127.0.0.1:8000/api/health && echo "voice agent   started" || { echo "voice agent   FAILED, see /tmp/pam.log"; tail -5 /tmp/pam.log; }
fi

# 2. the ACT policy, on this laptop so a dropped network cannot stall a rollout mid-demo
if up http://127.0.0.1:8011/info; then
    echo "policy server already up"
else
    nohup "$LEROBOT" vla/policy_server.py --policy-path "$CKPT" --device mps --host 127.0.0.1 --port 8011 \
        > /tmp/policy.log 2>&1 &
    for _ in $(seq 40); do up http://127.0.0.1:8011/info && break; sleep 2; done
    up http://127.0.0.1:8011/info && echo "policy server started ($CKPT)" || { echo "policy server FAILED, see /tmp/policy.log"; tail -5 /tmp/policy.log; }
fi

# 3. perception: the camera relay the phone's frames reach, and the pipeline that writes memories.
#    It IS the relay - app.py forwards /api/camera to wss://127.0.0.1:8765, which this serves. Without it
#    the phone's frames go nowhere, so no memories, no hand detection and no face tools.
#    --out runs/live matches app.py's default MEMORY_JSONL, so Pam finds the memories with no env var.
#    Restarting truncates events.jsonl but memory.jsonl is appended, so recorded memories survive.
#    yoloe-11s-seg.pt, not the 26s default: the 26s weights are not on this machine, and CLAUDE.md
#    records 26s calling a pill bottle a water bottle on these very objects.
if pgrep -f "memory_pipeline.py" > /dev/null; then
    echo "perception    already up"
elif [ ! -f server/.env ]; then
    echo "perception    SKIPPED, no server/.env to read ANTHROPIC_API_KEY from"
else
    KEY=$(awk -F= '/^ANTHROPIC_API_KEY=/{print substr($0,index($0,"=")+1)}' server/.env | tr -d "\"' \r")
    ( cd perception && ANTHROPIC_API_KEY="$KEY" nohup ./.venv/bin/python memory_pipeline.py \
        --source wss://0.0.0.0:8765 --cert ../phone/cert.pem --key ../phone/key.pem \
        --out runs/live --weights yoloe-11s-seg.pt \
        --targets "pill bottle,water bottle,keys,phone" \
        --static-camera --no-arm --device mps --fps 5 --imgsz 640 --conf 0.15 \
        > /tmp/perception.log 2>&1 & )
    for _ in $(seq 30); do lsof -nP -iTCP:8765 -sTCP:LISTEN > /dev/null 2>&1 && break; sleep 2; done
    lsof -nP -iTCP:8765 -sTCP:LISTEN > /dev/null 2>&1 \
        && echo "perception    started (relay on 8765, memories -> perception/runs/live)" \
        || { echo "perception    FAILED, see /tmp/perception.log"; tail -5 /tmp/perception.log; }
fi

echo
curl -s --max-time 5 http://127.0.0.1:8000/api/health 2>/dev/null | sed 's/^/health: /'
echo "pills:  $(curl -s --max-time 10 'http://127.0.0.1:8000/api/find?q=pills' 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["say"][:90])' 2>/dev/null)"

IP=$(ipconfig getifaddr en0 2>/dev/null)
cat <<EOF

phone:  https://${IP:-<no wifi>}:8443/     (accept the certificate warning - iOS needs HTTPS for the mic)

now run the arm, in its own terminal, with a hand near the 24 V switch:

  source ~/Documents/GitHub/dimos/.venv/bin/activate && cd $(pwd)
  PYTHONPATH=. python vla/run_policy.py --real --camera-index 0 \\
    --home vla/spots/left_01.json --handover vla/spots/handover.json \\
    --present-s 0.5 --catch-s 1 \\
    --voice-url http://127.0.0.1:8000/api/push \\
    --announce "Here are your pills." \\
    --return-after-s 3 --return-speed 0.15 --server http://127.0.0.1:8011

type 'go', then 'p'. Leave it there - saying "I forgot where I put my pills" to Pam drives the rest.

The hand-over starts the moment the jaws shut on the bottle: the arm turns round, says the line, waits
--catch-s and drops. Keep your hand underneath. --grasp-s (35s) is only a backstop for a pick that never
closes at all.

  'h' hand over now, without waiting out the grasp window
  'r' reset to home            (or: curl -X POST http://127.0.0.1:8020/home)
  'x' stop the policy
EOF
