#!/usr/bin/env bash
# Train and serve the arm's policy on a RunPod GPU pod (PyTorch template, HTTP port 8000 exposed).
#
#   bash vla/runpod.sh train /workspace/openyam act         # or smolvla; dataset from `dimos dataprep build`
#   bash vla/runpod.sh serve outputs/act/checkpoints/last/pretrained_model
#
# The laptop then points RemotePolicyModule at https://<pod-id>-8000.proxy.runpod.net
# STEPS / BATCH / DEVICE override the defaults below.
set -euo pipefail
cmd=${1:?usage: runpod.sh train <dataset_dir> [act|smolvla] | serve <pretrained_model_dir>}
path=${2:?missing path}
policy=${3:-act}
extras=dataset,training
[ "$policy" = smolvla ] && extras=$extras,smolvla
pip install -q "lerobot[$extras]==0.6.0" opencv-python-headless

case $cmd in
  train)
    if [ "$policy" = smolvla ]; then policy_arg=--policy.path=lerobot/smolvla_base; else policy_arg=--policy.type=$policy; fi
    lerobot-train --dataset.repo_id=local/openyam --dataset.root="$path" --dataset.video_backend=pyav \
      "$policy_arg" --policy.device="${DEVICE:-cuda}" --policy.push_to_hub=false \
      --output_dir="outputs/$policy" --steps="${STEPS:-20000}" --batch_size="${BATCH:-32}" ;;
  serve)
    python "$(dirname "$0")/policy_server.py" --policy-path "$path" --device "${DEVICE:-cuda}" ;;
  *)
    echo "unknown command: $cmd"; exit 1 ;;
esac
