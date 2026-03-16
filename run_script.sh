#!/bin/bash

TAG=${1:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "res_files/$TAG"

# NOTE: Generate the single manifest containing algorithm info before parallelizing
python runner.py --setup --tag "$TAG"
if [ $? -ne 0 ]; then
    echo "Setup failed. Exiting."
    exit 1
fi

for i in {0..4}; do
    tmux new-session -d -s "run_${TAG}_$i" "python runner.py --dataset $i --tag $TAG"
done

# NOTE: View active sessions with 'tmux ls'. Attach with 'tmux attach-session -t <session_name>'.