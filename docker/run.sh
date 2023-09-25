#!/bin/env bash

export SHELL=$(which bash)

if [[ $# -eq 0 ]]; then
  exec ${SHELL} --init-file <(echo "source /etc/profile; conda activate kmol") -i
else
  params="$@"
  exec ${SHELL} --init-file <(echo "source /etc/profile; conda activate kmol") -i -c "kmol ${params}"
fi
