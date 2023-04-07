#!/bin/bash

if [[ $# -eq 0 ]]; then
  exec /bin/bash --login
else
  cmd="$@"
  /bin/bash --login -i -c "${cmd}"
fi
