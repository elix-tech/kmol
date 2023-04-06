#!/bin/bash

if [[ $# -eq 0 ]]; then
  exec /bin/bash --login
else
  /bin/bash --login -i -c "$@"
fi
