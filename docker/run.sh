#!/bin/bash

if [[ $# -eq 0 ]]; then
  exec /bin/bash --login
else
  params="$@"
  /bin/bash --login -i -c "kmol ${params}"
fi
