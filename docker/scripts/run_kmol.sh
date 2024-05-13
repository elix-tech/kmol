#!/bin/bash -e

if [ -z "${KMOL_UID}" ]; then
  echo "KMOL_UID is not set"
  exit 1
fi

if [ -z "${KMOL_GID}" ]; then
  echo "KMOL_GID is not set"
  exit 1
fi

chown ${KMOL_UID}:${KMOL_GID} /home/kmol

if ! getent group kmol > /dev/null; then
  addgroup -q --gid ${KMOL_GID} kmol
fi

if ! getent passwd kmol > /dev/null; then
  adduser -q --disabled-password --uid ${KMOL_UID} --shell /bin/bash --ingroup kmol --gecos "kMoL user" kmol
fi

if ! id -Gn kmol | grep -qw kmol; then
  adduser -q kmol kmol
fi

cd /home/kmol
gosu kmol mkdir -p "${MPLCONFIGDIR}"

if [[ $# -eq 0 ]]; then
  exec gosu kmol /bin/bash --login -i
else
  params="$@"
  exec gosu kmol /bin/bash --login -i -c "kmol ${params}"
fi
