#!/usr/bin/env bash

# Script to install / update and clean unwanted package in one command

set -e

export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
if [ "$1" == "--purge" ]; then
    shift
    /usr/bin/apt-get -y purge $*
else
    /usr/bin/apt-get -y install --no-install-recommends $*
fi
/usr/bin/apt-get -y autoremove
/usr/bin/apt-get -y autoclean
/usr/bin/apt-get -y clean
/bin/rm -rf /var/lib/apt/lists/*