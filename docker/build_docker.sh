#!/bin/env bash

set +e

echo "Preparing kmol source..."
rm -f docker/kmol.tar.gz
tar \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.so' \
    -cf docker/kmol.tar.gz \
    LICENSE.txt pyproject.toml setup.cfg setup.py environment.yml ./src

echo "Creating docker image..."
docker build -f docker/Dockerfile -t elix-kmol:base docker/

echo "Building local project in the docker..."
## We need at least one gpu visible for kmol to compile (building wheels to install would
## be better but not available right now)
PID=$$
docker run --gpus=all --cidfile "/tmp/${PID}.container_id" -v $(pwd)/docker/install-venv.sh:/opt/elix/kmol/run.sh -ti elix-kmol:base
CONTAINER_ID=$(cat "/tmp/${PID}.container_id"); rm -f "/tmp/${PID}.container_id"


IMAGE_ID=$(docker commit -a "Kmol image builder" -m "Install kmol runtime" "${CONTAINER_ID}")
docker rm ${CONTAINER_ID}
echo "   => Image id: ${IMAGE_ID}"

docker tag "${IMAGE_ID}" "elix-kmol:1.1.4"


echo -e "\nLaunch cmd examples"
echo -e "   => Simplest command:"
echo -e "docker run --rm -ti --gpus=all -v ./data:/opt/elix/kmol/data elix-kmol:1.1.4 {job} {path_to_config} \n"
echo -e "   => If a connection though bash is needed instead"
echo -e "docker run --rm -ti --gpus=all --entrypoint /bin/bash elix-kmol:1.1.4"