#!/bin/bash +e

echo " -o- Preparing kMoL source..."
rm -f docker/kmol.tar.gz
cp environment.yml docker/environment.yml
tar \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.so' \
    -cf docker/kmol.tar.gz \
    LICENSE.txt pyproject.toml setup.cfg setup.py ./src

KMOL_VERSION=$(grep '^version *= *' setup.cfg | sed 's/^version *= *\([0-9.]*\)/\1/')

echo " -o- Creating docker image..."
cd docker
docker build -t elix-kmol:base .

echo " -o- Building kMoL ${KMOL_VERSION}..."
## We need at least one gpu visible for kmol to compile (building wheels to install would
## be better but not available right now)
PID=$$
docker run --gpus=all --cidfile "/tmp/${PID}.container_id" -v ./install-venv.sh:/opt/elix-inc/run_kmol.sh -ti elix-kmol:base
CONTAINER_ID=$(cat "/tmp/${PID}.container_id"); rm -f "/tmp/${PID}.container_id"

IMAGE_ID=$(docker commit -a "kMoL image builder" -m "Install kMoL runtime" "${CONTAINER_ID}")
docker rm ${CONTAINER_ID}
echo "   => Image id: ${IMAGE_ID}"

docker tag "${IMAGE_ID}" "elix-kmol:${KMOL_VERSION}"

echo
echo " -o-o-o-o-o-o-o-o-o-o-o-o-"
echo " -o-       Usage       -o-"
echo " -o-o-o-o-o-o-o-o-o-o-o-o-"
echo
echo "   - Simplest command (start kmol {job} {path_to_config} in a container)"
echo "     docker run --rm -ti --gpus=all -e KMOL_UID=\$(id -u) -e KMOL_GID=\$(id -g) -v ./data:/home/kmol/data elix-kmol:${KMOL_VERSION} {job} {path_to_config}"
echo
echo "   - Start an interactive shell in the kMoL virtual environment"
echo "     docker run --rm -ti --gpus=all -e KMOL_UID=\$(id -u) -e KMOL_GID=\$(id -g) -v ./data:/home/kmol/data elix-kmol:${KMOL_VERSION}"
echo
echo " -o-o-o-o-o-o-o-o-o-o-o-o-"
echo
