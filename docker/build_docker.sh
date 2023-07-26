#!/bin/bash

set +e

echo " -o- Preparing kmol source..."
rm -f docker/kmol.tar.gz
cp environment.yml docker/environment.yml
tar \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.so' \
    -cf docker/kmol.tar.gz \
    LICENSE.txt pyproject.toml setup.cfg setup.py ./src

echo " -o- Creating docker image..."
cd docker
docker build -t elix-kmol:base .

echo " -o- Building local project in the docker..."
## We need at least one gpu visible for kmol to compile (building wheels to install would
## be better but not available right now)
PID=$$
docker run --gpus=all --cidfile "/tmp/${PID}.container_id" -v $(pwd)/install-venv.sh:/opt/elix/kmol/run.sh -ti elix-kmol:base
CONTAINER_ID=$(cat "/tmp/${PID}.container_id"); rm -f "/tmp/${PID}.container_id"

IMAGE_ID=$(docker commit -a "Kmol image builder" -m "Install kmol runtime" "${CONTAINER_ID}")
docker rm ${CONTAINER_ID}
echo "   => Image id: ${IMAGE_ID}"

docker tag "${IMAGE_ID}" "elix-kmol:1.1.6"

echo
echo " -o-  Launch examples -o- "
echo
echo "   - Simplest command (start kmol {job} {path_to_config} in a container)"
echo "     docker run --rm -ti --gpus=all --user \$(id -u):\$(id -g) -e MPLCONFIGDIR=/opt/elix/kmol/data/.mplconfig -v ./data:/opt/elix/kmol/data elix-kmol:1.1.6 {job} {path_to_config}"
echo
echo "   - Running an interactive shell in the same environment"
echo "     docker run --rm -ti --gpus=all --user \$(id -u):\$(id -g) -e MPLCONFIGDIR=/opt/elix/kmol/data/.mplconfig -v ./data:/opt/elix/kmol/data elix-kmol:1.1.6"
echo
echo "   Remove the --user option to run as root in the container."
echo "   Matplot will use the MPLCONFIGDIR in the container as its workspace, it is recommended to set it to a ./data subfolder or add another volume."
echo "      This option must be set with the path usable from inside of the container (i.e. use the right part of the data volume mount as the base)"
echo " -o-o-o-o-o-o-o-o-o-o-o-o-"
echo
