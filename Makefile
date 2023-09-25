MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.RECIPEPREFIX = >

SHELL := bash
.ONESHELL:
.DELETE_ON_ERROR:
.SHELLFLAGS := -e -o pipefail -c

help:
> @echo "=== Usage ==="
> @echo ""
> @echo ""
> @echo "create-env             create a kmol conda environment for the project"
> @echo "build-docker           build the docker container"
> @echo "build-docker-openfold  build openfold docker image, only required to generate the msa script"

create-env:
> @./install.sh
> @echo "Installation done: run 'conda activate kmol' to enable the virtual env and be able to run the kmol command."

build-docker:
> @./docker/build_docker.sh

build-docker-openfold:
> @cd openfold
> @docker build -t openfold . \
  --build-arg HOST_UID=$$(id -u) \
  --build-arg HOST_GID=$$(id -g) \
  --build-arg HOST_USER=$${USER}
