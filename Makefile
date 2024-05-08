MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.RECIPEPREFIX = >

SHELL := bash
.ONESHELL:
.DELETE_ON_ERROR:
.SHELLFLAGS := -e -o pipefail -c

.PHONY: help
help:
> @echo "=== Usage ==="
> @echo ""
> @echo ""
> @echo "create-env             create a kmol conda environment for the project"
> @echo "build-docker           build the docker image"
> @echo "build-docker-openfold  build openfold docker image, only required to generate the msa script"

.PHONY: create-env
create-env:
> @./scripts/install.sh

.PHONY: build-docker
build-docker:
> @./docker/build_docker.sh

OPENFOLD_BUILD_ARGS:=$(foreach barg,HOST_UID=$(shell id -u) HOST_GID=$(shell id -g) HOST_USER=$${USER},--build-arg $(barg))

.PHONY: build-docker-openfold
build-docker-openfold:
> @cd openfold
> @docker build $(OPENFOLD_BUILD_ARGS) -t openfold .
