

# Get the user id and group, and use it so that created files are not owned by root
HOST_UID = $(shell id -u)
HOST_GID = $(shell id -g)
HOST_USER = $(USER)
export HOST_UID
export HOST_GID
export HOST_USER

help:
	@echo "=== Usage ==="
	@echo ""
	@echo ""
	@echo "create-env             create a kmol conda environment for the project"
	@echo "build-docker           build the docker container"
	@echo "build-docker-openfold  build openfold docker image  only necessary for the generate msa script"
	@echo "wheel	  			  create a wheel for the project"

wheel:
	python setup.py bdist_wheel

create-env:
	bash install.sh

build-docker:
	bash docker/build_docker.sh

build-docker-openfold:
	docker build -t openfold -f docker/Dockerfile_openfold . \
    --build-arg host_uid=$(HOST_UID) \
	--build-arg host_gid=$(HOST_GID) \
	--build-arg host_user=$(USER) \