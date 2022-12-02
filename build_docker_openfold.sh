HOST_UID=$(id -u)
HOST_GID=$(id -g)

docker build -t openfold -f Dockerfile_openfold . \
    --build-arg host_uid=$HOST_UID \
	--build-arg host_gid=$HOST_GID \
	--build-arg host_user=$USER \