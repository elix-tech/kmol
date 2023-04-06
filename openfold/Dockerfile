FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

LABEL org.opencontainers.image.version = "1.0.0"
LABEL org.opencontainers.image.authors = "Elix, Inc"
LABEL org.opencontainers.image.source = "https://github.com/aqlaboratory/openfold"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04"

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y sudo wget libxml2 cuda-minimal-build-11-3 libcusparse-dev-11-3 libcublas-dev-11-3 libcusolver-dev-11-3 git && \
    apt-get -y autoremove && \
    apt-get -y autoclean && \
    apt-get -y clean && \
    /bin/rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget -O Miniconda3-latest-Linux-x86_64.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh

COPY 99-conda.sh /etc/profile.d/99-conda.sh
SHELL [ "/bin/bash", "--login", "-c" ]

RUN conda config --system --set auto_update_conda False

ARG HOST_UID=0
ARG HOST_GID=0
ARG HOST_USER=root
RUN [ "${HOST_UID}" != "0" ] && \
    addgroup --gid ${HOST_GID} openfold && \
    adduser --uid ${HOST_UID} --gid ${HOST_GID} --disabled-password --gecos "" ${HOST_USER}

RUN mkdir -p /opt/openfold && chown ${HOST_UID}:${HOST_GID} /opt/openfold

WORKDIR /opt/openfold
USER ${HOST_USER}

RUN git clone --shallow-submodules --recurse-submodules --branch main --depth 1 https://github.com/aqlaboratory/openfold.git .

RUN conda env create -f environment.yml && conda clean -y -a

RUN cd && (echo; echo "conda activate openfold_venv") >> .bashrc

RUN cd /opt/openfold/openfold/resources && \
  wget -O stereo_chemical_props.txt https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN conda activate openfold_venv && \
    export SITEDIR=$(python3 -c 'import site;print(site.getsitepackages()[0])') && \
    ( [ -r lib/openmm.patch ] && patch -p0 -d "${SITEDIR}" < lib/openmm.patch ) && \
    pip install -e .

COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
