FROM ubuntu:20.04 as base

# Base set up and settings
RUN \
  /bin/echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/02apt-speedup &&\
  /bin/echo 'Acquire::http::No-Cache "true";' > /etc/apt/apt.conf.d/docker-no-cache &&\
  /bin/echo 'Apt::Install-Recommends "false";' > /etc/apt/apt.conf.d/docker-no-recommends

RUN /bin/sed -i 's/ main$/ main contrib non-free/g' /etc/apt/sources.list

ARG TIMEZONE="Asia/Tokyo"
RUN \
    apt-get install -y tzdata &&\
  (/bin/echo "tzdata tzdata/Areas select ${TIMEZONE%/*}" | /usr/bin/debconf-set-selections) &&\
  (/bin/echo "tzdata tzdata/Zones/Asia select ${TIMEZONE#*/}" | /usr/bin/debconf-set-selections) &&\
  /bin/rm -f /etc/localtime /etc/timezone &&\
  /usr/sbin/dpkg-reconfigure -f noninteractive tzdata

RUN apt-get install -y locales ca-certificates

RUN \
  for l in ja_JP.UTF-8 en_US.UTF-8; do sed -i "/${l}/s/^#[[:space:]]//" /etc/locale.gen; done && \
  /usr/sbin/locale-gen && \
  /usr/sbin/update-locale
  
ENV LC_ALL en_US.UTF-8

RUN mkdir -p /etc/xdg/pip && \
    ( echo "[global]"; \
      echo "no-cache-dir = true"; \
      echo "timeout = 60"; \
    ) > /etc/xdg/pip/pip.conf

# Downloading and setting up conda
FROM base as conda-install

RUN apt-get install -y wget

RUN mkdir -p /opt/elix

ARG ANACONDA_VERSION=2023.03
RUN wget -O Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh

RUN bash ./Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -p /opt/elix/anaconda3

ARG LOCAL_CONDA=/opt/elix/anaconda3/bin/conda

RUN ${LOCAL_CONDA} install -n base -y conda-libmamba-solver
RUN ${LOCAL_CONDA} config --system --set auto_update_conda False
RUN ${LOCAL_CONDA} config --system --add envs_dirs /opt/envs
RUN ${LOCAL_CONDA} config --system --set solver libmamba
RUN ${LOCAL_CONDA} clean -f -y -a

# Setting up the project and installing the base env
FROM base

ARG LOCAL_CONDA=/opt/elix/anaconda3/bin/conda
RUN apt-get install -y wget build-essential libxml2-dev

COPY --from=conda-install /opt/elix /opt/elix
COPY docker/scripts/setup /opt/elix/anaconda3/setup
RUN chmod 755 /opt/elix/anaconda3/setup

WORKDIR /opt/elix/kmol
SHELL [ "/bin/bash", "--login", "-c" ]

RUN mkdir -p /opt/envs /opt/elix/kmol
COPY --chown=root:root environment.yml /opt/elix/kmol/

# # Adding content
COPY manual_aggregator.yaml pyproject.toml setup.cfg setup.py /opt/elix/kmol/
COPY src/ /opt/elix/kmol/src
RUN rm -rf ./src/*.egg-info
# Clean up potential previous build env
RUN chown -R root /opt/elix/kmol
RUN find src -name __pycache__ -print0 | xargs -0 rm -rf
RUN find src -name '*.so' -print0 | xargs -0 rm -f
# Build base conda env
RUN ${LOCAL_CONDA} env create -f environment.yml

RUN chown -R root:root /opt/elix/kmol

ENV NVIDIA_VISIBLE_DEVICE=all

COPY docker/run.sh /opt/elix/kmol/run.sh
COPY docker/install-venv.sh /opt/elix/kmol/install-venv.sh
ENTRYPOINT [ "bash", "/opt/elix/kmol/run.sh" ]