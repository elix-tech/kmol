FROM ubuntu:20.04 as base

COPY scripts/install-packages.sh /usr/local/bin/install-packages.sh

RUN \
  /bin/echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/02apt-speedup &&\
  /bin/echo 'Acquire::http::No-Cache "true";' > /etc/apt/apt.conf.d/docker-no-cache &&\
  /bin/echo 'Apt::Install-Recommends "false";' > /etc/apt/apt.conf.d/docker-no-recommends

RUN /bin/sed -i 's/ main$/ main contrib non-free/g' /etc/apt/sources.list

ARG TIMEZONE="Asia/Tokyo"
RUN \
  /usr/local/bin/install-packages.sh tzdata &&\
  (/bin/echo "tzdata tzdata/Areas select ${TIMEZONE%/*}" | /usr/bin/debconf-set-selections) &&\
  (/bin/echo "tzdata tzdata/Zones/Asia select ${TIMEZONE#*/}" | /usr/bin/debconf-set-selections) &&\
  /bin/rm -f /etc/localtime /etc/timezone &&\
  /usr/sbin/dpkg-reconfigure -f noninteractive tzdata

RUN /usr/local/bin/install-packages.sh locales ca-certificates

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

RUN /usr/local/bin/install-packages.sh wget

RUN mkdir -p /opt/elix

ARG CONDA_VERSION=24.3.0-0
RUN bash -c '( wget -O ./Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/download/${CONDA_VERSION}/Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh && \
    chmod 755 ./Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh && \
    ./Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh -b -p /opt/elix-inc/py-runtime && \
    rm -f ./Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh )'

COPY scripts/setup /opt/elix-inc/py-runtime/setup
RUN chmod 755 /opt/elix-inc/py-runtime/setup

RUN bash -c '( \
 eval "$(/opt/elix-inc/py-runtime/setup)"; \
 enable-conda; \
 conda config --system --set notify_outdated_conda false; \
 conda config --system --set auto_update_conda false; \
 conda config --system --add envs_dirs /opt/envs )'

FROM base

ENV NVIDIA_VISIBLE_DEVICES=all
ENV MPLCONFIGDIR=/home/kmol/data/.mplconfig

RUN /usr/local/bin/install-packages.sh gosu wget build-essential libxml2-dev

COPY --from=conda-install /opt/elix-inc /opt/elix-inc

COPY scripts/setup /opt/elix-inc/py-runtime/setup
COPY scripts/99-conda.sh /etc/profile.d/99-conda.sh
RUN chmod 755 /opt/elix-inc/py-runtime/setup /etc/profile.d/99-conda.sh

RUN mkdir -p /opt/envs

COPY environment.yml /tmp/environment.yml
RUN bash -c '( \
 eval "$(/opt/elix-inc/py-runtime/setup)"; \
 enable-conda; \
 conda env create -f /tmp/environment.yml; \
 rm -f /tmp/environment.yml; \
 conda clean -a -f -y )'

COPY scripts/run_kmol.sh /opt/elix-inc/run_kmol.sh
RUN chmod 755 /opt/elix-inc/run_kmol.sh

RUN mkdir -p /opt/src
ADD kmol.tar.gz /opt/src

ENTRYPOINT [ "/opt/elix-inc/run_kmol.sh" ]
