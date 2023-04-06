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

SHELL [ "/bin/bash", "--login", "-c" ]

RUN /usr/local/bin/install-packages.sh wget

RUN mkdir -p /opt/elix

ARG ANACONDA_VERSION=2023.03
RUN bash -c '( wget -O ./Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh && \
  chmod 755 ./Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh && \
  ./Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -p /opt/elix/anaconda3;)'

COPY scripts/setup /opt/elix/anaconda3/setup
RUN chmod 755 /opt/elix/anaconda3/setup

RUN ( eval "$(/opt/elix/anaconda3/setup)"; \
 enable-conda; \
 conda install -n base -y conda-libmamba-solver; \
 conda config --system --set auto_update_conda False; \
 conda config --system --add envs_dirs /opt/envs; \
 conda config --system --set solver libmamba; \
 conda clean -f -y -a )

FROM base

ENV NVIDIA_VISIBLE_DEVICE=all

RUN /usr/local/bin/install-packages.sh wget build-essential libxml2-dev

SHELL [ "/bin/bash", "--login", "-c" ]

COPY --from=conda-install /opt/elix /opt/elix
WORKDIR /opt/elix/kmol

COPY scripts/setup /opt/elix/anaconda3/setup
RUN chmod 755 /opt/elix/anaconda3/setup

RUN mkdir -p /opt/envs /opt/elix/kmol

COPY environment.yml /opt/elix/kmol
RUN ( eval "$(/opt/elix/anaconda3/setup)"; enable-conda ; conda env create -f environment.yml )

ADD kmol.tar.gz /opt/elix/kmol
RUN chown -R root:root /opt/elix/kmol

COPY run.sh /opt/elix/kmol/run.sh
COPY install-venv.sh /opt/elix/kmol/install-venv.sh

ENTRYPOINT [ "bash", "/opt/elix/kmol/run.sh" ]
