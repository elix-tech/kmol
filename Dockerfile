FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update -y

RUN apt-get install -y wget build-essential
RUN apt-get install -y git

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Adding conda environemnt
RUN mkdir -p /kmol
COPY ./environment.yml /kmol/

# Setting miniconda environment
WORKDIR /kmol
RUN conda env create --file environment.yml
ENV CONDA_DEFAULT_ENV kmol
RUN conda init bash
RUN echo "conda activate kmol" >> ~/.bashrc
ENV PATH /opt/conda/envs/kmol/bin:$PATH

# installation of torch-geometric
RUN apt-get install -y build-essential
SHELL ["conda", "run", "-n", "kmol", "/bin/bash", "-c"]
RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
RUN pip install torch-geometric==1.6.3

# Adding content
COPY ./setup.cfg /kmol/
COPY ./pyproject.toml /kmol/
COPY ./README.md /kmol/
COPY .git /kmol/.git
COPY ./src /kmol/src
RUN pip install -e .

COPY ./docker/*.sh /kmol/
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["/bin/bash", "--login", "/kmol/run.sh"]
