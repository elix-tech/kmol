FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update -y

RUN apt-get install -y wget build-essential

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# It may be better to download these files via github after release
RUN mkdir -p /federated
COPY ./environment.yml /federated/
COPY ./*.py /federated/
COPY ./lib /federated/lib
COPY ./mila /federated/mila
COPY ./scripts /federated/scripts
COPY ./vendor /federated/vendor


# Setting miniconda environment
WORKDIR /federated
RUN conda env create --file environment.yml
ENV CONDA_DEFAULT_ENV federated
RUN conda init bash
RUN echo "conda activate federated" >> ~/.bashrc
ENV PATH /opt/conda/envs/federated/bin:$PATH

# installation of torch-geometric
RUN apt-get install -y build-essential
SHELL ["conda", "run", "-n", "federated", "/bin/bash", "-c"]
RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
RUN pip install torch-geometric==1.6.0


COPY ./docker/*.sh /federated/
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["/bin/bash", "--login", "/federated/run.sh"]

