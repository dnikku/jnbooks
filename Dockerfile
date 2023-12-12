FROM debian:latest as builder


# inspired from https://hub.docker.com/r/continuumio/miniconda3/dockerfile

ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh"
ARG CONDA_SHA256="d0643508fa49105552c94a523529f4474f91730d3e0d1f168f1700c43ae67595"
ARG CONDA_DIR="/opt/conda"

ENV PATH="$CONDA_DIR/bin:$PATH"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update --fix-missing
RUN apt-get install -y bzip2 ca-certificates curl git tini


# install conda && jupyterlab, pip
RUN curl "$CONDA_URL" -o /tmp/miniconda.sh
RUN echo "$CONDA_SHA256  /tmp/miniconda.sh" | sha256sum -c
RUN /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR
RUN rm /tmp/miniconda.sh

RUN conda install -y -c conda-forge pip jupyterlab numpy ipympl pandas scipy

# install numpyro for statistical rethinking https://fehiepsi.github.io/rethinking-numpyro/
RUN conda install -y -c conda-forge numpyro
RUN pip install arviz causalgraphicalmodels daft


RUN apt-get install -y sudo

# for debugging, list processes
RUN apt-get install -y procps psmisc

# set timezone
RUN ln -sf /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# clean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN /opt/conda/bin/conda clean -tipy


# dump actual package installed
RUN mkdir -p /home
RUN conda list --export > /home/package-list.txt
RUN pip freeze > /home/requirements.txt


# https://stackoverflow.com/questions/56117261/how-to-merge-dockers-layers-of-image-and-slim-down-the-image-file
FROM debian:latest

ARG MY_USER=ndascalu
ARG MY_USER_UID=1000

COPY --from=builder / /

# add user MY_USER as sudo
RUN adduser --uid $MY_USER_UID $MY_USER
RUN echo "\n$MY_USER     ALL=(ALL) NOPASSWD:ALL"

USER $MY_USER
WORKDIR /home/$MY_USER

RUN echo "\n/opt/conda/bin/jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/home/nbooks" \
    > start.sh && chmod +x start.sh

# ENTRYPOINT sets the process to run
ENTRYPOINT [ "/usr/bin/tini", "--"]

# while CMD supplies default arguments to that process
#CMD [ "sh", "-c", "echo 'sleeping...'; sleep 300s" ]
CMD [ "./start.sh" ]


