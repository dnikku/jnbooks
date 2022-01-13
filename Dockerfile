FROM debian:latest

# inspired from https://hub.docker.com/r/continuumio/miniconda3/dockerfile

ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"
ARG CONDA_SHA256="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"
ARG CONDA_DIR="/opt/conda"
ARG MY_USER=me
ARG MY_USER_PASS=a

ENV PATH="$CONDA_DIR/bin:$PATH"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update --fix-missing && \
    apt-get install -y bzip2 ca-certificates curl git tini && \
    # for debugging, list processes
    apt-get install -y procps psmisc && \
    # add user me as sudo
    apt-get install -y sudo && adduser $MY_USER && usermod -aG sudo $MY_USER && echo "$MY_USER:$MY_USER_PASS" | chpasswd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# install conda && jupyterlab, pip
RUN curl "$CONDA_URL" -o /tmp/miniconda.sh && echo "$CONDA_SHA256  /tmp/miniconda.sh" | sha256sum -c && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    conda install -c conda-forge pip jupyterlab numpy ipympl pandas scipy


USER $MY_USER

ENTRYPOINT [ "/usr/bin/tini", "--" ]
#CMD [ "tail", "-f", "/dev/null" ]

CMD [ "jupyter-lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.password=''", "--notebook-dir='/home/'"]







