FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV USER=ml
ENV GROUP=mlusers
ENV LANG=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8
ENV LC_ALL=C
ENV SHELL /bin/bash
ENV HOME /home/$USER

WORKDIR $HOME
USER root


#
# Python3.9
#
RUN apt-get -y update \
    && apt-get -y --no-install-recommends install \
    software-properties-common vim \
    python3-pip python3.9 ipython3 python3-venv
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

COPY requirements-linux.txt requirements-linux.txt
RUN pip install -r requirements-linux.txt
RUN rm requirements-linux.txt

#CMD ["tail", "-f", "/dev/null"]
