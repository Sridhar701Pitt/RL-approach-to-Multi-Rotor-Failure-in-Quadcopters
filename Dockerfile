### https://dev.to/et813/install-docker-and-nvidia-container-m0j
## According to the site,
## Make sure you have installed the NVIDIA driver and Docker 19.03 for your Linux distribution Note that you do not need to install the CUDA toolkit on the host, but the driver needs to be installed.
## ^^ See this for running GUIs

## GPU version

FROM nvidia/cuda:11.6.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

# INSTALL OTHER NECESSARY PACKAGES

#install net-tools
RUN apt-get update
RUN apt-get install -y net-tools
RUN apt-get install -y nano

#This installation supports gui in matplotlib
# https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
RUN apt-get install -y python3-tk

# Video recording
RUN apt-get update
RUN apt-get install -y ffmpeg

#TMUX
RUN apt-get install -y tmux

#git, wget
RUN apt-get install -y git
RUN apt-get install -y wget
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

#torch stuff
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#install requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install --upgrade grpcio==1.43

# copy gym-pybullet
RUN mkdir /root/trainer
ADD . / /root/trainer/

#Set Work dir
WORKDIR /root
# initialize project
RUN cd trainer/ && pip3 install -e .
RUN chmod +x trainer/experiments/learning/singleagent.py
# # Google Cloud CLI - https://cloud.google.com/sdk/docs/quickstart#deb
# RUN apt-get install -y gcc curl
# RUN apt-get install -y apt-transport-https ca-certificates gnupg
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
# COPY ./potent-arcade-341204-013d447e4564.json /tmp/potent-arcade-341204-013d447e4564.json
# RUN gcloud auth activate-service-account --key-file=/tmp/potent-arcade-341204-013d447e4564.json
# # https://stackoverflow.com/questions/37428287/not-able-to-perform-gcloud-init-inside-dockerfile
# RUN gcloud config set project potent-arcade-341204

# CMD ["tail", "-f", "/dev/null"]
ENTRYPOINT ["python", "trainer/experiments/learning/singleagent.py"]
#ENTRYPOINT ["python3"]