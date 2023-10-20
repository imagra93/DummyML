FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --fix-missing && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 git && \
    apt-get clean

ARG GITHUB_TOKEN

ENV GITHUB_TOKEN $GITHUB_TOKEN

RUN pip install -U poetry

ADD pyproject.toml .
ADD poetry.lock .

RUN poetry config virtualenvs.create false && poetry install

ENV PYTHONPATH '${PYTHONPATH}:/workspace'

RUN useradd -ms /bin/bash docker
USER docker
