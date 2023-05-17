# Simple way:

FROM python:3.9-slim-bullseye
ENV HOME /home

WORKDIR ${HOME}
ENV PYTHONPATH ${HOME}

COPY . .

RUN apt-get update && apt-get install -y gcc

RUN pip3 install -r requirements.txt    

ENTRYPOINT python3 app/api.py

# --------------------------------
# Leads to:
# Note: This error originates from the build backend, and is 
# likely not a problem with poetry but with lightfm (1.17) not supporting 
# PEP 517 builds. You can verify this by running 'pip wheel --use-pep517 "lightfm (==1.17)"'.
# Don't know how to solve error.

# FROM python:3.9-slim-bullseye

# ENV HOME /home
# WORKDIR ${HOME}
# ENV PYTHONPATH ${HOME}

# ENV POETRY_VERSION=1.4.2
# ENV POETRY_VENV=${HOME}/poetry-venv
# ENV POETRY_CACHE_DIR=${HOME}/.cache

# COPY . .

# RUN apt-get update && apt-get install -y gcc

# RUN python3 -m venv $POETRY_VENV \
#     && $POETRY_VENV/bin/pip install -U pip setuptools \
#     && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# ENV PATH="${PATH}:${POETRY_VENV}/bin"

# RUN poetry install

# ENTRYPOINT ["python3", "app/api.py"]

# --------------------------------