FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG LOCAL_UID=1001
ARG LOCAL_GID=1001

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git/ssh are used by VS Code; no browser-facing notebook server is started.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/b3-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/b3-requirements.txt \
    && python -m pip check \
    && python -m ipykernel install --prefix=/opt/conda \
        --name b3cnnpart --display-name "Python (B3CNNpart)"

RUN groupadd --gid ${LOCAL_GID} student \
    && useradd --uid ${LOCAL_UID} --gid ${LOCAL_GID} --create-home --shell /bin/bash student \
    && mkdir -p /workspace/B3CNNpart \
    && chown student:student /workspace/B3CNNpart

COPY docker/smoke_test.py /opt/b3cnnpart/smoke_test.py
USER student
WORKDIR /workspace/B3CNNpart
CMD ["sleep", "infinity"]
