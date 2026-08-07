FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Prevents Python stdout/stderr buffering (important for GC logs)
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV DIFFUSERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HOME=/home/user
ENV XDG_CACHE_HOME=/tmp/.cache
ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_CACHE=/tmp/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers

# Create a non-root user (Grand Challenge requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /tmp/.cache/huggingface/hub /tmp/.cache/huggingface/transformers \
    && chmod -R 1777 /tmp/.cache

WORKDIR /opt/app

# System dependencies required by SimpleITK/Pillow on runtime images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python dependencies (torch is already present in the base image)
COPY --chown=user:user requirements.txt /opt/app/
RUN pip install --no-cache-dir --requirement /opt/app/requirements.txt

# Algorithm code
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user runtime_lbm.py /opt/app/
COPY --chown=user:user submission_config.json /opt/app/

# Offline model assets staged by the notebook
COPY --chown=user:user resources/ /opt/app/resources/

# GC mounts /output as read-write; ensure it exists and is owned by our user
RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
