FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 triton==2.3.1 \
    --requirement /app/requirements.txt

COPY pipeline /app/pipeline

RUN chown -R appuser:appuser /app

ENV HOME=/home/appuser
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

USER appuser

RUN python3 -c "\
from pipeline import segmentation_settings as settings; \
import transformers; \
transformers.AutoProcessor.from_pretrained(settings.DETECTOR_MODEL_IDENTIFIER); \
transformers.AutoModelForZeroShotObjectDetection.from_pretrained(settings.DETECTOR_MODEL_IDENTIFIER); \
transformers.SamProcessor.from_pretrained(settings.SEGMENTOR_MODEL_IDENTIFIER); \
transformers.SamModel.from_pretrained(settings.SEGMENTOR_MODEL_IDENTIFIER)"

CMD ["sleep", "infinity"]
