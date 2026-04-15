FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Pillow 等轮子多为 manylinux，一般无需系统库；若镜像内处理 TIFF/WebP 等异常再按需安装 libjpeg/zlib 等
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000"]
