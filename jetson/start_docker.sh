#!/bin/bash
# ------------------------------------------------------------------
# Docker 실행 스크립트
# 이 스크립트는 필요한 Docker 이미지를 실행하고,
# 외부 SSD와 아두이노 장치를 컨테이너 내부로 연결
# ------------------------------------------------------------------
IMAGE_NAME="dustynv/transformers:r36.3.0"
HOST_DATA_PATH="/media/aiot/D0DE6417DE63F3DE"
ARDUINO_DEVICE="/dev/ttyACM0"

echo "================================================================="
echo "✅ Mood Light 컨테이너를 시작합니다."
echo "-----------------------------------------------------------------"
echo "  - 사용할 이미지: ${IMAGE_NAME}"
echo "  - 연결할 SSD 경로: ${HOST_DATA_PATH}"
echo "  - 연결할 아두이노: ${ARDUINO_DEVICE}"
echo "================================================================="

./run.sh --privileged \
    --volume=${HOST_DATA_PATH}:/mnt/data \
    --device=${ARDUINO_DEVICE}:${ARDUINO_DEVICE} \
    ${IMAGE_NAME}
