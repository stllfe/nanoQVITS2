#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "LJSpeech *.{wav,txt} files directory should be specified!"
    exit 1
fi

if [[ -z $2 ]]; then
    echo "Output alignment directory should be specified!"
    exit 1
fi

mkdir -p $2

FILES_DIR=$(realpath $1)
ALIGN_DIR=$(realpath $2)

MFA_CACHE_DIR="$(pwd)/.cache"
MFA_IMAGE=mmcauliffe/montreal-forced-aligner:v3.0.0a8
MFA_MODEL=english_mfa

echo "Pulling MFA from Docker Hub..."
docker image pull $MFA_IMAGE
echo "Done!"

echo "Running MFA inside a directory:"
echo $FILES_DIR
docker run -it \
    -v $FILES_DIR:/data \
    -v $ALIGN_DIR:/alignment \
    -v $MFA_CACHE_DIR:/mfa/pretrained_models $MFA_IMAGE \
    /bin/bash -c \
    "mfa model download acoustic $MFA_MODEL && mfa align -j 4 --single_speaker --clean /data $MFA_MODEL $MFA_MODEL /alignment"
echo "Done!"
echo "Alignments should be located here:"
echo $ALIGN_DIR
