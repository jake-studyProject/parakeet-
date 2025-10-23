#!/bin/bash

pip install --upgrade huggingface_hub

mkdir -p models/parakeet-tdt-0.6b-v3-onnx

huggingface-cli download istupakov/parakeet-tdt-0.6b-v3-onnx \
  --local-dir models/ \
  --local-dir-use-symlinks False