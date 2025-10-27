# /bin/bash


# Set your project folder
projectPath="/home/james/Desktop/code/homework"


# Run deep learning container
docker run -it --rm \
  -v "${projectPath}:/app" \
  -w /app \
  --name happ \
  pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime \
  /bin/bash
