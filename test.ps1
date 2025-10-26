# Set your project folder
$projectPath = "C:\Users\James\Desktop\code\homework"

# Run deep learning container
docker run -it --rm `
  -v "${projectPath}:/app" `
  -w /app `
  --name fapp `
  pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime `
  /bin/bash
