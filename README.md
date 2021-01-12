# FederatedLighting

## Installation

You only need [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the code.

```
docker run -d -p 8080:8080 --gpus all --env INCLUDE_TUTORIALS=false -v <repository directory>:/workspace  matech96/ml-workspace
```

## Custom model and data
