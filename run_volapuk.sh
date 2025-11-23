#!/usr/bin/bash

IMG="volapuk:latest"
CMD="bash"
xhost +local:docker
docker run --rm -it --gpus all \
                -p 5001:5001 -p 8000:8000 -p 8080:8080 -p 8888:8888 -p 11434:11434 \
                -e DISPLAY=$DISPLAY \
                -v $(pwd)/workspace:/workspace \
                -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
                $IMG \
                $CMD
