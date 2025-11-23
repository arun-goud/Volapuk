#!/usr/bin/env bash

docker exec -it `docker ps -aq --filter "ancestor=volapuk"` ollama serve
