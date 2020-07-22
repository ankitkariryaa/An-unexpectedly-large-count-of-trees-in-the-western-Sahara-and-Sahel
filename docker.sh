#!/usr/bin/env bash
docker run --runtime=nvidia --name treesInSahara -p 9876:8888 -v $PWD:/notebooks -d ankitkariryaa/keras-rasterio:2.1.1-gpu
