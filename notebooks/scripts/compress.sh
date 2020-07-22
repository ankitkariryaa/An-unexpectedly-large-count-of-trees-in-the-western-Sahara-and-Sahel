#!/usr/bin/env bash

# Run the following code where the generated images are located
for i in det*.tif;
    do gdal_translate -co "TILED=YES" -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW -ot Byte ${i} `basename ${i} .tif`_com.tif;
done
