#!/bin/sh
# Simple script to download RAGMAC data into the data/raw folder

# URL where to get the data from
DATA_URL=https://www.geo.uzh.ch/microsite/ragmac_experiment_data/files/

# Path to scripts base folder and target data directory
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
BASE_DIR=$(dirname $SCRIPT_DIR)
TARGET_DIR=$BASE_DIR/data/raw

# Check that target directory is empty, to not overwrite
if [ "$(ls -A $TARGET_DIR )" ]; then echo "Target directory not empty -> exiting"; exit; fi

# Download data
cmd="wget -r -nH --cut-dirs=3 --no-parent --reject='index.html*' --no-check-certificate $DATA_URL -P $TARGET_DIR"
echo -e "\n*** Running command: $cmd\n"; $cmd

# Unzipping archives
echo -e "\n*** Unzipping archives ***\n"
for f in `ls $TARGET_DIR/experiment_*/*/*zip`; do echo -e "\n$f"; unzip -n $f -d `dirname $f`; done

echo -e "\n*** Downloading finished ***\n"

# Force REF DEM for RU_FJL case to be sampled at exactly 30 m - the difference with original is less than 1e-10.
echo -e "Resampling REF_DEM for case RU_FJL"
gdalwarp -tr 30 30 data/raw/experiment_2/RU_FJL/RU_FJL_Copernicus_REF_DEM.tif data/raw/experiment_2/RU_FJL/RU_FJL_Copernicus_REF_DEM_30m.tif -r bilinear
