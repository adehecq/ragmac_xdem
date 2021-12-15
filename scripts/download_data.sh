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
