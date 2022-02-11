#!/bin/bash
# Simple utility to save all results to an archive

# cd to output folder
cwd=`dirname $0`
outdir=`realpath $cwd/../data/processed`
cd $outdir

# Output file
outfile="xdem_results.tar"
rm -f $outfile  # To avoid appending to existing file


# Archive all subfolders
for exp in experiment_1 experiment_2
do

    for dir in results_*
    do
	cmd="tar --append --file=$outfile $exp/*/$dir"
	echo $cmd; `$cmd`
    done
done

echo -e "Saved to $outdir/$outfile"
