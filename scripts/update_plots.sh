#!/bin/bash
## Utility to run experiments and update plots.
## Simplest solution for now since experiments check for processed products so no major reprocessing is done.
## TODO store all intermediate results needed for plotting on disk to update plots faster 

## Main experiments
python main_experiment1.py -c CH_Aletschgletscher
python main_experiment1.py -c AT_Hintereisferner
python main_experiment2.py -c PK_Baltoro

## Optional experiments
# python main_experiment1.py -c NO_Vestisen
# python main_experiment2.py -c RU_FJL
# python main_experiment2.py -c CL_NPI

# Move files to flat directory and zip
outdir="plots/mb_figs/"
mkdir -pv $outdir
find ../data/processed/ -name "*mb_fig.png" -type f | while read -r file
do
   new_file=${file/"../data/processed/"/}
   new_file=${new_file//\//_} #replace slashes with dashes
   cp "$file" "$outdir$new_file"
done

outdir="plots/ddem_figs/"
mkdir $outdir
find ../data/processed/ -name "*ddem_fig.png" -type f | while read -r file
do
   new_file=${file/"../data/processed/"/}
   new_file=${new_file//\//_} #replace slashes with dashes
   cp "$file" "$outdir$new_file"
done

tar cvzf plots.tar.gz plots/