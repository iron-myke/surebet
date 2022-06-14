#!/usr/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/surebet/
conda activate surebet
date >> ~/tmp/runs.out
today=$(date +"%Y-%m-%d_%H:%M")
python track_leagues.py >> ~/tmp/output_${today}.out
