#!/bin/bash

root_path=/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)

conda_path=${root_path}/miniconda/etc/profile.d/conda.sh
source ${conda_path}

echo "### start quest eval ###"
conda activate oc

which python

OC_PATH_ROOT=${root_path}/projects/ffa/huffkv-opencompass
cd $OC_PATH_ROOT

sleep 3

opencompass ${script_dir}/opencompass-eval-quest.py -w ${script_dir}/oc-eval-result/quest "$@"

sleep 3

python ${root_path}/occ.py
