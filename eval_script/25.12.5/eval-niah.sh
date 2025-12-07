#!/bin/bash

root_path=/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105
# root_path=/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)

conda_path=${root_path}/miniconda/etc/profile.d/conda.sh
source ${conda_path}

echo "### start eval ###"
conda activate oc

which python

OC_PATH_ROOT=${root_path}/projects/ffa/huffkv-opencompass
cd $OC_PATH_ROOT

sleep 3

opencompass ${script_dir}/opencompass-eval-niah.py -w ${script_dir}/oc-eval-result/niah "$@"

sleep 3

python ${root_path}/occ.py