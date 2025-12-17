#!/bin/bash

root_path=/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105
global_user_path=/inspire/hdd/global_user/liuzhigeng-253108120105

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)

conda_path=${global_user_path}/miniconda/etc/profile.d/conda.sh
source ${conda_path}

echo "### start twilight eval ###"
conda activate twi

which python

OC_PATH_ROOT=${root_path}/projects/ffa/huffkv-opencompass
cd $OC_PATH_ROOT

sleep 3

opencompass ${script_dir}/opencompass-eval-twilight.py -w ${script_dir}/oc-eval-result/twilight "$@"

sleep 3

python ${root_path}/occ.py
