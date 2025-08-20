#!/bin/bash
# root_path=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089
source /inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/public/resurrection.sh

root_path=/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089
source ${root_path}/local_conda.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)


echo "### start eval ###"
conda activate zgliu-huffkv-opencompass
cd ${root_path}/projects_zgliu/projects/huffKV/opencompass
sleep 3
opencompass ${script_dir}/opencompass-eval-niah.py -w ${script_dir}/oc-eval-result/niah -r
sleep 30