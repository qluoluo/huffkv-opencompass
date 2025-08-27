#!/bin/bash

# root_path=/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089
root_path=/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089
source ${root_path}/local_conda.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)


echo "### start eval ###"
conda activate zgliu-huffkv-opencompass
cd ${root_path}/projects_zgliu/projects/huffkv/huffkv-opencompass
sleep 3
opencompass ${script_dir}/opencompass-eval-longbench.py -w ${script_dir}/oc-eval-result/longbench -r
sleep 30