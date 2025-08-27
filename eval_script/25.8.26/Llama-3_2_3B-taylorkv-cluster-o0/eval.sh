#!/bin/bash

# root_path=/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089
root_path=/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089
source ${root_path}/local_conda.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)

bash ${root_path}/eval-longbench.sh
bash ${root_path}/eval-niah.sh