#!/bin/bash
source /aigc_cfs_2/sz/grpc/bin/activate


codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

model_key=${1:-"mcwy"}

current_time=$(date +"%Y-%m-%d_%H-%M")
out_dir="/aigc_cfs_3/sz/server/teximguv/${model_key}/${current_time}"
echo "model_key: ${model_key}, save log in ${out_dir}"

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


python3.8 server_teximguv.py --model_key ${model_key}