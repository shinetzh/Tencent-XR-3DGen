#!/bin/bash
source /aigc_cfs_2/sz/grpc/bin/activate


codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}


current_time=$(date +"%Y-%m-%d_%H-%M")
out_dir="/aigc_cfs_3/sz/server/gradio/${current_time}"
echo "save gradio log in ${out_dir}"

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

python3.8 web_main.py