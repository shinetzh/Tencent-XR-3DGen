#!/bin/bash

model_key=${1:-"uv_mcwy"}
lj_port=${2:-"8986"}

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
out_dir="/aigc_cfs_gdp/sz/server/texgen_gdp/${model_key}/${current_time}"
echo "model_key: ${model_key}, save log in ${out_dir}"

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


### run BlenderCVTServer
export blender_root="/usr/blender-3.6.2-linux-x64/blender"
${blender_root} -b -P server_blendercvt.py &
host=localhost
port=987
max_attempts=30
attempt=0
while ! netstat -tulpn 2>/dev/null | grep -q ":$port"; do
    sleep 1
    attempt=$((attempt + 1))
    if ((attempt >= max_attempts)); then
        echo "Error: BlenderCVTServer did not start within the expected time"
        exit 1
    fi
    echo "waiting BlenderCVTServer."
done
echo "BlenderCVTServer run done."
echo "begin init texgen...."

### run TexGenServer
source /aigc_cfs_gdp/cfs1/sz/grpc/bin/activate

python server_texgen.py --cfg_json gdp_ser_tex_gen.json --lj_port ${lj_port} --model_key ${model_key} 