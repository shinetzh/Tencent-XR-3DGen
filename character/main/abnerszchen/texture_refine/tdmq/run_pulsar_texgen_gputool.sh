#!/bin/bash

# try mount cos
if [ ! -d "/mnt/aigc_bucket_4/AIGC/" ]; then
  bash /usr/mount_ceph_cos.sh
  echo "add mount cos done"
fi
if [ ! -d "/mnt/aigc_bucket_4/AIGC/" ]; then
  echo "mount cos failed!!!!"
  exit 1
fi


cfg_json=${1:-"tex_gen.json"}
model_name=${2:-"uv_mcwy"}
export blender_root="/usr/blender-3.6.2-linux-x64/blender"

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "codedir=${codedir}"
cd ${codedir}

bash ./kill_pulsar_texgen.sh

current_time=$(date +"%Y-%m-%d_%H-%M-%S-%3N")
out_dir="/aigc_cfs_gdp/sz/pulsar/texgen_gdp/${model_name}/${current_time}"
echo "model_name: ${model_name}, save log in ${out_dir}"

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

source /opt/conda/etc/profile.d/conda.sh
conda activate interface

########## run BlenderCVTServer
echo "begin init BlenderCVTServer...."
${blender_root} -b -P "${codedir}/../grpc_backend/server_blendercvt.py" &
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

# 以后台模式每周重启 server_blendercvt.py
restart_server_blendercvt() {
  while true; do
    sleep 604800
    pkill -9 server_blendercvt.py
    pkill -9 blender
    sleep 10
    if ! (netstat -tulpn 2>/dev/null | grep -q ":$port"); then
        echo "restart server_blendercvt"
        ${blender_root} -b -P "${codedir}/../grpc_backend/server_blendercvt.py" &
    fi
  done
}

cleanup() {
  kill $restart_server_blendercvt_pid
  exit
}

trap cleanup SIGINT SIGTERM
restart_server_blendercvt &
restart_server_blendercvt_pid=$!

echo "begin init gputool...."
python ${codedir}/../blender_render_gif/tdmq_consumer_gputool.py --cfg_json ${codedir}/../blender_render_gif/configs/tdmq_gputool.json &

########## run TexGenConsumer
echo "begin init texgen...."
python consumer_texgen.py --cfg_json ${cfg_json} --model_name ${model_name} 
echo "all done"

conda deactivate
cleanup
