#!/bin/bash

cfg_json=${1:-"tex_gen.json"}
model_name=${2:-"uv_mcwy"}

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

echo "model_name: ${model_name}"

source /opt/conda/etc/profile.d/conda.sh
conda activate interface

########## run TexGenConsumer
echo "begin init texgen...."
python consumer_texgen.py --cfg_json ${cfg_json} --model_name ${model_name} 
echo "all done"

conda deactivate
