#!/bin/bash
if [ -d "/aigc_cfs_2/sz/grpc" ]; then
    source /aigc_cfs_2/sz/grpc/bin/activate
    export blender_py="/aigc_cfs/sz/software/blender-3.6.2-linux-x64/3.6/python/bin/python3.10"
elif [ -d "/aigc_cfs_gdp/cfs1/sz/grpc" ]; then
    source /aigc_cfs_gdp/cfs1/sz/grpc/bin/activate
    export blender_py="/aigc_cfs_gdp/cfs1/sz/software/blender-3.6.2-linux-x64/3.6/python/bin/python3.10"
else
    echo "Error: Neither /aigc_cfs_2/sz/grpc nor /aigc_cfs_gdp/cfs1/sz/grpc directories exist."
    exit 1
fi

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}
echo ${codedir}

python -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/srgen.proto

# uv base
python -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/texgen.proto

# render-baking base
python -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/texcreator.proto

deactivate


${blender_py} -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/blendercvt.proto
