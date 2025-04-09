#!/bin/bash
source /aigc_cfs_2/sz/grpc/bin/activate

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}
echo ${codedir}

python3.8 -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/texgen.proto

deactivate