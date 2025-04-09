#!/bin/bash
source /aigc_cfs_2/sz/grpc/bin/activate

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}
echo ${codedir}

# old version. useless
python3.8 -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/texcreator.proto
python3.8 -m grpc_tools.protoc -I ../protos --python_out=.. --pyi_out=.. --grpc_python_out=.. ../protos/teximguv.proto

deactivate


