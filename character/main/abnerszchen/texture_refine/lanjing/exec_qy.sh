podname=$1
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

kubectl exec -it ${podname} -n ieg-aigc3d-4-game-gpu-qy --kubeconfig ./my.kubeconfig /bin/bash
# kubectl get pod -n ieg-aigc3d-4-game-gpu-qy --kubeconfig ./my.kubeconfig -o yaml | grep nvidia
