codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

kubectl apply -f lanjing_qy.yaml -n ieg-aigc3d-4-game-gpu-qy --kubeconfig my.kubeconfig
kubectl get pod -n ieg-aigc3d-4-game-gpu-qy --kubeconfig ./my.kubeconfig | grep shenzhou
# kubectl get multiclusterresourcequota -n ieg-aigc3d-4-game-gpu-qy -o yaml --kubeconfig ./my.kubeconfig