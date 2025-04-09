codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

kubectl apply -f lanjing.yaml -n ieg-aigc3d-4-game-gpu-nj --kubeconfig my.kubeconfig
kubectl get pod -n ieg-aigc3d-4-game-gpu-nj --kubeconfig ./my.kubeconfig | grep shenzhou
# kubectl get multiclusterresourcequota -n ieg-aigc3d-4-game-gpu-nj -o yaml --kubeconfig ./my.kubeconfig
