indir=$1
outdir=$2
rclone copy ${indir} ${outdir} --multi-thread-streams=128 --transfers=128 -P
