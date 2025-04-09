indir=$1
outdir=$2
rclone copy ${indir} ${outdir} --multi-thread-streams=32 --transfers=128 
