indir=$1
outdir=$2
rclone copy ${indir} ${outdir} --transfers=128 
