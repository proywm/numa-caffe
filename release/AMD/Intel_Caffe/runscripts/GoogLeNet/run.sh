OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $OUTDIR/out_NODE1_48B
export OMP_NUM_THREADS=12
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $OUTDIR/out_NODE2_48B

export OMP_NUM_THREADS=24
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $OUTDIR/out_NODE4_48B

export OMP_NUM_THREADS=48
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3,4,5,6,7 --membind 0,1,2,3,4,5,6,7 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $OUTDIR/out_NODE8_48B

