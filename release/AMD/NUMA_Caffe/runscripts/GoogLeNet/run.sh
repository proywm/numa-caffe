OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_48.prototxt -vd=0) &> $OUTDIR/out_NODE1_48B


(time numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_24.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_24B

(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_12.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_12B

(time numactl --cpunodebind 0,1,2,3,4,5,6,7 --membind 0,1,2,3,4,5,6,7 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_6.prototxt -vd=0,1,2,3,4,5,6,7) &> $OUTDIR/out_NODE8_6B

