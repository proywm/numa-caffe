OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter

(time  numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_64.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_64B

export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter

(time  numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_128.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_128B

export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter

(time  numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_256.prototxt -vd=0) &> $OUTDIR/out_NODE1_256B

