OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter

export OMP_WAIT_POLICY=ACTIVE

(time numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_112.prototxt) &> $OUTDIR/out_NODE1_112B

export OMP_NUM_THREADS=28
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_112.prototxt) &> $OUTDIR/out_NODE2_112B

export OMP_NUM_THREADS=56
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_112.prototxt) &> $OUTDIR/out_NODE4_112B

