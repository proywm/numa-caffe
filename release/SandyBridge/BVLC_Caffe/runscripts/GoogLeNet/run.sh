OUTDIR=$PWD

cd ../../
export OMP_NUM_THREADS=32
export KMP_AFFINITY=verbose,scatter
(time  numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/out_NODE4_32B
export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter
(time   numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/out_NODE1_32B
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,scatter
(time  numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/out_NODE2_32B
