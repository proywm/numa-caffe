OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter

(time  numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_8.prototxt -vd=0,1,2,3) &> $CDIR/out_NODE4_8B

(time  numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_16.prototxt -vd=0,1) &> $CDIR/out_NODE2_16B

(time  numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver_32.prototxt -vd=0) &> $CDIR/out_NODE1_32B


