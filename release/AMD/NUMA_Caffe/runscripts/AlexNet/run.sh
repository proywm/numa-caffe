OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust.prototxt -vd=0) &> $OUTDIR/out_NODE1_256B

export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust_128B.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_128B

export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust_64B.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_64B



export OMP_NUM_THREADS=6
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3,4,5,6,7 --membind 0,1,2,3,4,5,6,7 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust_B32.prototxt -vd=0,1,2,3,4,5,6,7) &> $OUTDIR/out_NODE8_32B
