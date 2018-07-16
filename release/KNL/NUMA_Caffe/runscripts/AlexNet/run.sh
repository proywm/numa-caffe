OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0 --membind 0 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust.prototxt -vd=0) &> $OUTDIR/out_NODE1_256B
(time numactl --cpunodebind 0,1 --membind 0,1 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust_128.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_128B
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver_cust_64.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_64B
