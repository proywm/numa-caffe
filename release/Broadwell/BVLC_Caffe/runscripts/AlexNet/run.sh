OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter

export OMP_WAIT_POLICY=ACTIVE

(time numactl --cpunodebind 0 --membind 0 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver.prototxt) &> $OUTDIR/out_NODE1_224B

export OMP_NUM_THREADS=28
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1 --membind 0,1 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver.prototxt) &> $OUTDIR/out_NODE2_224B

export OMP_NUM_THREADS=56
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 ./build/tools/caffe.bin train --solver=models/bvlc_alexnet/solver.prototxt) &> $OUTDIR/out_NODE4_224B

