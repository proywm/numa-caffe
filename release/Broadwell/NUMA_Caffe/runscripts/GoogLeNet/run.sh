OUTDIR=$PWD

cd ../../

export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter

export OMP_WAIT_POLICY=ACTIVE

(time numactl --cpunodebind 0 --membind 0 ./build/tools/caffe.bin train --solver=models/mkl2017_googlenet_v1_bdw/solver_112.prototxt -vd=0) &> $OUTDIR/out_NODE1_112B

export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1 --membind 0,1 ./build/tools/caffe.bin train --solver=models/mkl2017_googlenet_v1_bdw/solver_56.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_56B

export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 ./build/tools/caffe.bin train --solver=models/mkl2017_googlenet_v1_bdw/solver_28.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_28B

