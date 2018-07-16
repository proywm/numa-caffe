CDIR=$PWD
cd ../../
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,scatter

(time numactl --cpunodebind 0 --membind 0 ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize256.prototxt -vd=0) &> $OUTDIR/out_NODE1_256B
(time numactl --cpunodebind 0,1 --membind 0,1 ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize128.prototxt -vd=0,1) &> $OUTDIR/out_NODE2_128B
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize64.prototxt -vd=0,1,2,3) &> $OUTDIR/out_NODE4_64B

#throughput
sh throughput.sh
