CDIR=$PWD
cd ../../
export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3 ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize56.prototxt -vd=0,1,2,3) &> $CDIR/out_NODE4_56B
export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0 --membind 0  ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize224.prototxt -vd=0) &> $CDIR/out_NODE1_224B
export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1 --membind 0,1 ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize112.prototxt -vd=0,1) &> $CDIR/out_NODE2_112B


#throughput
sh throughput.sh
