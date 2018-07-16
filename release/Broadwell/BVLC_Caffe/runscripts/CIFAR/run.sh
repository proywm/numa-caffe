CDIR=$PWD
cd ../../
export OMP_NUM_THREADS=56
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1,2,3 --interleave 0,1,2,3 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize112.prototxt) &> $CDIR/out_NODE4_112B
export OMP_NUM_THREADS=14
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0 --membind 0 perf stat -e node-loads -e node-load-misses  ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize112.prototxt) &> $CDIR/out_NODE1_112B
export OMP_NUM_THREADS=28
export KMP_AFFINITY=verbose,scatter
(time numactl --cpunodebind 0,1 --interleave 0,1 perf stat -e node-loads -e node-load-misses ./build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize112.prototxt) &> $CDIR/out_NODE2_112B

#throughput
sh throughput.sh
