CDIR=$PWD
cd ../../

export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,scatter

(time  numactl --cpunodebind 0,1,2,3 --membind 0,1,2,3  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize32.prototxt -vd=0,1,2,3) &> $CDIR/out_NODE4_32B

(time  numactl --cpunodebind 0,1 --membind 0,1  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize64.prototxt -vd=0,1) &> $CDIR/out_NODE2_64B

(time  numactl --cpunodebind 0 --membind 0  perf stat -e node-loads -e node-load-misses $CAFFEDIR/build/tools/caffe.bin train --solver=examples/cifar10/cifar10_full_solver_200_0T_bsize128.prototxt -vd=0) &> $CDIR/out_NODE1_128B

#throughput
sh throughput.sh
