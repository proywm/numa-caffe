OUTDIR=$PWD

cd ../../
(time OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=compact,granularity=fine OMP_NUM_THREADS=48 $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/reproduced_out_Node8_48B
(time OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=compact,granularity=fine OMP_NUM_THREADS=24 $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/reproduced_out_Node4_48B
(time OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=compact,granularity=fine OMP_NUM_THREADS=12 $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/reproduced_out_Node2_48B
(time OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=compact,granularity=fine OMP_NUM_THREADS=6 $CAFFEDIR/build/tools/caffe.bin train --solver=models/bvlc_googlenet/solver.prototxt) &> $CDIR/reproduced_out_Node1_48B
