/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new string("empty buffer"));
  }
}

DataReader::QueuePair::~QueuePair() {
  string* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  const caffe::DataParameter *data_param = &param_.data_param();
  CHECK(data_param) << "Failed to obtain data_param";

  shared_ptr<DBWrapper> dbw(data_param->shuffle() ?
                        static_cast<DBWrapper*>(new DBShuffle(param_)):
                        static_cast<DBWrapper*>(new DBSequential(param_)));

  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    VirtDevManager::virtDevAllocationLoc.lock();
    int orgCPU =   VirtDevManager::virtDevGetNextDataReaderCore();//param_.device_id());//param_.device_id() ;//sched_getcpu();
    VirtDevManager::virtDevAllocationLoc.unlock();
    LOG(INFO) << "inside DATAREADER " << solver_count;
    for (int i = 0; i < solver_count; ++i) {
      //before allocating change cpu to set NUMA Node
      //done affinity setting
#if 1
         cpu_set_t my_set;        /* Define your cpu_set bit mask. */
       CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
       int numaCPU = i*COARSE_THREADS;
       CPU_SET(numaCPU, &my_set);     /* set the bit that represents core 7. */
       sched_setaffinity(0, sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
#endif
	LOG(INFO) << "NUMA DOMAIN " << numaCPU%2;
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(dbw.get(), qp.get());
      qps.push_back(qp);
    }
    //going back to original cpuid
#if 1
    cpu_set_t my_set;        /* Define your cpu_set bit mask. */
       CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
       CPU_SET(orgCPU, &my_set);     /* set the bit that represents core 7. */
       sched_setaffinity(getpid(), sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
#endif
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(dbw.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(DBWrapper* dbw, QueuePair* qp) {
  CHECK(dbw);
  CHECK(qp);

  string* data = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  *data = dbw->value();
  qp->full_.push(data);

  dbw->Next();
}



DataReader::DBWrapper::DBWrapper(const LayerParameter& param) {
  db.reset(db::GetDB(param.data_param().backend()));
  db->Open(param.data_param().source(), db::READ);
  cursor.reset(db->NewCursor());
}

DataReader::DBShuffle::DBShuffle(const LayerParameter& param):DBWrapper(param) {
  CHECK(param.data_param().backend() != DataParameter_DB_LEVELDB)
                                      << "LevelDB doesn't support shuffle";
  while (cursor->valid()) {
    image_pointers_.push_back(cursor->valuePointer());
    cursor->Next();
  }
  CHECK(!image_pointers_.empty());
  current_image_ = image_pointers_.begin();

  // randomly shuffle data
  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();
}

void DataReader::DBShuffle::Next() {
  current_image_++;
  if (current_image_ == image_pointers_.end()) {
    ShuffleImages();
    current_image_ = image_pointers_.begin();
  }
}

void DataReader::DBShuffle::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_pointers_.begin(), image_pointers_.end(), prefetch_rng);
}

void DataReader::DBSequential::Next() {
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}



}  // namespace caffe
