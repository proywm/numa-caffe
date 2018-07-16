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

#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }
  inline Dtype* pdata() const {
    return pdata_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient
  Dtype* pdata_;                 // Network Parameters

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::pdata_;
};

// Params stored in VirtDev memory.
template<typename Dtype>
class VIRTDEVParams : public Params<Dtype> {
 public:
  VIRTDEVParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~VIRTDEVParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::pdata_;
};

class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Synchronous data parallelism using map-reduce between local GPUs.
template<typename Dtype>
class P2PSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback,
    public InternalThread {
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                   P2PSync<Dtype>* parent, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  void Prepare(const vector<int>& gpus,
               vector<shared_ptr<P2PSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }

 protected:
  void on_start();
  void on_gradients_ready();

  void InternalThreadEntry();

  P2PSync<Dtype>* parent_;
  vector<P2PSync<Dtype>*> children_;
  BlockingQueue<P2PSync<Dtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::pdata_;
};

// Synchronous data parallelism using map-reduce between local VirtDevs.
template<typename Dtype>
class V2VSync : public VIRTDEVParams<Dtype>, public Solver<Dtype>::Callback,
    public InternalThread {
 public:
  explicit V2VSync(shared_ptr<Solver<Dtype> > root_solver,
                   V2VSync<Dtype>* parent, const SolverParameter& param);
  virtual ~V2VSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& virtDevs);
  void Prepare(const vector<int>& virtDevs,
               vector<shared_ptr<V2VSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }
 protected:
  void on_start();
  void on_gradients_ready();

  void InternalThreadEntry();

  V2VSync<Dtype>* parent_;
  vector<V2VSync<Dtype>*> children_;
  BlockingQueue<V2VSync<Dtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::pdata_;
};

}  // namespace caffe

#endif
