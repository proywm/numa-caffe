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
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include <omp.h>

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_virtDev,
  replace_cpu_diff,
  replace_gpu_diff,
  replace_virtDev_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_virtDev:
     //   LOG(INFO) << "applying buffer by replacing VirtDev pthread ID" << pthread_self() << " DevID: "<<dev << " ptr " << ptr;
        blobs[i]->data()->set_virtDev_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
      case replace_virtDev_diff:
    //    LOG(INFO) << "applying buffer by replacing VirtDev Diff pthread ID" << pthread_self() << " DevID: "<<dev << " ptr " << ptr;
 	blobs[i]->diff()->set_virtDev_data(ptr);
	break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
}

template<typename Dtype>
VIRTDEVParams<Dtype>::VIRTDEVParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifdef VIRTDEV_ONLY
  int initial_device;
  CHECK(VirtDevManager::virtDevGetDevice(&initial_device));

  // Allocate device buffers
  CHECK(VirtDevManager::virtDevSetDevice(device));
  CHECK(VirtDevManager::virtDevMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  //do we need some locking?
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  CHECK(VirtDevManager::virtDevMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_set(size_, Dtype(0), diff_);
  CHECK(VirtDevManager::virtDevMalloc(&pdata_, size_ * sizeof(Dtype)));

  CHECK(VirtDevManager::virtDevSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
VIRTDEVParams<Dtype>::~VIRTDEVParams() {
#ifdef VIRTDEV_ONLY
  CHECK(VirtDevManager::virtDevFree(data_));
  CHECK(VirtDevManager::virtDevFree(diff_));
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

template<typename Dtype>
void VIRTDEVParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_virtDev);
  apply_buffers(net, diff_, size_, replace_virtDev_diff);
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        cudaDeviceProp a, b;
        CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
        CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
        if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        CUDA_CHECK(
            cudaDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#elif defined(VIRTDEV_ONLY)
#if 0
  vector<int> remaining(devices);
  for (int d = 1; d < remaining.size(); ++d) {
  	pairs->push_back(DevicePair(remaining[0], remaining[d]));
        DLOG(INFO) << "VirtDev : " << remaining[0] << ":" << remaining[d];
  }
  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));
#else
  vector<int> remaining(devices);
  for (int d = 0; d < remaining.size(); ++d) {
        float index = d;
        int numaDomainRoot = floor(index / CORES_PER_NUMA) * CORES_PER_NUMA;
        if(numaDomainRoot!=d)
        {
                pairs->push_back(DevicePair(remaining[numaDomainRoot+floor(((float)(d%CORES_PER_NUMA)+1.0)/2)]-1, remaining[d]));
                DLOG(INFO) << "VirtDev : " << remaining[numaDomainRoot] << ":" << remaining[d];
        }
        else if((numaDomainRoot == d) && (numaDomainRoot != 0))
        {
                pairs->push_back(DevicePair(remaining[0], remaining[d]));
                DLOG(INFO) << "VirtDev : " << remaining[0] << ":" << remaining[d];
        }
  }
  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));
#endif
#else
  NO_GPU;
#endif
}

//
template<typename Dtype>
V2VSync<Dtype>::V2VSync(shared_ptr<Solver<Dtype> > root_solver,
                        V2VSync<Dtype>* parent, const SolverParameter& param)
    : VIRTDEVParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifdef VIRTDEV_ONLY
  int initial_device;
  CHECK(VirtDevManager::virtDevGetDevice(&initial_device));
  const int self = param.device_id();
  CHECK(VirtDevManager::virtDevSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CHECK(VirtDevManager::virtDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CHECK(VirtDevManager::virtDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "Virtual Device " << self << " does not have V2V access to Virtual Device " << peer;
    }
    // Allocate receiving buffer on parent
    CHECK(VirtDevManager::virtDevSetDevice(peer));
    CHECK(VirtDevManager::virtDevMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CHECK(VirtDevManager::virtDevSetDevice(self));
  }

  CHECK(VirtDevManager::virtDevSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
V2VSync<Dtype>::~V2VSync() {
  LOG(ERROR) << "CAME HERE IN ~V2VSync";
#ifdef VIRTDEV_ONLY
  int initial_device;
  CHECK(VirtDevManager::virtDevGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CHECK(VirtDevManager::virtDevSetDevice(self));

  if (parent_) {
    CHECK(VirtDevManager::virtDevFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CHECK(VirtDevManager::virtDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CHECK(VirtDevManager::virtDeviceDisablePeerAccess(peer));
    }
  }

  CHECK(VirtDevManager::virtDevSetDevice(initial_device));
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void V2VSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
   LOG(INFO)<< " solver_->param().device_id() " <<  solver_->param().device_id() << " root_solver " << Caffe::root_solver() << " thread ID "<< pthread_self();
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  //setting up CPU 
#if 0
   VirtDevManager::virtDevAllocationLoc.lock();
   int orgCPU =   VirtDevManager::virtDevGetNextDataReaderCore();
   VirtDevManager::virtDevAllocationLoc.unlock();
   cpu_set_t my_set;        /* Define your cpu_set bit mask. */
   CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
   CPU_SET(orgCPU, &my_set);     /* set the bit that represents core 7. */
   sched_setaffinity(getpid(), sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
#endif

  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}


template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void V2VSync<Dtype>::on_start() {
#ifdef VIRTDEV_ONLY
#ifdef DEBUG
  int device;
  CHECK(VirtDevManager::virtDevGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    V2VSync<Dtype> *parent = queue_.pop("on_start waiting to copy data to parent"); 
    //syncing here
 //   if(queue_.pop(&parent))
    {
    	CHECK(parent == parent_);
	//copy from pdata to data
	//VirtDevManager::virtDevAllocationLoc.lock();
        CHECK(VirtDevManager::virtDevMemcpy(data_, parent_->data_, size_ * sizeof(Dtype), VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice));
	//VirtDevManager::virtDevAllocationLoc.unlock();
    }
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->pdata_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    //CHECK(VirtDevManager::virtDevMemcpyAsync(dst, src, size_ * sizeof(Dtype),
      //  VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice, VirtDevStreamKind::virtDevStreamDefault));
    //VirtDevManager::virtDevAllocationLoc.lock();
    //CHECK(VirtDevManager::virtDevMemcpy(dst, src, size_ * sizeof(Dtype),
      //  VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice));
    //VirtDevManager::virtDevAllocationLoc.unlock();
    //CHECK(VirtDevManager::virtDevStreamSynchronize(VirtDevStreamKind::virtDevStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void V2VSync<Dtype>::on_gradients_ready() {
#ifdef VIRTDEV_ONLY
#ifdef DEBUG
  int device;
  CHECK(VirtDevManager::virtDevGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif
//   LOG(info) << " on_gradients_ready "
 //  LOG(INFO)<< " on_gradients_ready Function Device Id " << solver_->param().device_id() << " cpu id " << sched_getcpu() << " thread ID "<< pthread_self();

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    V2VSync<Dtype> *child = queue_.pop("on_gradients_ready waiting to copy gradients from children");
    //syncing here
    //if(queue_.pop(&child))
    {
    Dtype* src = child->diff_;//parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    virtPointerAttributes attributes;
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif
   // LOG(ERROR)<< " on_gradients_ready Function Adding child's data Device Id " << solver_->param().device_id() << " cpu id " << sched_getcpu() << " thread ID "<< pthread_self();
    VirtDevManager::virtDevAllocationLoc.lock();
    caffe_add(size_, src, dst, dst);
    VirtDevManager::virtDevAllocationLoc.unlock();
    }
  }

  // Send gradients to parent
  if (parent_) {
    //Dtype* src = diff_;
   // Dtype* dst = parent_grads_;

#ifdef DEBUG
    virtPointerAttributes attributes;
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CHECK(VirtDevManager::virtDevPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif
    //replace this with pointer
   //CHECK(VirtDevManager::virtDevMemcpyAsync(dst, src, size_ * sizeof(Dtype),  
     //   VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice, VirtDevStreamKind::virtDevStreamDefault));
  //  VirtDevManager::virtDevAllocationLoc.lock();
/*
    if(!omp_in_parallel())
    {
        #pragma omp parallel num_threads(8)
        {
		int num_of_threads_ = omp_get_max_threads();
                while(num_of_threads_ !=1)
                {
                   if(0 == size_%num_of_threads_)
                     break;
                   else
                     num_of_threads_ = num_of_threads_ / 2;
                }
           #pragma omp parallel for
           for(int iter = 0; iter < num_of_threads_; iter++)
           {
               CHECK(VirtDevManager::virtDevMemcpy(dst+iter*(size_/num_of_threads_), src+iter*(size_/num_of_threads_), (size_/num_of_threads_)*sizeof(Dtype), VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice));
           }
        }
    }
    else
*/
		//CHECK(VirtDevManager::virtDevMemcpy(dst, src, size_ * sizeof(Dtype), VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice));
//    VirtDevManager::virtDevAllocationLoc.unlock();
    //CHECK(VirtDevManager::virtDevStreamSynchronize(VirtDevStreamKind::virtDevStreamDefault));
 //    LOG(INFO)<< " on_gradients_ready Function Adding child's data Device Id " << solver_->param().device_id() << " cpu id " << sched_getcpu() << " thread ID "<< pthread_self();
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}

template<typename Dtype>
void V2VSync<Dtype>::Run(const vector<int>& virtDevs) {
  vector<shared_ptr<V2VSync<Dtype> > > syncs(virtDevs.size());
  Prepare(virtDevs, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();
  
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

template<typename Dtype>
void V2VSync<Dtype>::Prepare(const vector<int>& virtDevs,vector<shared_ptr<V2VSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(virtDevs, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "Virtual pairs " << s.str();

  SolverParameter param(solver_->param());
  //LOG(INFO)<< "After Solver Param Set " << s.str();
  //vector<shared_ptr<V2VSync<Dtype> > > syncs(virtDevs.size());
  //LOG(INFO)<< "After Syncs Set " << s.str();

  // Build the Virtual Device tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        V2VSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          V2VSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new V2VSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((V2VSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);
INSTANTIATE_CLASS(VIRTDEVParams);
INSTANTIATE_CLASS(V2VSync);

}  // namespace caffe
