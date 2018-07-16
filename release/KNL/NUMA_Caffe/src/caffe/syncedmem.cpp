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

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
/*
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
*/
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
  case HEAD_AT_VDEV:
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
#if defined(VIRTDEV_ONLY)
      //own_cpu_data_ = true;
    head_ = SYNCED;
#else
    NO_GPU;
#endif
#endif
    break;
  case HEAD_AT_PRV:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    CHECK(prv_descriptor_.get());
    prv_descriptor_->convert_from_prv(cpu_ptr_);
    prv_descriptor_->on_to_cpu();
    head_ = SYNCED_PRV;
    break;
  case SYNCED_PRV:
  case HEAD_AT_CPU:
    if (prv_descriptor_.get()) {
        if ( prv_descriptor_->on_to_cpu())
            head_ = SYNCED;
    }
    break;
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_PRV:
    to_cpu();
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

inline void SyncedMemory::to_virtDev() {
#if defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CHECK(VirtDevManager::virtDevGetDevice(&virtDev_device_));
    if(NULL == cpu_ptr_)
    {
    	CHECK(VirtDevManager::virtDevMalloc(&cpu_ptr_, size_));
        caffe_memset(size_, 0, cpu_ptr_);
    }
    head_ = HEAD_AT_VDEV;
    own_virtDev_data_ = true;
    break;
  case HEAD_AT_CPU:
    //  CHECK(VirtDevManager::virtDevGetDevice(&virtDev_device_));
     // own_virtDev_data_ = true;
    //LOG(INFO) << "going to copy from cpu to virtdev " << cpu_ptr_ << " "<< virtDev_ptr_;
    //numauser: if virtDev is root device then dont need to copy cpu_prt_ to virtDev_ptr_
 //   caffe_virtDev_copy(size_, cpu_ptr_, virtDev_ptr_);
 //   if(virtDev_ptr_ != cpu_ptr_ || cpu_ptr_ == NULL)
     //  LOG(INFO) << "ERROR VDEV and CPU data location is not same " << cpu_ptr_ << " "<< virtDev_ptr_ << " head at cpu";
     //if(virtDev_ptr_!=cpu_ptr_)
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case HEAD_AT_VDEV:
  case SYNCED:
   //  if(virtDev_ptr_ == NULL)
       //  LOG(INFO) << "to_virtDev() ERROR VDEV and CPU data location is not same " << cpu_ptr_ << " "<< virtDev_ptr_;
    break;
  }
#else
  NO_GPU;
#endif
  
}

const void* SyncedMemory::cpu_data() {
  boost::mutex::scoped_lock lock(mtx);
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  boost::mutex::scoped_lock lock(mtx);
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  boost::mutex::scoped_lock lock(mtx);
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

const void* SyncedMemory::virtDev_data() {
 boost::mutex::scoped_lock lock(mtx);
#ifdef VIRTDEV_ONLY
  to_virtDev();
  //to_cpu();
  return (const void*)cpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  boost::mutex::scoped_lock lock(mtx);
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void SyncedMemory::set_virtDev_data(void* data) {
   boost::mutex::scoped_lock lock(mtx);
#if defined(VIRTDEV_ONLY)
  CHECK(data);
  if (own_virtDev_data_) {
     LOG(ERROR) << "Free ptr from virtDev Set Ptr";
    int initial_device;
    VirtDevManager::virtDevGetDevice(&initial_device);
    if (virtDev_device_ != -1) {
      CHECK(VirtDevManager::virtDevSetDevice(virtDev_device_));
    }
 //   CHECK(VirtDevManager::virtDevFree(cpu_ptr_));
    VirtDevManager::virtDevSetDevice(initial_device);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_VDEV;
  own_virtDev_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  boost::mutex::scoped_lock lock(mtx);
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void* SyncedMemory::mutable_virtDev_data() {
   boost::mutex::scoped_lock lock(mtx);
//#ifndef CPU_ONLY
#if defined(VIRTDEV_ONLY)
  //LOG(INFO) << virtDev_ptr_ << " from SyncedMemory::mutable_virtDev_data virtDev_ptr_ " << cpu_ptr_ << " cpu_ptr_";
  to_virtDev();
  //LOG(INFO) << virtDev_ptr_ << " from after SyncedMemory::mutable_virtDev_data virtDev_ptr_ " << cpu_ptr_ << " cpu_ptr_";
 // to_cpu();
  head_ = HEAD_AT_VDEV;
  return cpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}
#if !defined(CPU_ONLY) && !defined(VIRTDEV_ONLY)
//#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  boost::mutex::scoped_lock lock(mtx);
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
#if defined(VIRTDEV_ONLY)
void SyncedMemory::async_virtDev_push(const virtDevStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (cpu_ptr_ == NULL) {
	CHECK(VirtDevManager::virtDevGetDevice(&virtDev_device_));
     CHECK(VirtDevManager::virtDevMalloc(&cpu_ptr_, size_));
     own_virtDev_data_ = true;
  }
  head_ = SYNCED;
}
#endif
void SyncedMemory::set_prv_descriptor(shared_ptr<PrvMemDescr> descriptor,
        bool same_data) {
  // If it wasn't synced before, it won't be now.
  if (descriptor == NULL) {
    if (head_ != UNINITIALIZED)
      head_ = HEAD_AT_CPU;
  } else {
    if ((head_ != HEAD_AT_PRV) && same_data)
      head_ = SYNCED_PRV;
    else
      head_ = HEAD_AT_PRV;
  }

  prv_descriptor_ = descriptor;
}

const void* SyncedMemory::prv_data() {
  if ((head_ != HEAD_AT_PRV) &&
     (head_ != SYNCED_PRV)) {
    return NULL;
  }

  CHECK(prv_descriptor_.get());
  return (const void* ) prv_descriptor_->prv_ptr();
}

void* SyncedMemory::mutable_prv_data() {
  CHECK(prv_descriptor_.get());
  if (head_ == HEAD_AT_CPU) {
    prv_descriptor_->convert_to_prv(cpu_ptr_);
  }
  head_ = HEAD_AT_PRV;
  return prv_descriptor_->prv_ptr();
}

}  // namespace caffe