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

#ifndef CAFFE_MKL2017_LAYERS_HPP_
#define CAFFE_MKL2017_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/mkl_memory.hpp"
#include "mkl_dnn_cppwrapper.h"

namespace caffe {

template <typename Dtype>
class MKLConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit MKLConvolutionLayer(const LayerParameter& param);

  virtual ~MKLConvolutionLayer();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // Customized methods
  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();

  void Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

 private:
  /* Fwd step */
  shared_ptr<MKLData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                 fwd_bias_data;
  dnnPrimitive_t convolutionFwd;

  /* Bwd data step */
  shared_ptr<MKLDiff<Dtype> > bwdd_top_diff, bwdd_bottom_diff;
  shared_ptr<MKLData<Dtype> > bwdd_filter_data;
  dnnPrimitive_t convolutionBwdData;

  /* Bwd filter step */
  shared_ptr<MKLDiff<Dtype> > bwdf_top_diff, bwdf_filter_diff;
  shared_ptr<MKLDiff<Dtype> > bwdf2fwd_filter_diff;
  shared_ptr<MKLData<Dtype> > bwdf_bottom_data;
  dnnPrimitive_t convolutionBwdFilter;

  /* Bwd bias step */
  shared_ptr<MKLDiff<Dtype> > bwdb_top_diff, bwdb_bias_diff;
  dnnPrimitive_t convolutionBwdBias;

  /* In case of (iter_size > 1) we need additional buffers */
  shared_ptr<MKLDiff<Dtype> > bwdf_filter_diff_iter,
                              bwdb_bias_diff_iter;

  // TODO: temp. compatibility vs. older cafe
  size_t width_,
         height_,
         width_out_,
         height_out_,
         kernel_w_,
         kernel_h_,
         stride_w_,
         stride_h_;
  int    pad_w_,
         pad_h_;
};


/**
 * @brief Normalize the input in a local region across feature maps.
 */

template <typename Dtype>
class MKLLRNLayer : public Layer<Dtype> {
 public:
  explicit MKLLRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        lrnFwd(static_cast<dnnPrimitive_t>(NULL)),
        lrnBwd(static_cast<dnnPrimitive_t>(NULL)),
        fwd_top_data    (new MKLData<Dtype>()),
        fwd_bottom_data (new MKLData<Dtype>()),
        bwd_top_diff    (new MKLDiff<Dtype>()),
        bwd_bottom_diff (new MKLDiff<Dtype>()),
        lrn_buffer_(static_cast<Dtype*>(NULL)) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~MKLLRNLayer();

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;
  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
 private:
  dnnPrimitive_t lrnFwd, lrnBwd;
  shared_ptr<MKLData<Dtype> > fwd_top_data, fwd_bottom_data;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;
  Dtype *lrn_buffer_;
};



template <typename Dtype>
class MKLPoolingLayer : public Layer<Dtype> {
 public:
  explicit MKLPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      fwd_top_data    (new MKLData<Dtype>()),
      fwd_bottom_data (new MKLData<Dtype>()),
      bwd_top_diff    (new MKLDiff<Dtype>()),
      bwd_bottom_diff (new MKLDiff<Dtype>()),
      poolingFwd(NULL), poolingBwd(NULL) {}
  ~MKLPoolingLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_, num_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<size_t> max_idx_;

 private:
  size_t kernel_size[2],
         kernel_stride[4];
  int src_offset[2];
  shared_ptr<MKLData<Dtype> > fwd_top_data, fwd_bottom_data;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  dnnPrimitive_t poolingFwd, poolingBwd;
};

template <typename Dtype>
class MKLReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit MKLReLULayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param),
      fwd_top_data_    (new MKLData<Dtype>()),
      fwd_bottom_data_ (new MKLData<Dtype>()),
      bwd_top_diff_    (new MKLDiff<Dtype>()),
      bwd_bottom_diff_ (new MKLDiff<Dtype>()),
      reluFwd_(NULL),
      reluBwd_(NULL) {}

  ~MKLReLULayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReLU"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<MKLData<Dtype> > fwd_top_data_;
  shared_ptr<MKLData<Dtype> > fwd_bottom_data_;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff_;
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff_;
  dnnPrimitive_t reluFwd_, reluBwd_;
  vector<size_t> sizes_;
  vector<size_t> strides_;
};

template <typename Dtype>
class MKLConcatLayer : public Layer<Dtype> {
 public:
  explicit MKLConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        concatFwd_(static_cast<dnnPrimitive_t>(NULL)),
        concatBwd_(static_cast<dnnPrimitive_t>(NULL)),
        fwd_top_data_    (new MKLData<Dtype>()),
        bwd_top_diff_    (new MKLDiff<Dtype>()),
        split_channels_  (NULL) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Concat"; }
  ~MKLConcatLayer();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

 private:
  dnnPrimitive_t concatFwd_;
  dnnPrimitive_t concatBwd_;
  shared_ptr<MKLData<Dtype> > fwd_top_data_;
  vector<shared_ptr<MKLData<Dtype> > > fwd_bottom_data_;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff_;
  vector<shared_ptr<MKLDiff<Dtype> > > bwd_bottom_diff_;
  size_t *split_channels_;

  size_t width_;
  size_t height_;
  size_t channels_;
  size_t num_;
  size_t num_concats_;
};

template <typename Dtype>
class MKLBatchNormLayer : public Layer<Dtype> {
 public:
  explicit MKLBatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        fwd_top_data(new MKLData<Dtype>()),
        fwd_bottom_data(new MKLData<Dtype>()),
        bwd_top_diff(new MKLDiff<Dtype>()),
        bwd_bottom_diff(new MKLDiff<Dtype>()),
        batchNormFwd(static_cast<dnnPrimitive_t>(NULL)),
        batchNormBwdData(static_cast<dnnPrimitive_t>(NULL)),
        batchNormBwdScaleShift(static_cast<dnnPrimitive_t>(NULL)),
        workspace_buffer_(static_cast<Dtype*>(NULL)),
        scaleShift_buffer_(static_cast<Dtype*>(NULL)),
        layout_usr_(static_cast<dnnLayout_t>(NULL)) {}

  virtual ~MKLBatchNormLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

//  Dtype moving_average_fraction_;
  Dtype eps_;
  bool use_weight_bias_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_;
  int width_;

 private:
  shared_ptr<MKLData<Dtype> > fwd_top_data;
  shared_ptr<MKLData<Dtype> > fwd_bottom_data;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff;
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff;
  Blob<Dtype> temp_;
  dnnPrimitive_t batchNormFwd, batchNormBwdData, batchNormBwdScaleShift;
  Dtype *workspace_buffer_;
  Dtype *scaleShift_buffer_;
  dnnLayout_t layout_usr_;
};

template <typename Dtype>
class MKLSplitLayer : public Layer<Dtype> {
 public:
  explicit MKLSplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        bwd_bottom_diff (new MKLDiff<Dtype>()),
        sumPrimitive(static_cast<dnnPrimitive_t>(NULL)) {}

  virtual ~MKLSplitLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Split"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff;
  vector<shared_ptr<MKLDiff<Dtype> > > bwd_top_diff;
  vector<Dtype> coeffs_;
  size_t num_tops;
  vector<size_t> sizes_src_;
  vector<size_t> strides_src_;
  dnnPrimitive_t sumPrimitive;
};

template <typename Dtype>
class MKLEltwiseLayer : public Layer<Dtype> {
 public:
  explicit MKLEltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        fwd_top_data       (new MKLData<Dtype>()),
        sumPrimitive(static_cast<dnnPrimitive_t>(NULL)) {}

  virtual ~MKLEltwiseLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  void Init(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Eltwise"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<MKLData<Dtype> > fwd_top_data;
  vector<shared_ptr<MKLData<Dtype> > > fwd_bottom_data;
  vector<shared_ptr<MKLDiff<Dtype> > > bwd_bottom_diff;

  dnnPrimitive_t sumPrimitive;
  dnnPrimitive_t convertPrimitive;

  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
  Blob<int> max_idx_;
  size_t num_bottoms;
  int channels_, num_;
  int height_, width_;

  bool stable_prod_grad_;
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKL2017_LAYERS_HPP_
