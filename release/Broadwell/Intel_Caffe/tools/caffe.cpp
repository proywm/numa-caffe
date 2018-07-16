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

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/make_shared.hpp"
#include "caffe/caffe.hpp"
#include "caffe/internode/mpiutil.hpp"
#include "caffe/multinode/multinode.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");
DEFINE_string(param_server, "",
    "Optional; triggers multinode mode, usage: --param_server=mpi");
DEFINE_string(listen_address, "",
    "Optional; multinode mode, bind address for data server");
DEFINE_int32(comm_threads, 1,
    "Optional; multinode mode,"
    " The number of threads used by communication code.");
DEFINE_bool(forward_only, false,
    "Optional; Execute only forward pass");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (FLAGS_param_server != "") {
    LOG(INFO) << "Configuring multinode setup";

      if (FLAGS_param_server != "mpi") {
        LOG(ERROR) << "currently unsupported";
        return 1;
      }
      caffe::SynchronousNode<float> sync(solver, FLAGS_comm_threads);
      LOG(INFO) << "Starting Multi-node Optimization in mpi environment";
      sync.run();
  } else if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);

int data_server() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }
  LOG(INFO) << "Starting Data Server";
  caffe::DataServer<float> server(
    solver, FLAGS_listen_address, FLAGS_param_server, FLAGS_comm_threads);
  server.run();
  return 0;
}
RegisterBrewFunction(data_server);

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  if (!FLAGS_forward_only) {
    LOG(INFO) << "Performing Backward";
    caffe_net.Backward();
  }

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    if (!FLAGS_forward_only) {
      backward_timer.Start();
      for (int i = layers.size() - 1; i >= 0; --i) {
        timer.Start();
        layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                            bottom_vecs[i]);
        backward_time_per_layer[i] += timer.MicroSeconds();
      }
      backward_time += backward_timer.MicroSeconds();
      LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
        << iter_timer.MilliSeconds() << " ms.";
    } else {
      LOG(INFO) << "Iteration: " << j + 1 << " forward time: "
        << iter_timer.MilliSeconds() << " ms.";
    }
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    if (!FLAGS_forward_only) {
      LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
        "\tbackward: " << backward_time_per_layer[i] / 1000 /
        FLAGS_iterations << " ms.";
    }
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  if (!FLAGS_forward_only) {
    LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
      FLAGS_iterations << " ms.";
  }
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);


// collect & compare: Debugging extansion for CPU-GPU functional comparison
#include <stdio.h>
typedef float real_t;

void getFileName(char *file_name, bool use_gpu, const char *name, int id) {
  const char *prefix = use_gpu ? "GPU" : "CPU";
  snprintf(file_name, FILENAME_MAX, "%s%s%04i.bin", prefix, name, id);
}

bool saveToFile(bool use_gpu, const char *name, int id,
    const real_t *data, unsigned count) {
  char file_name[FILENAME_MAX];
  getFileName(file_name, use_gpu, name, id);

  FILE *file = fopen(file_name, "w+b");
  if (!file) {
    LOG(ERROR) << "Failed to create file '" << file_name << "'.";
    return false;
  }

  size_t bytesToWrite = count * sizeof(data[0]);
  size_t bytesWritten = fwrite(data, 1, bytesToWrite, file);
  fclose(file);

  if (bytesWritten != bytesToWrite) {
    LOG(ERROR) << "Failed to write data to '" << file_name << "' file.";
    return false;
  }

  return true;
}

bool loadFromFile(bool use_gpu, const char *name, int id,
    real_t *data, unsigned count) {
  char file_name[FILENAME_MAX];
  getFileName(file_name, use_gpu, name, id);

  FILE *file = fopen(file_name, "rb");
  if (!file) {
    LOG(ERROR) << "Failed to open file '" << file_name << "' for read.";
    return false;
  }

  size_t bytesToRead = count * sizeof(data[0]);
  size_t bytesRead = fread(data, 1, bytesToRead, file);
  fclose(file);

  if (bytesRead != bytesToRead) {
    LOG(ERROR) << "Failed to read data from '" << file_name << "' file.";
    return false;
  }

  return true;
}

int collect() {
  #ifndef DETERMINISTIC
    LOG(ERROR) << "Recompile caffe with DETERMINISTIC to run collect tool";
    return 1;
  #endif
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition!";

  vector<int> gpus;
  get_gpus(&gpus);
  bool use_gpu = (gpus.size() != 0);
  if (use_gpu) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<real_t> caffe_net(FLAGS_model, caffe::TRAIN);
  const vector<shared_ptr<Layer<real_t> > >& layers = caffe_net.layers();
  const vector<shared_ptr<Blob<real_t> > >& params = caffe_net.params();
  const vector<vector<Blob<real_t>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<real_t>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
    caffe_net.bottom_need_backward();

  FILE *infoFile = fopen(use_gpu ? "GPUInfo.txt" : "CPUInfo.txt", "w+t");
  LOG(INFO) << "*** Collect procedure begins ***";

  for (int i = 0; i < params.size(); i++) {
    caffe::caffe_set(params[i]->count(), 0.f,
      params[i]->mutable_cpu_diff());
  }

  for (int i = 0; i < layers.size(); ++i) {
    LOG(INFO) << "Collecting FW Layer[" << i << "]: " << layers[i]->type();
    fprintf(infoFile, "Fwrd%04i: %s\n", i, layers[i]->type());
    layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    saveToFile(use_gpu, "Fwrd", i,
      top_vecs[i][0]->cpu_data(), top_vecs[i][0]->count());
  }

  for (int i = layers.size() - 1; i >= 0; --i) {
    LOG(INFO) << "Collecting BW Layer[" << i << "]: " << layers[i]->type();
    fprintf(infoFile, "Bwrd%04i: %s\n", i, layers[i]->type());
    layers[i]->Backward(top_vecs[i], bottom_need_backward[i], bottom_vecs[i]);
    if (bottom_need_backward[i][0]) {
      saveToFile(use_gpu, "Bwrd", i,
        bottom_vecs[i][0]->cpu_diff(), bottom_vecs[i][0]->count());
    }
  }

  LOG(INFO) << "Collecting gradients and weights";
  for (int i = 0; i < params.size(); i++) {
    saveToFile(use_gpu, "Grad", i,
      params[i]->cpu_diff(), params[i]->count());
    saveToFile(use_gpu, "Wght", i,
      params[i]->cpu_data(), params[i]->count());
  }

  LOG(INFO) << "*** Collect procedure ends ***";
  fclose(infoFile);
  return 0;
}
RegisterBrewFunction(collect);

int compare() {
  #ifndef DETERMINISTIC
    LOG(ERROR) << "Recompile caffe with DETERMINISTIC to run compare tool";
    return 1;
  #endif
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition!";

  vector<int> gpus;
  get_gpus(&gpus);
  bool use_gpu = (gpus.size() != 0);
  if (use_gpu) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<real_t> caffe_net(FLAGS_model, caffe::TRAIN);
  const vector<shared_ptr<Layer<real_t> > >& layers = caffe_net.layers();
  const vector<shared_ptr<Blob<real_t> > >& params = caffe_net.params();
  const vector<vector<Blob<real_t>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<real_t>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
    caffe_net.bottom_need_backward();

  FILE *infoFile = fopen(use_gpu ? "GPUInfo.txt" : "CPUInfo.txt", "w+t");
  LOG(INFO) << "*** Compare procedure begins ***";

  for (int i = 0; i < params.size(); i++) {
    caffe::caffe_set(params[i]->count(), 0.f,
      params[i]->mutable_cpu_diff());
  }

  for (int i = 0; i < layers.size(); ++i) {
    LOG(INFO) << "Collecting FW Layer[" << i << "]: " << layers[i]->type();
    fprintf(infoFile, "Fwrd%04i: %s\n", i, layers[i]->type());
    layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    saveToFile(use_gpu, "Fwrd", i,
      top_vecs[i][0]->cpu_data(), top_vecs[i][0]->count());
    loadFromFile(!use_gpu, "Fwrd", i,
      top_vecs[i][0]->mutable_cpu_data(), top_vecs[i][0]->count());
  }

  for (int i = layers.size() - 1; i >= 0; --i) {
    LOG(INFO) << "Collecting BW Layer[" << i << "]: " << layers[i]->type();
    fprintf(infoFile, "Bwrd%04i: %s\n", i, layers[i]->type());
    layers[i]->Backward(top_vecs[i], bottom_need_backward[i], bottom_vecs[i]);
    if (bottom_need_backward[i][0]) {
      saveToFile(use_gpu, "Bwrd", i,
        bottom_vecs[i][0]->cpu_diff(), bottom_vecs[i][0]->count());
      loadFromFile(!use_gpu, "Bwrd", i,
        bottom_vecs[i][0]->mutable_cpu_diff(), bottom_vecs[i][0]->count());
    }
  }

  LOG(INFO) << "Collecting gradients and weights";
  for (int i = 0; i < params.size(); i++) {
    saveToFile(use_gpu, "Grad", i,
      params[i]->cpu_diff(), params[i]->count());
    saveToFile(use_gpu, "Wght", i,
      params[i]->cpu_data(), params[i]->count());
  }

  LOG(INFO) << "*** Compare procedure ends ***";
  fclose(infoFile);
  return 0;
}
RegisterBrewFunction(compare);


int main(int argc, char** argv) {
  caffe::internode::mpi_init(argc, argv);
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  data_server     run data server - remote data source\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time\n"
      "  collect         collects layer data on specified device\n"
      "  compare         collects layer data using inputs from other device");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      int ret = GetBrewFunction(caffe::string(argv[1]))();
      caffe::internode::mpi_finalize();
      return ret;
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      caffe::internode::mpi_finalize();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
  caffe::internode::mpi_finalize();
  return 0;
}
