#ifndef VIRTDEV_DEVICE_HPP_
#define VIRTDEV_DEVICE_HPP_
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>
#include <boost/thread.hpp>
//#include "caffe/common.hpp"

//#define S2N1 1
//#define 4S1N 1
//#define 8S1N 1

//#if defined(S8N1)
//#define VIRTDEVSCOUNT 32
//#define COARSE_THREADS 1
//#elif defined(S4N1)
//#define VIRTDEVSCOUNT 16
//#define COARSE_THREADS 2
//#elif defined(S2N1)

#define VIRTDEVSCOUNT 8
#define COARSE_THREADS 6
//#endif

#define LOOP_COUNT 10000

#define CORES_PER_NUMA 6
#define NUM_MKL_THREADS 1
#define NUM_MKL_THREADS_IN_SOLVER 6  //64//COARSE_THREADS

namespace caffe {

typedef void* virtDevStream_t;
class VirtDevMemcpyKind { 
   public:
	static const int virtDevMemcpyHostToHost = 0;
	static const int virtDevMemcpyHostToDevice = 1;
	static const int virtDevMemcpyDeviceToHost = 2;
	static const int virtDevMemcpyDeviceToDevice = 3;
	static const int virtDevMemcpyDefault = 4;
	VirtDevMemcpyKind();
	~VirtDevMemcpyKind();
};
class VirtDevStreamKind {
   public:
	static virtDevStream_t virtDevStreamDefault;
	//flag
	static const int virtDevStreamNonBlocking = 1;
	VirtDevStreamKind();
	~VirtDevStreamKind();
};
class VirtDevStatus{
  public:
	static const int VDEV_STATUS_SUCCESS = 0;
	static const int virtDevSuccess = 0;
	static const int VDEV_STATUS_NOT_INITIALIZED = 1;
	static const int VDEV_STATUS_ALLOC_FAILED = 2;
	static const int VDEV_STATUS_INVALID_VALUE = 3;
	static const int VDEV_STATUS_ARCH_MISMATCH = 4;
	static const int VDEV_STATUS_MAPPING_ERROR = 5;
	static const int VDEV_STATUS_EXECUTION_FAILED = 6;
	static const int VDEV_STATUS_INTERNAL_ERROR = 7;
	static const int VDEV_STATUS_NOT_SUPPORTED = 8;
	VirtDevStatus();
	~VirtDevStatus();
};

//typedef class VirtDevStatus* virtDevStatus_t;
typedef int virtDevStatus_t;

class VirtDeviceProp{
   public:
	VirtDeviceProp();
	~VirtDeviceProp();
   static bool virtDevGetDeviceProperties(class VirtDeviceProp * thisProp, int deviceId);
   int major;
   int minor;
   char name[16];
   int totalGlobalMem;
   int sharedMemPerBlock;
   int regsPerBlock;
   int warpSize;
   int memPitch;
   int maxThreadsPerBlock;
   int maxThreadsDim[3];
   int maxGridSize[3];
   int clockRate;
   int totalConstMem;
   int textureAlignment;
   bool deviceOverlap;
   int multiProcessorCount;
   bool kernelExecTimeoutEnabled;
};

class VDEVHandle{
  public:
    VDEVHandle();
    ~VDEVHandle();
};
typedef class VDEVHandle* VDEVHandle_t;
/*
class VDEVHandle_t{
  public:
    VDEVHandle_t();
    ~VDEVHandle_t();
  };
*/

class VirtDevrand_generator{
   public:
        VirtDevrand_generator();
        ~VirtDevrand_generator();
};

typedef class VirtDevrand_generator* VirtDevrandGenerator_t;

#define MAXDEVS 256
class VirtDev {
 public:
  VirtDev();
  ~VirtDev();
  int virtDevId;
  
};
//#include <threads.h>
#include <pthread.h>
  //enum virtDevMemcpyKind { virtDevMemcpyHostToHost, virtDevMemcpyHostToDevice, virtDevMemcpyDeviceToHost, virtDevMemcpyDeviceToDevice , virtDevMemcpyDefault};
  //enum virtDevStreamKind {virtDevStreamDefault, virtDevStreamNonBlocking};
class VirtDevManager {
 public:
  VirtDevManager();
  ~VirtDevManager();
  static bool virtDevCreate(VDEVHandle_t *handle);
  static bool virtDevGetDevice(int *deviceID);
  static bool virtDevSetDevice(int deviceID);
  template<typename Dtype>
  static bool virtDevMalloc(Dtype **ptr, int size);
  template<typename Dtype>
  static bool virtDevFree(Dtype *ptr);
  static bool virtDevDestroy(VDEVHandle_t handle);
  static int virtDevMemcpyAsync 	( 	void *  	dst,
		const void *  	src,
		size_t  	count,
		int kind,
		virtDevStream_t  	stream = NULL
	);
  static bool virtDevMemcpy 	( 	void *  	dst,
		const void *  	src,
		size_t  	count,
		int kind	 
	);
  static bool virtDevStreamCreateWithFlags(virtDevStream_t stream,int flag);
  static int virtDevStreamSynchronize(virtDevStream_t stream);
  static int virtDevStreamDestroy(virtDevStream_t stream);
  static bool virtDevrandDestroyGenerator(VirtDevrandGenerator_t generator);
  static bool virtDeviceCanAccessPeer( 	int *  	canAccessPeer,
		int  	device,
		int  	peerDevice	 
	);
  static bool virtDeviceEnablePeerAccess( int peerDevice, unsigned int flags);
  static bool virtDeviceDisablePeerAccess(int peerDevice);
  //static int rootVirtDevCPUID;
  static std::vector<VirtDev*> listOfVirtDev;
  static __thread int virtDevId;
  static std::vector<int> CoreOccupancyList;
  static int virtDevGetNextDataReaderCore();
  static int virtDevGetNextSolverCore();
  static int solverCount;
  static boost::mutex virtDevAllocationLoc;
  static bool virtDevAssignOMPThreads();
    static bool placeSolverOnSocket(int deviceID);
  static bool placeMKLInThread(int parentCPU);
  static int placeCOARSEOMPWithinSocket(int deviceID);
  static bool ENABLE_BLAS_PARALLEL_IN_SOLVER(int deviceID, bool withSMT=false);
  static bool DISABLE_BLAS_PARALLEL_IN_SOLVER(int deviceID);
  static bool ENABLE_BLAS_PARALLEL_IN_CORE(int parentCPU, bool withSMT=false);
  static bool DISABLE_BLAS_PARALLEL_IN_CORE(int parentCPU);
};
/*
template <typename Dtype>
bool VirtDevManager::virtDevMalloc(Dtype **ptr, int size){
 //       LOG(INFO) << "vdev malloc ";
        //ask thread to malloc and first touch there;
        *ptr = (Dtype **)malloc(size);
        return true;
}

template <typename Dtype>
bool VirtDevManager::virtDevFree(Dtype *ptr){
        return true;
}
*/
}
#endif
