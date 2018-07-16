#include "caffe/util/virtDev_device.hpp"
//#include <cstring>
#include "caffe/common.hpp"
#include <omp.h>
#include "mkl.h"
#include <math.h> 

namespace caffe { 

VirtDev::VirtDev()
{};
VirtDev::~VirtDev(){};


std::vector<VirtDev*> VirtDevManager::listOfVirtDev(10, NULL);
std::vector<int> VirtDevManager::CoreOccupancyList(31, 0);
//static int VirtDevManager::virtDevId = 0;
//__thread virtPointerAttributes *VirtDevManager::virtAttr= new virtPointerAttributes();
VirtDevManager::VirtDevManager(){
}

VirtDevManager::~VirtDevManager(){}
__thread int VirtDevManager::virtDevId = 0;
int VirtDevManager::solverCount = 0;
boost::mutex VirtDevManager::virtDevAllocationLoc;
bool VirtDevManager::virtDevCreate(VDEVHandle_t *handle){
	//initialize handle
        //LOG(INFO) << "came to create vdev";
        //rootVirtDevCPUID = 0;
	return true;//VirtDevStatus::VDEV_STATUS_SUCCESS;
}

bool VirtDevManager::virtDevGetDevice(int *deviceID){
	//device 	- Returns the device on which the active host thread executes the device code.
        //LOG(INFO) << "get Dev";
        *deviceID = virtDevId;
        //*deviceID = virtDevId;
	return true;
}
bool VirtDevManager::virtDevAssignOMPThreads()
{
#if 1
        if(sched_getcpu() == (sched_getcpu()%2)+omp_get_thread_num()*2)
	  return true;
#if 0
	cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
        CPU_SET((sched_getcpu()%2)+omp_get_thread_num()*4, &my_set);
        int rc = pthread_setaffinity_np(pthread_self(),
                                    sizeof(cpu_set_t), &my_set);
#endif
	int rc = 0;
	if(0==rc)
	{
		LOG(INFO)<< " Forward_cpu " << " cpu id " << sched_getcpu() << " omp_get_thread_num " << omp_get_thread_num() << " set to " << (sched_getcpu()%2)+omp_get_thread_num()*2 << " pthread id " << pthread_self();
		return true;
	}
	else
	{
		LOG(INFO) << "FAILED TO SET OMP THREAD TO CPU";
		return false;
	}
#endif
}

int VirtDevManager::placeCOARSEOMPWithinSocket(int deviceID)//solverID
{
#if 1
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
        int a = 0;
                if(omp_get_thread_num()>= 0 && omp_get_thread_num() < COARSE_THREADS)
                {
                        CPU_SET((deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS)), &my_set);     /* set the bit that represents core 7. */
                        a = (deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS));
                }
                else if(omp_get_thread_num()>= COARSE_THREADS && omp_get_thread_num() < 2*COARSE_THREADS)
                {
                        CPU_SET(56+((deviceID%VIRTDEVSCOUNT))+(4*(omp_get_thread_num()%COARSE_THREADS)), &my_set);
                        a = 56+((deviceID%VIRTDEVSCOUNT))+(4*(omp_get_thread_num()%COARSE_THREADS));
                }
		else
			LOG(INFO)<< " placeCOARSEOMPWithinSocket cpu: " << a << " omp_get_thread_num() "<< omp_get_thread_num();
        if(sched_setaffinity(0, sizeof(cpu_set_t), &my_set))
                printf("Error Setting up OMP Threads on CPU %d and omp_get_thread_num(): %d\n",a,omp_get_thread_num());
        return a;
#endif
	return 0;
}

bool VirtDevManager::placeMKLInThread(int parentCPU)
{
#if 1
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
        CPU_SET(parentCPU, &my_set);     /* set the bit that represents core 7. */
        if(NUM_MKL_THREADS==4)
        {
                CPU_SET(parentCPU+64, &my_set);
		CPU_SET(parentCPU+128, &my_set);
		CPU_SET(parentCPU+192, &my_set);
		
        }
 //       if(sched_setaffinity(0, sizeof(cpu_set_t), &my_set))
   //             printf("Error Setting up OMP Threads on CPU %d",parentCPU);
   	 return sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#endif
	return 0;
}
bool VirtDevManager::ENABLE_BLAS_PARALLEL_IN_CORE(int parentCPU, bool withSMT)
{
#if 1
	bool ret;
#pragma omp parallel num_threads(1)//(NUM_MKL_THREADS)
{
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
        if(omp_get_thread_num()<1)
             CPU_SET(parentCPU+omp_get_thread_num(), &my_set);     /* set the bit that represents core 7. */
        else if(omp_get_thread_num()==1)
             CPU_SET(parentCPU+omp_get_thread_num()+64, &my_set);
	else if(omp_get_thread_num()==2)
             CPU_SET(parentCPU+omp_get_thread_num()+64*2, &my_set);
	else if(omp_get_thread_num()==3)
             CPU_SET(parentCPU+omp_get_thread_num()+64*3, &my_set);
        ret =  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
	//LOG(INFO) << "Parent CPU Core "<< parentCPU <<" ENABLE_BLAS_IN_CORE "<< parentCPU+omp_get_thread_num()+(32*omp_get_thread_num()) << " omp_get_num_threads "<<omp_get_num_threads();
}
        return ret;
#endif 
	//return 0;
}
bool VirtDevManager::DISABLE_BLAS_PARALLEL_IN_CORE(int parentCPU)
{
#if 1
        bool ret;
//        mkl_set_num_threads_local(1);
        #pragma omp parallel num_threads(1)
        {
                cpu_set_t my_set;
                CPU_ZERO(&my_set);
                CPU_SET(parentCPU+omp_get_thread_num(), &my_set);
                ret = sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
        }
        return ret;
#endif
	return 0;
}

bool VirtDevManager::ENABLE_BLAS_PARALLEL_IN_SOLVER(int deviceID, bool withSMT)
{
#if 1
	bool ret;
	//if(withSMT)
	//	mkl_set_num_threads_local(2*NUM_MKL_THREADS_IN_SOLVER);
	//else
	//	mkl_set_num_threads_local(NUM_MKL_THREADS_IN_SOLVER);
	#pragma omp parallel num_threads(NUM_MKL_THREADS_IN_SOLVER)
	{
		//LOG(INFO)<< " ENABLE BLAS PARALLEL IN SOLVER  deviceID " << deviceID << " omp_get_num_threads "<<omp_get_num_threads();
		cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        	CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
		//if(omp_get_thread_num()<=7)
		int a = 0;
		if(omp_get_thread_num()>= 0 && omp_get_thread_num() < COARSE_THREADS)
		{
			CPU_SET((deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS)), &my_set);     /* set the bit that represents core 7. */
                        a = (deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS));
		}
		else if(omp_get_thread_num()>= COARSE_THREADS && omp_get_thread_num() < 2*COARSE_THREADS)
		{
			CPU_SET(56+(deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS)), &my_set);     /* set the bit that represents core 7. */
                        a = 56+(deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS));
		}
//		if(omp_get_thread_num()>=COARSE_THREADS)
//			CPU_SET((COARSE_THREADS*(deviceID%VIRTDEVSCOUNT))+omp_get_thread_num()+32, &my_set);
        	ret =  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
    	//	if(a>15 && a<64)
		//LOG(INFO)<< " ENABLE BLAS PARALLEL IN SOLVER deviceID " << deviceID << " cpu " << a <<" omp_get_thread_num "<< omp_get_thread_num() << " omp_get_num_threads "<<omp_get_num_threads();
	}
	return ret;
#endif
	//return 0;
}

bool VirtDevManager::DISABLE_BLAS_PARALLEL_IN_SOLVER(int deviceID)
{
#if 1
	bool ret;
//        mkl_set_num_threads_local(1);
        #pragma omp parallel num_threads(1)
        {
		cpu_set_t my_set;
		CPU_ZERO(&my_set);
		CPU_SET((deviceID%VIRTDEVSCOUNT)+(4*(omp_get_thread_num()%COARSE_THREADS)), &my_set);
		//CPU_SET((COARSE_THREADS*(deviceID%VIRTDEVSCOUNT))+omp_get_thread_num(), &my_set);
		ret = sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
		//LOG(INFO)<< "DISABLING BLAS PARALLEL IN SOLVER deviceID ";
        }
        return ret;
#endif
	return 0;
}

bool VirtDevManager::placeSolverOnSocket(int deviceID)//solverID
{
#if 1
        int orgCPU = (deviceID%VIRTDEVSCOUNT);//COARSE_THREADS*(deviceID%VIRTDEVSCOUNT);
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */

	for(int c=0 ;c< COARSE_THREADS;c++)
        	CPU_SET(orgCPU+c*4, &my_set);     /* set the bit that represents core 7. */
        //if(sched_setaffinity(0, sizeof(cpu_set_t), &my_set)) //setting it 0 is must get_pid is not a good solution
          //      printf("Error Setting up Solvers on CPU %d",orgCPU);
        //printf("pthread -- (CPU: %d)>> Setting up solvers at %d %d %d %d %d \n",sched_getcpu(),orgCPU,orgCPU+1,orgCPU+2,orgCPU+3,orgCPU+4);
        return sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#endif
	return 0;
}
bool VirtDevManager::virtDevSetDevice(int deviceID){
#if 1
        //Setting current device ID, For VirtDev it is nothing but setting the CPU and storing the device ID
        // for launching new threads / malloc will be done on this CPU Core
        virtDevId = deviceID;
        VirtDev *vdev = new VirtDev();
        vdev->virtDevId = deviceID;
        listOfVirtDev.push_back(vdev);
        //LOG(INFO) << "Set vdev " << deviceID << " thread ID "<< pthread_self();
        //Changing CPU to Next NUMA NODE
        //int orgCPU = deviceID;
	int orgCPU = (deviceID%VIRTDEVSCOUNT);
        cpu_set_t my_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
	int c = 0;
       // for(int c=0 ;c< COARSE_THREADS;c++)
        	CPU_SET(orgCPU+c*4, &my_set);     /* set the bit that represents core 7. */
        sched_setaffinity(0, sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
        //omp_set_num_threads(8);
        //setting attribute structure , remove this later

	return true;
#endif
	return true;
}
template bool VirtDevManager::virtDevMalloc(void **ptr, int size);
template bool VirtDevManager::virtDevMalloc(int **ptr, int size);
template bool VirtDevManager::virtDevMalloc(double **ptr, int size);
template bool VirtDevManager::virtDevMalloc(float **ptr, int size);
//moved to header as template function
template <typename Dtype>
bool VirtDevManager::virtDevMalloc(Dtype **ptr, int size){
        //LOG(INFO) << "vdev malloc ";
        //ask thread to malloc and first touch there;
	*ptr = (Dtype *)malloc(size);
 //       LOG(INFO) << "done vdev malloc, returning " << virtDevId << " cpu Id " << sched_getcpu();
	return true;
}
template bool VirtDevManager::virtDevFree(void *ptr);
template bool VirtDevManager::virtDevFree(int *ptr);
template bool VirtDevManager::virtDevFree(double *ptr);
template bool VirtDevManager::virtDevFree(float *ptr);
template <typename Dtype>
bool VirtDevManager::virtDevFree(Dtype *ptr){
        //LOG(INFO) << "came to free ptr in virtDevFree";
        free(ptr);
 //       ptr = NULL;
	return true;
}
int VirtDevManager::virtDevGetNextSolverCore()
{
	int iter = 0;
        if(0 == solverCount%2)
        {
           while(iter <= 14  && (1 == CoreOccupancyList[iter]))
                iter = iter +2;
        }
        else
        {
           iter = 1;
           while(iter <= 15  && (1 == CoreOccupancyList[iter]))
                iter = iter + 2;
        }
        CoreOccupancyList[iter] = 1;
        solverCount++;
        LOG(INFO) << "found a CPU core " << iter <<" for Solver on device " << virtDevId << " thread ID "<< pthread_self();

        if(iter >15 )
                return 0;
        else
                return iter;
}
int VirtDevManager::virtDevGetNextDataReaderCore()
{
	//even odd NUMA mapping
#if 1
        int iter = 14;
	if(0 == virtDevId%2)
        {
	   while(iter >= 0  && (1 == CoreOccupancyList[iter]))
		iter = iter -2;
        }
	else
	{
	   iter = 15;
	   while(iter >= 1  && (1 == CoreOccupancyList[iter]))
                iter = iter - 2;
	}
	CoreOccupancyList[iter] = 1;
        LOG(INFO) << "found a CPU core " << iter <<" for Data Reader on device " << virtDevId <<  " thread ID "<< pthread_self();
        if(iter <0 )
	        return 14;
	else 
		return iter;
#endif
	return 0;
}
bool VirtDevManager::virtDevDestroy(VDEVHandle_t handle){
	return true;
}

int VirtDevManager::virtDevMemcpyAsync(void * dst,
                const void * src,
                size_t count,
                int kind,
                virtDevStream_t stream){
	LOG(INFO) << "vdev async mem cpy";
	return 0;
}

bool VirtDevManager::virtDevMemcpy( void *dst,
               const void *src,
                size_t count,
                int kind){
        //LOG(INFO) << "vdev mem cpy";
	switch(kind)
	{
		case VirtDevMemcpyKind::virtDevMemcpyDefault:
			memcpy(dst, src, count);
			break;
		case VirtDevMemcpyKind::virtDevMemcpyHostToHost:
			memcpy(dst, src, count);
                        break;
		case VirtDevMemcpyKind::virtDevMemcpyHostToDevice:
			memcpy(dst, src, count);
		//	dst = src;
                        break;
                case VirtDevMemcpyKind::virtDevMemcpyDeviceToHost:
		//	dst = src;
			memcpy(dst, src, count);
                        break;
		case VirtDevMemcpyKind::virtDevMemcpyDeviceToDevice:
			//dst = src;
			memcpy(dst, src, count);
                        break;
	}
       // LOG(INFO) << "done vdev mem cpy";
        return true;//VirtDevStatus::VDEV_STATUS_SUCCESS;
	//return VirtDevStatus::virtDevSuccess;
}

bool VirtDevManager::virtDevStreamCreateWithFlags(virtDevStream_t stream,int flag){
	return 0;
}

int VirtDevManager::virtDevStreamSynchronize(virtDevStream_t stream){
	return 0;
}

int VirtDevManager::virtDevStreamDestroy(virtDevStream_t stream){
	return 0;
}

bool VirtDevManager::virtDevrandDestroyGenerator(VirtDevrandGenerator_t generator){
        LOG(ERROR) << "CAME TO DESTRY GENERATOR" <<  " thread ID "<< pthread_self();
	return true;
}

bool VirtDevManager::virtDeviceCanAccessPeer(   int *   canAccessPeer,
                int     device,
                int     peerDevice
        ){
	return true;
}

bool VirtDevManager::virtDeviceEnablePeerAccess( int peerDevice, unsigned int flags){
	return true;
}

bool VirtDevManager::virtDeviceDisablePeerAccess(int peerDevice){
	return true;
}

VirtDevrand_generator::VirtDevrand_generator(){}
VirtDevrand_generator::~VirtDevrand_generator(){}

VirtDevStatus::VirtDevStatus(){}
VirtDevStatus::~VirtDevStatus(){}

VirtDevStreamKind::VirtDevStreamKind(){}
VirtDevStreamKind::~VirtDevStreamKind(){}

VirtDevMemcpyKind::VirtDevMemcpyKind(){}
VirtDevMemcpyKind::~VirtDevMemcpyKind(){}

virtDevStream_t VirtDevStreamKind::virtDevStreamDefault = NULL;

VirtDeviceProp::VirtDeviceProp(){}
VirtDeviceProp::~VirtDeviceProp(){}
bool VirtDeviceProp::virtDevGetDeviceProperties(class VirtDeviceProp * thisProp, int deviceId){
	return true;
}

}
