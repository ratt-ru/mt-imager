/* CUDADeviceManager.h:  Definition of the CUDADeviceManager class
 *      
 * Copyright (C) 2013  Daniel Muscat
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * Author's contact details can be found at http://www.danielmuscat.com 
 *
 */
#ifndef __CUDADEVICEMANAGER_H__
#define	__CUDADEVICEMANAGER_H__
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#define ALLOC

namespace GAFW { namespace GPU 
{
    class OperationDescriptor;
    class CUDADeviceManager :public GAFW::LogFacility,public  GAFW::Identity
    {
       class FalseMutex
       {
       public:
           void lock(){}
           void unlock(){};
       };
       
    
    private:
        enum AllocAndPopulateReturn
        {
            AllocSuccess,
            UnableToAlloc, //If a misisng cache and unabel toAlloc... this will be 
            MissingCache,
            LinkedForOverwrite
        };
        //CUDADeviceManager(const CUDADeviceManager& orig):deviceNo(-1) {};
    protected:
        GPUEngine *engine;
        const int deviceNo;
        CudaEventPool eventPool;
        boost::atomic<bool> cache_and_deallocate; 
        SynchronousObject<bool> SubmitReadyForUnknownBuffer;  
        
        SynchronousObject<int> knownOperations;
        SynchronousObject<size_t> gpuMemoryAllocated;
        static boost::mutex s_gpuAllocMutex;  
        static boost::mutex s_memAllocKernelSubmitMutex;//This co-ordinates teh submit thread with teh alloc thread ..such they don't work together
        
        boost::mutex gpuAllocMutex; 
        //public:
#ifndef NOLOCK
        /*static*/ boost::mutex memAllocKernelSubmitMutex;//This co-ordinates teh submit thread with teh alloc thread ..such they don't work together
#else   
                   FalseMutex memAllocKernelSubmitMutex;
#endif                   
    public:
#ifdef ALLOC
        static SynchronousObject<int> inalloc;
#endif
    protected:
        
        GAFW::GPU::GPUDeviceDescriptor *deviceDescriptor;
        boost::thread * myThread;
        boost::thread * submitThread;
        cudaStream_t memoryCopyToGPUStream;
        cudaStream_t memoryCopyToHostStream;
        cudaStream_t &memoryStream;
        cudaStream_t kernelStream;
        
        boost::thread * threadAlloc;
        void threadAllocFunc();
        boost::thread * threadGPUSubmit;
        void threadGPUSubmitFunc();
        boost::thread * threadPostCalcWork;
        void threadPostCalcWorkFunc();
        boost::thread * threadExtraAllocates;
        void threadExtraAllocatesFunc();
        boost::thread *threadWaitCopyToGPU;
        void threadWaitCopyToGPUFunc();
        boost::thread *threadWaitKernelExecutionReady;
        void threadWaitKernelExecutionReadyFunc();
        
        SynchronousQueue<OperationDescriptor *> newReqeustsQueue;
        SynchronousQueue<OperationDescriptor *> copyWaitQueue;
        SynchronousQueue<OperationDescriptor *> gpuSubmitQueue;
        SynchronousQueue<OperationDescriptor *> waitKernelExecutionQueue;
        SynchronousQueue<OperationDescriptor *> readyQueue;
        SynchronousQueue<DataDescriptor *> allocDataQueue;
     
        void allocateAndPopulate(OperationDescriptor* desc,bool &noMoreAlloc,bool &missingCache ,int &totalAlloc);
        enum AllocAndPopulateReturn allocateAndPopulate(OperationDescriptor *desc);
        size_t getAllocMemSize(OperationDescriptor *desc);
        size_t getMoreAllocRequiredSize(OperationDescriptor *desc);
        void reverse_allocateAndPopulate(OperationDescriptor *desc);
        bool checkIfReadyForSubmit(OperationDescriptor* desc);
        bool postOperationManagment(OperationDescriptor *desc);
        bool dataPostOperationProcessing(DataDescriptor *data);
        bool handleCache(DataDescriptor *data , bool &deallocate); //postOperation set to true if called from the post operation thread
        // false if from extra alloc thread 
        //returns true if the memory on GPU can be deallocated
        void freeGPUMemory(DataDescriptor * data );
        void allocForUnkownSizeBuffer(OperationDescriptor * desc);
        bool gpuMemAlloc(void ** pointer,size_t size);
        void gpuMemFree(void ** pointer,size_t size);
        
    public:
        CUDADeviceManager(int deviceNo, 
                 GPUEngine *engine);
        virtual ~CUDADeviceManager();
        bool submitOperation(OperationDescriptor *);
        void submitDataDescriptorForReview(DataDescriptor *);
        
    };

}};

#endif	/* CUDADEVICEMANAGER_H */

