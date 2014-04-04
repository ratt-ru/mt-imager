/* CUDADeviceManager.cpp:  Implementation of CUDADeviceManager class 
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

#include "GPUafw.h"
#include "cuda_runtime.h"
#include <list>
#include <boost/thread/pthread/mutex.hpp>
#include <boost/memory_order.hpp>
using namespace GAFW::GPU;
using namespace boost;
using namespace GAFW;
using namespace std;
// boost::mutex CUDADeviceManager::memAllocKernelSubmitMutex;
#ifdef ALLOC
SynchronousObject<int> CUDADeviceManager::inalloc(0);
#endif
CUDADeviceManager::CUDADeviceManager(int deviceNo, 
        /*SynchronousQueue<OperationDescriptor *> &finaliseEngineQueue, 
        SynchronousQueue<void *> &broadcastQueue*/ GPUEngine *engine)
       :Identity("DeviceManager","DeviceManager"),
        deviceNo(deviceNo),cache_and_deallocate(false),
        engine(engine),memoryStream(this->memoryCopyToGPUStream)//,memAllocKernelSubmitMutex(CUDADeviceManager::s_memAllocKernelSubmitMutex)
        
{
    LogFacility::init();
//    FactoryAccess::init();
    
    this->logDebug(other,"Initialising buffers");
   
    
    this->logDebug(other,"Initialising device streams and descriptor");
    
    checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
    checkCudaError(cudaDeviceReset(),"Error while resetting device");
    checkCudaError(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),"Unable to set up flags for device");
    checkCudaError(cudaDeviceSetLimit(cudaLimitMallocHeapSize,0),"Unable to set malloc limit");
    this->deviceDescriptor=new GPUDeviceDescriptor(deviceNo);
    //Now creating the streams
    checkCudaError(cudaStreamCreateWithFlags(&(this->memoryCopyToGPUStream),cudaStreamNonBlocking),"Error while creating memory stream");
    checkCudaError(cudaStreamCreateWithFlags(&(this->memoryCopyToHostStream),cudaStreamNonBlocking),"Error while creating memory stream");
   
    checkCudaError(cudaStreamCreateWithFlags(&(this->kernelStream),cudaStreamNonBlocking),"Error while creating kernel stream");
    
     this->logDebug(other,"Initialising Threads");
     this->SubmitReadyForUnknownBuffer=true;  
     this->knownOperations=0;
     this->gpuMemoryAllocated=0;
     
     
     atomic_thread_fence(memory_order_release);
     this->cache_and_deallocate=false;
     this->threadAlloc=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadAllocFunc));
     this->threadWaitCopyToGPU=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadWaitCopyToGPUFunc));
     this->threadGPUSubmit=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadGPUSubmitFunc));
     this->threadPostCalcWork=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadPostCalcWorkFunc));
     this->threadExtraAllocates=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadExtraAllocatesFunc));
     this->threadWaitKernelExecutionReady=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::threadWaitKernelExecutionReadyFunc));
     
    
    // this->submitThread=new thread(ThreadEntry<CUDADeviceManager>(this,&CUDADeviceManager::submitThreadExec));
    //this->myThread=new thread(ThreadEntry<CUDADeviceManager>(this, &CUDADeviceManager::mainThreadExec));
    
   
}
 void CUDADeviceManager::threadAllocFunc()
 {
     //This thread allocates memory and initilises all required copies to GPU memory
     this->logDebug(other,"Thread Alloc initialised");
     
     checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
     //void **t;
     //cudaMalloc(t,2e9);
     //cout <<"ALLOC";
     OperationDescriptor * currentDesc=NULL;
     for (;;)
     {
         this->newReqeustsQueue.wait();

         memAllocKernelSubmitMutex.lock();
#ifdef ALLOC
         this->inalloc+=1;
#endif
         for (;this->newReqeustsQueue.pop(currentDesc);) //Process all 
         {
         if (currentDesc->specialActions==OperationDescriptor::ThreadShutdown)
         {
             //this->logDebug(other,"Thread Alloc shutting down");
             memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
             this->inalloc-=1;
#endif
             return;
         }
            
         this->logInfo(other,currentDesc->arrayOperator,"threadAllocFunc(). Operator poped in");
         //new Entry
            //We have to check here if we have to stop allocating and wait for when buffer is unknown
            //Immediately increase the counter
            //Check buffer...first thing because if it is unknown the we have to do a different algo implemented .. allocForUnknownSizeBuffer
         if (currentDesc->buffer.mode==GAFW::GPU::Buffer::Buffer_UnkownSize)
         {
             this->allocForUnkownSizeBuffer(currentDesc);
             continue;
         }
             
         
            for (unsigned int i=0;i<currentDesc->noOfInputs;i++)
            {
                currentDesc->inputs[i]->DataMutex.lock();
                currentDesc->inputs[i]->GPUKnownOperations[this->deviceNo]++;
                currentDesc->inputs[i]->DataMutex.unlock();

            }
            for (unsigned int i=0;i<currentDesc->noOfOutputs;i++)
            {
                currentDesc->outputs[i]->DataMutex.lock();
                currentDesc->outputs[i]->GPUKnownOperations[this->deviceNo]++;
                if ((currentDesc->outputs[i]->copyToReady==false)&&(currentDesc->outputs[i]->copyTo==NULL))
                {
                    currentDesc->outputs[i]->copyTo=currentDesc->outputs[i]->resultDataStore->allocMemeory();
                }
                    
                currentDesc->outputs[i]->DataMutex.unlock();
                
            }
         this->knownOperations+=1;
         
         
         
         
         
      //  bool noMoreAlloc=true;
        for(;;)
        {
            
                bool noMoreAlloc=false; 
                int currentKnownOperations=this->knownOperations;
                 
                atomic_thread_fence(boost::memory_order_release);
                int descAllocSize;
                bool missingCache;
                this->allocateAndPopulate(currentDesc,noMoreAlloc,missingCache,descAllocSize);
                if (!noMoreAlloc) break;
                if (missingCache)
                {
                    memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
                    this->inalloc-=1;
#endif
                    memAllocKernelSubmitMutex.lock(); //give space to the submit kernel thread
#ifdef ALLOC
                    this->inalloc+=1;
#endif
                    continue; //we will spin in this case
                }
                    //It means that we havve to wait
                if (currentKnownOperations==1)
                {
                    noMoreAlloc=false;
                    this->allocateAndPopulate(currentDesc,noMoreAlloc,missingCache,descAllocSize);
                    
                    if (!noMoreAlloc) break;
                    if (missingCache) continue; //we will spin in this case 
                    //Ther si not point to give kernel submit space as there are no operators around
                    //we have to request as to cache and deallocate everything
                    //Not best solution but will see for the future
                    //Ok we must ask for caching
                    this->cache_and_deallocate=true; ///This informs all threads as to cache and deallocate
                    //but inreality it is only one thread that can help here.. and we need to send it a signal via its queue
                    DataDescriptor *signal=new DataDescriptor(1);
                    signal->specialActions=DataDescriptor::Cache_and_deallocate;
                    this->allocDataQueue.push(signal);
                    //signal will be deallocated by the other thread
                    //WE wait for everything to deallocate
                    int currentGPUAlloc=(int)this->gpuMemoryAllocated;
                    while(currentGPUAlloc!=descAllocSize)
                    {
                        this->gpuMemoryAllocated.wait_until_change(currentGPUAlloc);
                        
                        currentGPUAlloc=(int)this->gpuMemoryAllocated;
                    }
                    //Let's try again
                    noMoreAlloc=false;
                    this->allocateAndPopulate(currentDesc,noMoreAlloc,missingCache,descAllocSize);
                    if (!noMoreAlloc)
                    {
                        //Ok done... just reset mode
                        this->cache_and_deallocate=false;
                        break;
                    }
                    if (missingCache)
                    {
                        this->cache_and_deallocate=false;
                        continue; //We spin here
                    }
                    //This is the worst case scenario as either there is not enough memory or there is too much fragmentation
                    //We hope for the latter one as the first one cannot be solved
                    this->reverse_allocateAndPopulate(currentDesc); //It is expected that this function completely dealloctes all memory
                    //Let's try
                    noMoreAlloc=false;
                    this->allocateAndPopulate(currentDesc,noMoreAlloc,missingCache,descAllocSize);
                    if (!noMoreAlloc)
                    {
                        //Ok allocation successfull
                        this->cache_and_deallocate=false;
                        break;
                    }
                    if (missingCache)
                    {
                        this->cache_and_deallocate=false;
                        continue; //We spin here
                    }
                    
                    //Nothing to do... we have to throw an exception
                    throw GeneralException2("Unable to allocate all memory for operation. Execution can't continue",currentDesc->arrayOperator);
                }
                else
                {
                    
                    memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
                    this->inalloc-=1;
#endif        
                    this->knownOperations.wait_until_change(currentKnownOperations);
                    memAllocKernelSubmitMutex.lock();
#ifdef ALLOC
                    this->inalloc+=1;
#endif         
                }
                
        }
        this->logInfo(other,currentDesc->arrayOperator,"threadAllocFunc(). Pushing out");
        this->copyWaitQueue.push(currentDesc);
         }
         //Ok we can give control to the submit kernel thread
         memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
         this->inalloc-=1;
#endif
                    
     }
 }
 void CUDADeviceManager::allocForUnkownSizeBuffer(OperationDescriptor * desc)
 {
     //We immediately need to cache and deallocate everything
     this->cache_and_deallocate=true; ///This informs all threads as to cache and deallocate
     DataDescriptor *signal=new DataDescriptor(1);
     signal->specialActions=DataDescriptor::Cache_and_deallocate;
     this->allocDataQueue.push(signal);
     //First wait until all known operations is 0.. We must give space to the kernel submit duringthsi period
     memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
     this->inalloc-=1;
#endif                    
     int currentKnownOperations=this->knownOperations;
     while (currentKnownOperations!=0)
     {
         this->knownOperations.wait_until_change(currentKnownOperations);
         currentKnownOperations=(int)this->knownOperations;
     }
     //Require the lock again and we can move on
     memAllocKernelSubmitMutex.lock();
#ifdef ALLOC
     this->inalloc+=1;
#endif                    
     //An we wait agin as to have 0 allocated memory
     int currentGPUAlloc=(int)this->gpuMemoryAllocated;
     while(currentGPUAlloc!=0)
     {
        this->gpuMemoryAllocated.wait_until_change(currentGPUAlloc);
        currentGPUAlloc=(int)this->gpuMemoryAllocated;
     }
     
     for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        desc->inputs[i]->DataMutex.lock();
        desc->inputs[i]->GPUKnownOperations[this->deviceNo]++;
        desc->inputs[i]->DataMutex.unlock();

    }
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        desc->outputs[i]->DataMutex.lock();
        desc->outputs[i]->GPUKnownOperations[this->deviceNo]++;
        if ((desc->outputs[i]->copyToReady==false)&&(desc->outputs[i]->copyTo==NULL))
        {
            desc->outputs[i]->copyTo=desc->outputs[i]->resultDataStore->allocMemeory();
        }
        desc->outputs[i]->DataMutex.unlock();

    }
//Ok now we allocate what we require
     bool noMoreAlloc=false;
     bool missingCache=true;
     int descAllocSize;
     while (missingCache)
     {
        this->allocateAndPopulate(desc,noMoreAlloc,missingCache,descAllocSize);
     }
     if (noMoreAlloc)
     {
         //This measn that too much memory has been requested
         throw GeneralException2("Unable to allocate all memory requested",desc->arrayOperator);
     }
     this->cache_and_deallocate=false;
     //Great so we now just prepare as to lock ourselves.. submit and wait until this operation is submitted
     this->SubmitReadyForUnknownBuffer=false;
     atomic_thread_fence(boost::memory_order_release);
     this->knownOperations+=1;
     this->copyWaitQueue.push(desc);
     
     memAllocKernelSubmitMutex.unlock();
#ifdef ALLOC
     this->inalloc-=1;
#endif
     this->SubmitReadyForUnknownBuffer.wait_until_change(false);
     memAllocKernelSubmitMutex.lock();
#ifdef ALLOC
     this->inalloc+=1;
#endif               
 }
 
                    
void CUDADeviceManager::threadWaitCopyToGPUFunc()
{
    this->logDebug(other,"Thread WaitCopy initialised");
    checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
    OperationDescriptor * currentDesc=NULL;
   
    for (;;)
    {
         this->copyWaitQueue.pop_wait(currentDesc);
         if (currentDesc->specialActions==OperationDescriptor::ThreadShutdown)
             break; //Loop 
         
         this->logInfo(other,currentDesc->arrayOperator,"threadWaitCopyToGPUFunc(). Operator poped in");
    
         //We examine inputs one by one.. if we find a copy that has not yet finished we wait
        for (unsigned int i=0;i<currentDesc->noOfInputs;i++)
        {
            DataDescriptor *input=currentDesc->inputs[i];
            //input->DataMutex.lock(); I don't believe that there is  need for a lock here as far as I do not change
            // This is the only thread that will change the status of the copy so I am safe
            if (input->MemCpyCacheToGPUInProgress[this->deviceNo])
            {
               //Ok check if it is still being done
                checkCudaError(cudaEventSynchronize(*(input->cacheToGPUEndEvent[this->deviceNo])),"Error while synchronising with a cuda event");
                //Lock and change
                this->eventPool.checkInEvent(input->cacheToGPUEndEvent[this->deviceNo]);
                input->cacheToGPUEndEvent[this->deviceNo]=NULL;
                this->eventPool.checkInEvent(input->cacheToGPUStartEvent[this->deviceNo]);
                input->cacheToGPUStartEvent[this->deviceNo]=NULL;
                
                input->DataMutex.lock();
                input->MemCpyCacheToGPUInProgress[this->deviceNo]=false;
                input->DataMutex.unlock();
                
            }   
           
        }
        //outputs.. we mainly have to take care of overwrites
        for (unsigned int i=0;i<currentDesc->noOfOutputs;i++)
        {
                DataDescriptor *output=currentDesc->outputs[i];
                if (output->overwrite==NULL) continue; // no point even in creating the mutex
                //we just wait for the link to be down
                //If there is not link tne the belwo function returns immedietly
                output->overwrite->wait_for_unsetLinkForOverwrite(this->deviceNo);
                
                //But it might have well been that instead there is a copy form the cache of the overwrite to the o
                if (output->overwrite->MemCpyCacheToForeign[this->deviceNo])
                {
                   // check if it is still being done
                    checkCudaError(cudaEventSynchronize(*(output->cacheToGPUEndEvent[this->deviceNo])),"Unexpected error while synchronising with end event");
                    //Good copy is ready
                    this->eventPool.checkInEvent(output->cacheToGPUStartEvent[this->deviceNo]);
                    output->cacheToGPUStartEvent[this->deviceNo]=NULL;
                    this->eventPool.checkInEvent(output->cacheToGPUEndEvent[this->deviceNo]);
                    output->cacheToGPUEndEvent[this->deviceNo]=NULL;
                    
                    output->overwrite->DataMutex.lock();
                    output->overwrite->MemCpyCacheToForeign[this->deviceNo]=false;
                    output->overwrite->DataMutex.unlock();
                }   
        
    }
    //No checks for buffers
        this->logInfo(other,currentDesc->arrayOperator,"threadAllocFunc(). Pushing out");
        this->gpuSubmitQueue.push(currentDesc);
     }
 }
    

 void CUDADeviceManager::threadGPUSubmitFunc()
 {
     this->logDebug(other,"Thread GPUSubmit initialised");
     checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
     for (;;)
     {
         this->gpuSubmitQueue.wait();
#ifdef ALLOC
         for (int ok=this->inalloc;ok!=0;ok=this->inalloc)
         {
             this->inalloc.wait_until_change(ok);
         }
#endif
       memAllocKernelSubmitMutex.lock();
        OperationDescriptor * desc;
        while(this->gpuSubmitQueue.pop(desc))
        {
            if (desc->specialActions==OperationDescriptor::ThreadShutdown)
            {
                this->logDebug(other,"Submit Thread shutting down");
                memAllocKernelSubmitMutex.unlock();
                return;
            }
                 

            desc->eventStart=this->eventPool.requestEvent();
            desc->eventEnd=this->eventPool.requestEvent();

            GPUSubmissionData &data=desc->submissionData;
            data.stream=this->kernelStream;
            data.postExecutePointer=&desc->data;
            data.endEventRecorded=false;
            data.deviceDescriptor=this->deviceDescriptor;
            //Processing inputs
            for (int i=0;i<data.noOfInputs;i++)
            {
                    data.inputs[i].pointer=desc->inputs[i]->GPUPointer[this->deviceNo];
            }
            //Processing outputs
            for (int i=0;i<data.noOfOutputs;i++)
            {
                    data.outputs[i].pointer=desc->outputs[i]->GPUPointer[this->deviceNo];
            }

            data.startEvent=desc->eventStart;
            data.endEvent=desc->eventEnd;
            //We have to handle buffer
            switch (desc->buffer.mode)
            {
                case GAFW::GPU::Buffer::Buffer_DeallocBeforeSubmit:
                     //Ok we have to freeze the allocation thread
                    this->gpuAllocMutex.lock();
                    gpuMemFree(&desc->buffer.GPUPointer,desc->buffer.size);
                    break;
                case GAFW::GPU::Buffer::Buffer_UnkownSize:
                    //If it is unkown size then we know that the alloc thread is already freezed
                    // for now we do not do anything
                    break;
                case GAFW::GPU::Buffer::Buffer_Normal:
                    //the easiest case left till the end
                    data.bufferGPUPointer=desc->buffer.GPUPointer;
                    data.buffersize=desc->buffer.size; //if it is size 0 and NULL this code will not have effect
                    break;
                default:
                    throw GeneralException2("Buffer mode set to an unknown value",desc->arrayOperator);
            }


            //Everything is populated except the start event...
            this->logDebug(other,desc->arrayOperator,"Creating start event");
            checkCudaError(cudaEventRecord(*(desc->eventStart),this->kernelStream),"Unable to record event");
            //Everything is populated.. Now instruct the operator to load itself to the GPU
            logDebug(other,desc->arrayOperator,"Submitting to GPU");
            desc->arrayOperator->submitToGPU(data) ;
            checkCudaError(cudaGetLastError(),string("An error has been detected after submission of ")+string(desc->arrayOperator->objectName)+ string(" Operator Name:")+string(desc->arrayOperator->name));
            if (!data.endEventRecorded)
            {
                logDebug(other,desc->arrayOperator,"Recording end event");
                checkCudaError(cudaEventRecord(*(desc->eventEnd),this->kernelStream),"Unable to record event");
            }
#if 0            
            checkCudaError(cudaEventSynchronize(*(desc->eventEnd)),"error while synchronizing");
            checkCudaError(cudaGetLastError(),string("An error has been detected after submission of ")+string(desc->arrayOperator->objectName)+ string(" Operator Name:")+string(desc->arrayOperator->name));
#endif       
            
            //We have agin to check buffer as to unlock

            switch (desc->buffer.mode)
            {
                case GAFW::GPU::Buffer::Buffer_DeallocBeforeSubmit:
                     //Ok we have to freeze the allocation thread
                    this->gpuAllocMutex.unlock(); //Allocation can continue
                    break;
                case GAFW::GPU::Buffer::Buffer_UnkownSize:
                    //we must unfreeze the allocation thread 
                    // for now we do not do anything
                    this->SubmitReadyForUnknownBuffer=true;
                    break;
                default:
                    break;
            }



            //Submission ready
            this->waitKernelExecutionQueue.push(desc);

         }
        memAllocKernelSubmitMutex.unlock();
     }
     
 
 }
void CUDADeviceManager::threadWaitKernelExecutionReadyFunc()
{
    this->logInfo(other,"Thread WaitKernelExecutionReady initialised");
    checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
    for (;;)
    {
         OperationDescriptor * desc;
         this->waitKernelExecutionQueue.pop_wait(desc);
         if (desc->specialActions==OperationDescriptor::ThreadShutdown)
             break; //Loop 
         
         this->logInfo(other,desc->arrayOperator,"threadWaitKernelExecutionReadyFunc() Popped in");
        
         //Just sync on the end event and move on
         checkCudaError(cudaEventSynchronize(*(desc->eventEnd)),"Error while synchronising with end event");
         //WE need to get statistics immediately... TODO
         if (cudaEventElapsedTime(&desc->kernelExecutionDuration, *desc->eventStart, *desc->eventEnd)!=cudaSuccess)//,"Unable to calculate duration of kernel execution");
         ; //    std::cout << " An event was not well recorded";
         
         checkCudaError(cudaGetLastError(),"Unknown error");
         
         this->eventPool.checkInEvent(desc->eventStart);
         this->eventPool.checkInEvent(desc->eventEnd);
         desc->eventEnd=NULL;
         desc->eventStart=NULL;
         
         //Run post execution
         desc->arrayOperator->postRunExecute(desc->data);
         if (desc->buffer.GPUPointer!=NULL)
         {    
             this->gpuMemFree(&desc->buffer.GPUPointer,desc->buffer.size);
             
         }
         
         this->logInfo(other,desc->arrayOperator,"threadWaitKernelExecutionReadyFunc() Pushing out");
         this->readyQueue.push(desc);
         
         
    }   
}
        
 void CUDADeviceManager::threadPostCalcWorkFunc()
 {
     this->logDebug(other,"Thread PostCalc initialised");
     checkCudaError(cudaSetDevice(this->deviceNo),"Error while setting device");
     std::list<OperationDescriptor *> myList; 
     for (;;)
     {
         
         
         //In this thread we are in a hurry as to deallocate ...we only pop_wait if our list has no elments
         OperationDescriptor * desc;
        
         bool newElement=true;
       
         while(newElement)
         {
            if (myList.size()==0)
            {
                
                this->readyQueue.pop_wait(desc);
                newElement=true;
            
            }
            else
            {
                newElement=this->readyQueue.pop(desc);
            }
            if (desc->specialActions==OperationDescriptor::ThreadShutdown)
             return; //Loop 
         
            //Do the initial stuff here
            if (newElement)
            {   this->logDebug(other,desc->arrayOperator,"Popped in");
                //We need to increase counters immediately
                for (unsigned int i=0;i<desc->noOfInputs;i++)
                {
                    boost::mutex::scoped_lock lock(desc->inputs[i]->DataMutex);
                    desc->inputs[i]->GPUKnownReadyOperations[this->deviceNo]++; 
                }
                for (unsigned int i=0;i<desc->noOfOutputs;i++)
                {
                    boost::mutex::scoped_lock lock(desc->outputs[i]->DataMutex);
                    desc->outputs[i]->GPUKnownReadyOperations[this->deviceNo]++; 
                }

                postOperationManagment(desc);
                myList.push_back(desc);
            }
         }
            if (desc->specialActions==OperationDescriptor::ThreadShutdown)
            {
                    return;
            }
                
         
         //Ok now we have to go through all the list and do all memory managment.... 
         //If everything is done and the lement is at the begining of teh list then it is pushed out to
         // the engine queue.. Relevant dataDescriptors are sent to the other queue
         for (std::list<OperationDescriptor *>::iterator i=myList.begin();
                 ((myList.size()!=0)&&(i!=myList.end()));
                 /*The iterator is incremented within loop*/)
         {
            if (postOperationManagment(*i)&&(i==myList.begin()))
            {
                OperationDescriptor *myDesc=*i;
                //Ok this entry is completely ready and we can now put in the outputBuffer
                //We have also to take care of counters and insert in lists
                for (unsigned int n=0;n<(myDesc->noOfInputs+myDesc->noOfOutputs);n++)
                {
                    DataDescriptor *data;
                    if (n<myDesc->noOfInputs) data=myDesc->inputs[n];
                    else {
                        data=myDesc->outputs[n-myDesc->noOfInputs];
                        data->underCalculation=false;
                    }
                    data->DataMutex.lock();
                    data->GPUKnownOperations[this->deviceNo]--;
                    data->GPUKnownReadyOperations[this->deviceNo]--;
                    data->relatedOperationsCounter--;
                    
                    if ((data->GPUPointer[this->deviceNo]!=NULL)&&(data->GPUKnownOperations[this->deviceNo]==0)&&(!data->inNonLinkedList[this->deviceNo]))
                    {
                         this->submitDataDescriptorForReview(data);
                    }
                    data->DataMutex.unlock();
                }
                //and that's it
                this->logInfo(other,myDesc->arrayOperator,"Pushing to engine ready queue");
                
                this->engine->submitOperationReady(myDesc);
                
                this->knownOperations-=1;
                myList.erase(i);
                i=myList.begin();
            }
            else
            {
                i++;
            }
            
        } 
         
       
         
         
     }
 }
 void CUDADeviceManager::freeGPUMemory(DataDescriptor * data )
 {
     
     this->logDebug(other,data->array,"Freeing memory");
     
        if (data->linkedForOverwrite[this->deviceNo]) 
         {
                data->GPUPointer[this->deviceNo]=NULL;
                data->unsetLinkForOverWrite(this->deviceNo);
                
            }
            else
            {
                this->gpuMemFree(&data->GPUPointer[this->deviceNo],data->size);
                
            }
 
 }    
 void CUDADeviceManager::threadExtraAllocatesFunc()
 {
     //This threads just takes care of those extra allocations that are not deallocated 
     //since we try not to deallocate possible future allocates 
     this->logDebug(other,"Thread ExtraAlloc initialised");
     std::list<DataDescriptor *> myList;
     for (;;)
     {  
         DataDescriptor * data;
         this->allocDataQueue.wait();
         while(this->allocDataQueue.pop(data))
         {
             this->logInfo(other,"New entry popped in");
             if (data->specialActions==DataDescriptor::ThreadShutdown)
             {
                return; 
             }
                
            if (data->specialActions==DataDescriptor::Cache_and_deallocate)
            {
                this->logInfo(other,"Entry is a cache and deallocate entry");
                //we have to immediately cache everything and deallocate.....but we do not worry that much 
                //as this means that the  global atomic parameter cache_and_deallocate is set to true...
                //w ejust delete this desc
                delete data;
                data=NULL;
            }
            if (data!=NULL)
            {
                boost::mutex::scoped_lock lock(data->DataMutex);
                //First of all assert that counter is right
                data->SubmitForGPUReviewCounter[this->deviceNo]--;
                if (data->SubmitForGPUReviewCounter[this->deviceNo]<0) throw GeneralException("BUG::Counter is less then 0");
                
                //First very important question....
                //Should this object be maintained by this thread?
                if (data->GPUPointer[this->deviceNo]==NULL)
                {
                    //This might have been sent by the engine as to tell me to remove from my list as to delete
                    if (!data->inNonLinkedList[this->deviceNo])
                        this->engine->submitDataDescriptorReview(data);
                    //Essentially we ignore.. but for every ignore it is best as to send to engine
                    //we do not send only if we know about it and thsu we will process later
                    continue;
                }
                bool deallocate;
                bool cacheOk=this->handleCache(data,deallocate);
                if (cacheOk&&deallocate)
                {
                    this->freeGPUMemory(data);
                     //We are ignoring again..so we send to engine
                    if (!data->inNonLinkedList[this->deviceNo])
                        this->engine->submitDataDescriptorReview(data);
                    
                }
                else
                {
                    if (!data->inNonLinkedList[this->deviceNo])
                            if (data->GPUPointer[this->deviceNo]!=NULL)
                            {    
                                data->inNonLinkedList[this->deviceNo]=true;
                                myList.push_back(data);
                            }

                }
            }
         }
         bool loop=true;
         while (loop)
         {
             loop=false;
             for(std::list<DataDescriptor *>::iterator i=myList.begin();i!=myList.end();/*increase in loop*/)
             {
                 data=*i;
                 boost::mutex::scoped_lock lock(data->DataMutex);
             
                if (data->GPUPointer[this->deviceNo]!=NULL)
                {
                    bool deallocate;
                    bool cacheOk=this->handleCache(data,deallocate);

                    if (cacheOk&&deallocate)
                    {
                        this->freeGPUMemory(data);
                                
                    }

                    //Should we still maintain in list 
                    if (!cacheOk) 
                    {
                        loop=true;
                        continue;
                    }
                }    
                if (data->GPUPointer[this->deviceNo]==NULL)
                {
                    this->logInfo(other,"Removing from list as memory has been deallocated");
                    
                    data->inNonLinkedList[this->deviceNo]=false;
                    i=myList.erase(i);
                    this->engine->submitDataDescriptorReview(data);
                    
                    continue;
                }
                if (data->GPUKnownOperations[this->deviceNo]!=0)
                {
                    data->inNonLinkedList[this->deviceNo]=false;
                    this->logInfo(other,"Removing from list since it is known somewhere else");
                    i=myList.erase(i); 
                    //There is no need to inform engine in this case
                    continue;
                }
                i++;
             }
         }
     }
 }
        


CUDADeviceManager::~CUDADeviceManager()
{
    this->logInfo(other,"Sending shutdown descriptors to threads");
    OperationDescriptor *shutdown=new OperationDescriptor();
    
    shutdown->specialActions=OperationDescriptor::ThreadShutdown;
    this->copyWaitQueue.push(shutdown);
    this->gpuSubmitQueue.push(shutdown);
    this->newReqeustsQueue.push(shutdown);
    this->readyQueue.push(shutdown);
    this->waitKernelExecutionQueue.push(shutdown);
    
    DataDescriptor *shutdownd=new DataDescriptor(this->deviceNo);
    shutdownd->specialActions=DataDescriptor::ThreadShutdown;
    
    this->allocDataQueue.push(shutdownd);
    this->threadGPUSubmit->join();
    this->logInfo(other,"GPUSubmit Thread returned");
    this->threadExtraAllocates->join();
    this->logInfo(other,"Extra allocation thread returned");
    
    this->threadPostCalcWork->join();
    this->logInfo(other,"Post Calculation thread returned");
    this->threadWaitCopyToGPU->join();
    this->logInfo(other,"Wait For Copy to GPU thread ended");
    this->threadWaitKernelExecutionReady->join();
    this->logInfo(other,"Wait For Kernel Execution Ready Returned");
    this->threadAlloc->join();
    this->logInfo(other,"Allocation thread returned");
    
    
    
    delete shutdown;
    delete shutdownd;
    cudaSetDevice(this->deviceNo);
    cudaDeviceReset();
    
           

}

bool CUDADeviceManager::submitOperation(OperationDescriptor *desc)
{
    this->newReqeustsQueue.push(desc);
    return true;
}

bool CUDADeviceManager::postOperationManagment(OperationDescriptor *desc)
{
    //Ok many things to think about
    //First let's anlayze inputs.... 
    //IMP: Inputs never have new data, and for sure we do not need to copy to result array
    //but we might need to cache still since it might have been an output before
    bool ready=true;
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        ready=ready&&this->dataPostOperationProcessing(desc->inputs[i]);
    }
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        ready=ready&&this->dataPostOperationProcessing(desc->outputs[i]);
    }
    return ready;
     
}
bool CUDADeviceManager::dataPostOperationProcessing(DataDescriptor *data)
{
        bool toReturn=true; 
        boost::mutex::scoped_lock lock(data->DataMutex);
        
        if (data->GPUPointer[this->deviceNo]==NULL) 
        {
            return true; /// there is no post work to do on this
        }
        bool deallocate; //will be set by below
        toReturn=this->handleCache(data,deallocate); 
        //We now have a look if we are supposed to copy to some result
        if (!data->copyToReady)
        {
            //Ok we need to copy.. but is it already in progress
            if (data->MemCpyGPUToSystemInProgress)
            {
                 if (data->copyToSystemDeviceNo==this->deviceNo)
                 {

                    cudaError_t err=cudaEventQuery(*(data->GPUToSystemEndEvent));
                    switch(err)
                    {
                        case cudaErrorNotReady:
                            toReturn=false; //we will have to wait for caching
                            break;
                        case cudaSuccess:
                            data->MemCpyGPUToSystemInProgress=false;
                            data->copyToReady=true;
                            //We are still ready in this case so toReturn will leave as before
                            data->resultDataStore->setDataAsValid();
                            this->eventPool.checkInEvent(data->GPUToSystemEndEvent);
                            data->GPUToSystemEndEvent=NULL;
                            this->eventPool.checkInEvent(data->GPUToSystemStartEvent);
                            data->GPUToSystemStartEvent=NULL;
                            break;
                        default:
                           checkCudaError(err,"Unexpected error on querying end event");
                    }
                 }
            }
            else
            {
                //we need to imedieatly request a copy
                
                data->GPUToSystemStartEvent=this->eventPool.requestEvent();
                data->GPUToSystemEndEvent=this->eventPool.requestEvent();
                
                checkCudaError(cudaEventRecord(*(data->GPUToSystemStartEvent),this->memoryStream),"Unable to record event");
                checkCudaError(cudaMemcpyAsync(data->copyTo,data->GPUPointer[this->deviceNo],data->size,cudaMemcpyDeviceToHost,this->memoryStream),"Error while issue memcpy()");
                checkCudaError(cudaEventRecord(*(data->GPUToSystemEndEvent),this->memoryStream),"Unable to record event");
                data->copyToSystemDeviceNo=this->deviceNo;
                data->MemCpyGPUToSystemInProgress=true;
                toReturn=false;
            }
        }
        //Final thing .. should we deallocate GPU memory
        if (toReturn==false) // This clearly indicates that I shoudl not deallocate as there is copying in progress
        {
            return false;
        }
        
        if (deallocate)  //note if there is a copyTO copy then process will not arrive here
           this->freeGPUMemory(data);
        return true;
      
        
}
bool CUDADeviceManager::handleCache(DataDescriptor *data , bool &deallocate)
{
    //Note data->DataMutex is assumed to be already locked
    //let's not have surprises.. we get the value of cache and deallocate here and use it
    deallocate=false;
    bool _cache_and_deallocate=this->cache_and_deallocate;
    
    
    
    if (data->cache==NULL)
    {
        bool doCaching=data->requireImmediateCache;
        //Ok let's decide if we need caching
        ///Other factors as to cache
        //Does it apply for caching?
        if (_cache_and_deallocate||data->linkedForOverwrite[this->deviceNo])
        {
            //WE will cache if it really applies for caching
            if (data->reusable) doCaching=true;
            if (data->relatedOperationsCounter!=0)
                if (data->GPUKnownOperations[this->deviceNo]!=data->relatedOperationsCounter)
                        doCaching=true;
            if ((data->linkedForOverwrite[this->deviceNo]==false)&&data->forOverwrite)
                doCaching=true;
            //There are unknown operations that might require this
            //we say "might" because that unkown operation might be scheduled to another GPU
            
                
        }
        if (doCaching)
        {
            this->logDebug(other, string("Caching ")+data->array->objectName);
            if (data->cachingDeviceNo > -1) throw GeneralException("Request to create cache when cache already was done");
            //above is just an assertion.. It should never happen as cache would not be null
            data->cachingDeviceNo=this->deviceNo;
            checkCudaError(cudaHostAlloc(&(data->cache),data->size,cudaHostAllocPortable),"Error during allocation of memory on GPU");
            data->GPUToCacheStartEvent=this->eventPool.requestEvent();
            data->GPUToCacheEndEvent=this->eventPool.requestEvent();
            
            checkCudaError(cudaEventRecord(*(data->GPUToCacheStartEvent),this->memoryStream),"Unable to record event");
            checkCudaError(cudaMemcpyAsync(data->cache,data->GPUPointer[this->deviceNo],data->size,cudaMemcpyDeviceToHost,this->memoryStream),"Error while issue memcpy()");
            checkCudaError(cudaEventRecord(*(data->GPUToCacheEndEvent),this->memoryStream),"Unable to record event");
            data->MemCpyGPUToCacheInProgress=true;
            //Since we have requested caching then clearly we cannot deallocate for now... so just return false;
            return false;
        }
    }
    if ((data->MemCpyGPUToCacheInProgress)&&(data->cachingDeviceNo==this->deviceNo))
    {
        if ((data->MemCpyGPUToCacheInProgress)&&(data->cachingDeviceNo==this->deviceNo))
        {
            //ok caching in progress ... but let's check if ready
            cudaError_t err=cudaEventQuery(*(data->GPUToCacheEndEvent));
            if (err==cudaErrorNotReady)
            {
                return false;   //memory should not be deallocated as cache is still in progress
            }
            if (err==cudaSuccess)
            {
                data->MemCpyGPUToCacheInProgress=false;
                this->eventPool.checkInEvent(data->GPUToCacheStartEvent);
                data->GPUToCacheStartEvent=NULL;
                this->eventPool.checkInEvent(data->GPUToCacheEndEvent);
                data->GPUToCacheEndEvent=NULL;
                data->requireImmediateCache=false;
                
                
            }
            checkCudaError(err,"Unexpected error on querying end event");
            
        }
    }
    
    //Should a deallocation happen?
    //deallocate for now will be false... we just return true if we want it false 
   // if (data->linkedForOverwrite[this->deviceNo])
   //     deallocate=false;
    if (data->MemCpyGPUToSystemInProgress)
        deallocate=false; //there is copy going around 
    else if (data->GPUKnownOperations[this->deviceNo]==data->GPUKnownReadyOperations[this->deviceNo]) //we do not seem to need it currently
    {
        //if above is 0 then it must be coming from the extra alloc thread
        if (data->linkedForOverwrite[this->deviceNo]) deallocate=true;  //check on this... since deallocation will just remove the link
        else if (_cache_and_deallocate) deallocate=true; //we would not have cached if related operations Counter is 0
        else if (data->reusable) deallocate=false; 
        else if (data->forOverwrite) deallocate=false;
        else if (data->relatedOperationsCounter==data->GPUKnownReadyOperations[this->deviceNo]) deallocate=true;  //There is not more need for caching
    }
    else
    {
            deallocate=false;
    }
    return true;
    
}
void CUDADeviceManager::reverse_allocateAndPopulate(OperationDescriptor *desc)
{
    bool ready=false;
    while (!ready)
    {
        ready=true;
        //First work on inputs
        for (unsigned int i=0;i<desc->noOfInputs;i++)
        {
            DataDescriptor *input=desc->inputs[i];
            boost::mutex::scoped_lock lock(input->DataMutex);
            if (input->GPUPointer[this->deviceNo]!=NULL)
            {
                //We only work on allocated memory
                input->requireImmediateCache=true;
                bool useless;
                bool cacheReady=this->handleCache(input,useless);
                if (cacheReady)
                {
                    //We can deallocte
                    //Ok we can safely deallocate
                    this->gpuMemFree(&input->GPUPointer[this->deviceNo],input->size);
                    
                }
                else
                {
                    ready=false;
                }


            }
        }
        //Time for outputs
        for (unsigned int i=0;i<desc->noOfOutputs;i++)
        {
            //OK output can have something important if we have overwrite
            DataDescriptor *output=desc->outputs[i];
            boost::mutex::scoped_lock lock(output->DataMutex);
            if (output->GPUPointer[this->deviceNo]!=NULL)
            {   
                if (output->overwrite==NULL)
                {
                    //We just deallocate
                    this->gpuMemFree(&output->GPUPointer[this->deviceNo],output->size);
                    output->underCalculation=false;
                }
                else
                {
                    DataDescriptor *overwrite=output->overwrite;
                    boost::mutex::scoped_lock lock(overwrite->DataMutex);
                    if (overwrite->linkedForOverwrite[this->deviceNo])
                    {
                        //Unlink immediatly and pointer can be null
                        output->GPUPointer[this->deviceNo]=NULL;
                        overwrite->unsetLinkForOverWrite(this->deviceNo);
                        output->underCalculation=false;
                        
                    }
                    else
                    {
                        //This means there must be a copy from overwrite cache
                        if (overwrite->MemCpyCacheToForeign[this->deviceNo])
                        {
                            //is it ready???
                            cudaError_t err=cudaEventQuery(*(overwrite->cacheToGPUEndEvent[this->deviceNo]));
                            if (err==cudaErrorNotReady)
                            {
                                ready=false;
                            }
                            else
                            {
                                checkCudaError(err,"Unexpected error on querying end event");
                                //This means success
                                overwrite->MemCpyCacheToForeign[this->deviceNo]=false;
                                this->eventPool.checkInEvent(overwrite->cacheToGPUStartEvent[this->deviceNo]);
                                overwrite->cacheToGPUStartEvent[this->deviceNo]=NULL;
                                this->eventPool.checkInEvent(overwrite->cacheToGPUEndEvent[this->deviceNo]);
                                overwrite->cacheToGPUEndEvent[this->deviceNo]=NULL;
                            }
                        }
                        if (!overwrite->MemCpyCacheToForeign[this->deviceNo])
                        {
                            //We can directly unallocate .... 
                            this->gpuMemFree(&output->GPUPointer[this->deviceNo],output->size);
                            output->underCalculation=false;
                        }
                    }
                    //We need to make sure that overwrite is cached
                    if (overwrite->GPUPointer[this->deviceNo]!=NULL)
                    {
                        overwrite->requireImmediateCache=true;
                        bool useless;
                        bool cacheReady=this->handleCache(overwrite,useless);
                        if (cacheReady)
                        {
                    //We can deallocte
                    //Ok we can safely deallocate
                            this->gpuMemFree(&overwrite->GPUPointer[this->deviceNo],overwrite->size);
                              
                        }
                        else
                        {
                                ready=false;
                        }
                    }
                }
            }
        }
                    
        
    }
}
enum CUDADeviceManager::AllocAndPopulateReturn CUDADeviceManager::allocateAndPopulate(OperationDescriptor *desc)
{
    boost::mutex::scoped_lock lock (this->gpuAllocMutex);   //Submit thread will freeze up if it has to deallocate a buffer
    bool missingCache=false;  //will be set to true if we need to wait for some cache to be filled from another GPU
    bool overwriteLink=false; //set to true of there is some input/output is linked for an overwrite
    bool allocFailed=false; //True when a memory allocation failed
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *input=desc->inputs[i];
        boost::mutex::scoped_lock lock(input->DataMutex);
        if (input->linkedForOverwrite[this->deviceNo])
        {
            //WE can't do anything ... we have to wait until the link is gone
            overwriteLink=true;
            continue;
        }
        if (input->GPUPointer[this->deviceNo]==NULL) //We are only interested in allocation... so only with pinter NULL
        {
           
            //Check taht cahe contains valid data first
            if (input->cache==NULL)
            {
                input->requireImmediateCache=true;
                //We do not continue
                //TODO.. we have to signal that we need a copy form another GPU
                missingCache=true;
                continue;
            }
            //Ok Check if there are any copies to cache... ie still being populated
            if (input->MemCpyGPUToCacheInProgress)
            {
                missingCache=true;
                continue;
            }
            bool success;
            int current_gpuMemoryAllocation;
            do 
            {
                current_gpuMemoryAllocation=(int)this->gpuMemoryAllocated;
                atomic_thread_fence(boost::memory_order_release);
                success=this->gpuMemAlloc(&(input->GPUPointer[this->deviceNo]),input->size);
                if (success) break;
            }
            while(current_gpuMemoryAllocation!=int(this->gpuMemoryAllocated));
            if (!success)
            {
                allocFailed=true;
                continue;
            }
            //Decide from where to copy
            //It must be in cache/... We always assume it is valid//
            //But we check for bugs
            if (!input->cache) throw GeneralException("Cache found to be NULL!");
            input->cacheToGPUStartEvent[this->deviceNo]=this->eventPool.requestEvent();
            input->cacheToGPUEndEvent[this->deviceNo]=this->eventPool.requestEvent();
            
            checkCudaError(cudaEventRecord(*(input->cacheToGPUStartEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
            checkCudaError(cudaMemcpyAsync(input->GPUPointer[this->deviceNo],input->cache, input->size,cudaMemcpyHostToDevice,this->memoryStream),"Error while issue memcpy()");
            checkCudaError(cudaEventRecord(*(input->cacheToGPUEndEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
            input->MemCpyCacheToGPUInProgress[this->deviceNo]=true;
        }
    }
    //Time for outputs
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        DataDescriptor *output=desc->outputs[i];
        boost::mutex::scoped_lock lock(output->DataMutex);
        if (output->GPUPointer[this->deviceNo]==NULL)
        {
            if (output->overwrite==NULL)  //The simple case
            {
                if (!this->gpuMemAlloc(&(output->GPUPointer[this->deviceNo]),output->size)) 
                        allocFailed=true;
                else
                    output->underCalculation=true;
                continue;
            }
            
            if (output->overwrite!=NULL)  //not very simple
            {
                boost::mutex::scoped_lock lock(output->overwrite->DataMutex);
                //ok we should fill up what we allocated with overwrite, but we might not even allocate at all
                //Is it on GPU??
                if (output->overwrite->GPUPointer[this->deviceNo]!=NULL)
                {
                    output->GPUPointer[this->deviceNo]=output->overwrite->GPUPointer[this->deviceNo];
                    output->overwrite->linkedForOverwrite[this->deviceNo]=true;
                    output->underCalculation=true;
                    continue;
                    //Since we might still not even have any data in that pointer caching etc.. have to be done else where
                }
                else
                {
                    //Then it must be in cache...If not we need to request the data from anther GPU
                    if (output->overwrite->cache==NULL)
                    {
                        output->overwrite->requireImmediateCache=true;
                        missingCache=true;
                        continue;
                    }
                    else if (output->overwrite->MemCpyGPUToCacheInProgress)
                    {
                        missingCache=true;
                        continue;
                    }
                    //The last els is done after normal alloc
                    //Ok we can try to allocate and the copy
                    
                    if (!this->gpuMemAlloc(&(output->GPUPointer[this->deviceNo]),output->size)) 
                    {
                        allocFailed=true;
                        continue;
                    }
                    output->underCalculation=true;
                    output->cacheToGPUStartEvent[this->deviceNo]=this->eventPool.requestEvent();
                    output->cacheToGPUEndEvent[this->deviceNo]=this->eventPool.requestEvent();
                    checkCudaError(cudaEventRecord(*(output->cacheToGPUStartEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
                    checkCudaError(cudaMemcpyAsync(output->GPUPointer[this->deviceNo],output->overwrite->cache, output->size,cudaMemcpyHostToDevice,this->memoryStream),"Error while issue memcpy()");
                    checkCudaError(cudaEventRecord(*(output->cacheToGPUEndEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
                    output->overwrite->MemCpyCacheToForeign[this->deviceNo]=true;
                    //This output is thus ok 
                    
                    
                }
            } //End of if overwrite logic
            
        
        }
    }  //End of for loop of outpi
    
    //bufferMode is not important here
    if (desc->buffer.GPUPointer==NULL)
    {
        if (desc->buffer.size!=0)
        {
            if (!this->gpuMemAlloc(&(desc->buffer.GPUPointer),desc->buffer.size))
            {
                    allocFailed=true;
            }
        }
    }
    
    if (allocFailed) return CUDADeviceManager::UnableToAlloc;
    if (missingCache) return CUDADeviceManager::MissingCache;
    if (overwriteLink) return CUDADeviceManager::LinkedForOverwrite;
    return CUDADeviceManager::AllocSuccess; 
    //Ok time to decide what to return according to the flags set up
    
}
size_t CUDADeviceManager::getAllocMemSize(OperationDescriptor *desc)
{
    size_t ret=0;
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *input=desc->inputs[i];
        boost::mutex::scoped_lock lock(input->DataMutex);
        if (input->GPUPointer[this->deviceNo]!=NULL) //We are only interested in allocation... so only with pinter NULL
        {
            ret+=input->size;
        }
    }
    //Outputs
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        DataDescriptor *output=desc->outputs[i];
        boost::mutex::scoped_lock lock(output->DataMutex);
        if (output->GPUPointer[this->deviceNo]!=NULL)
        {
            ret+=output->size;
        }
    }  //End of for loop of output
    
    //bufferMode is not important here
    if (desc->buffer.GPUPointer!=NULL)
    {
        ret+=desc->buffer.size;
    }
    return ret;
    
}
size_t CUDADeviceManager::getMoreAllocRequiredSize(OperationDescriptor *desc)
{
    size_t ret=0;
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *input=desc->inputs[i];
        boost::mutex::scoped_lock lock (input->DataMutex);
        if (input->GPUPointer[this->deviceNo]==NULL) //We are only interested in allocation... so only with pinter NULL
        {
            ret+=input->size;
        }
    }
    //Outputs
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        DataDescriptor *output=desc->outputs[i];
        boost::mutex::scoped_lock lock (output->DataMutex);
        if (output->GPUPointer[this->deviceNo]==NULL)
        {
            ret+=output->size;
        }
    }  //End of for loop of output
    
    //bufferMode is not important here
    if (desc->buffer.GPUPointer==NULL)
    {
        ret+=desc->buffer.size;
    }
    return ret;
    
}

void CUDADeviceManager::allocateAndPopulate(OperationDescriptor* desc,bool &noMoreAlloc,bool &missingCache, int &totalAlloc)
{
    boost::mutex::scoped_lock lock(this->gpuAllocMutex);   //Submit thread will freeze up if it has to deallocate a buffer
    missingCache=false;
    //First work on inputs
    totalAlloc=0;
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *input=desc->inputs[i];
        boost::mutex::scoped_lock lock(input->DataMutex);
        if (input->linkedForOverwrite[this->deviceNo])
        {
            //WE can't do anything ... we have to wait until the link is gone
            noMoreAlloc=true;
            continue;
        }
        if (input->GPUPointer[this->deviceNo]==NULL) //We are only interested in allocation... so only with pinter NULL
        {
            if (noMoreAlloc) continue;
            //Check taht cahe contains valid data first
            if (input->cache==NULL)
            {
                if (input->requireImmediateCache==false)
                {
                        input->requireImmediateCache=true;
                        this->engine->submitDataDescriptorReview(input); //The engine will send "the alert" to the right CUDADeviceManager 
                }
                //We do not continue
                noMoreAlloc=true;
                missingCache=true;
                continue;
            }
            //Ok Check if there are any copies to cache... ie still being populated
            if (input->MemCpyGPUToCacheInProgress)
            {
                noMoreAlloc=true;
                missingCache=true;
                continue;
            }
            bool success;
            int current_gpuMemoryAllocation;
            do 
            {
                current_gpuMemoryAllocation=(int)this->gpuMemoryAllocated;
                atomic_thread_fence(boost::memory_order_release);
                success=this->gpuMemAlloc(&(input->GPUPointer[this->deviceNo]),input->size);
                if (!success) continue;
                
                //This means a success
                
                success=true;
                totalAlloc+=input->size;
                break;
            }
            while(current_gpuMemoryAllocation!=int(this->gpuMemoryAllocated));
            if (!success)
            {
                noMoreAlloc=true;
                continue;
            }
            //Decide from where to copy
            //It must be in cache/... We always assume it is valid//
            //But we check for bugs
            if (!input->cache) throw GeneralException("Cache found to be NULL!");
            input->cacheToGPUStartEvent[this->deviceNo]=this->eventPool.requestEvent();
            input->cacheToGPUEndEvent[this->deviceNo]=this->eventPool.requestEvent();
            
            checkCudaError(cudaEventRecord(*(input->cacheToGPUStartEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
            checkCudaError(cudaMemcpyAsync(input->GPUPointer[this->deviceNo],input->cache, input->size,cudaMemcpyHostToDevice,this->memoryStream),"Error while issue memcpy()");
            checkCudaError(cudaEventRecord(*(input->cacheToGPUEndEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
            input->MemCpyCacheToGPUInProgress[this->deviceNo]=true;
        }
        else
        {
            totalAlloc+=input->size;
        }
    }
    //Time for outputs
    for (unsigned int i=0;i<desc->noOfOutputs;i++)
    {
        
        DataDescriptor *output=desc->outputs[i];
        boost::mutex::scoped_lock lock(output->DataMutex);
        if (output->GPUPointer[this->deviceNo]==NULL)
        {
            if (output->overwrite)
            {
                output->overwrite->DataMutex.lock();
                //ok we should fill up what we allocated with overwrite, but we might not even allocate at all
                //Is it on GPU
                if (output->overwrite->GPUPointer[this->deviceNo]!=NULL)
                {
                    output->GPUPointer[this->deviceNo]=output->overwrite->GPUPointer[this->deviceNo];
                    output->underCalculation=true;
                    if (output->overwrite->GPUKnownOperations[this->deviceNo]==0)
                    {
                        output->overwrite->GPUPointer[this->deviceNo]=NULL;
                    }
                    else
                    {
                        output->overwrite->linkedForOverwrite[this->deviceNo]=true;
                    }
                    totalAlloc+=output->size;
                    output->overwrite->DataMutex.unlock();
                    continue;
                    //Since we might still not even have any data in that pointer caching etc.. have to be done else where
                    
                }
                else
                {
                    //The it must be in cache...
                    if (output->overwrite->cache==NULL)
                    {
                        if (!output->overwrite->requireImmediateCache)
                        {
                                output->overwrite->requireImmediateCache=true;
                                this->engine->submitDataDescriptorReview(output->overwrite);
                        }
                        noMoreAlloc=true;
                        missingCache=true;
                        output->overwrite->DataMutex.unlock();
                        continue;
                    }
                    else if (output->overwrite->MemCpyGPUToCacheInProgress)
                    {
                        noMoreAlloc=true;
                        missingCache=true;
                        output->overwrite->DataMutex.unlock();
                        continue;
                    }
                    //The last els is done after normal alloc
                }
            } 
            if (noMoreAlloc)
            {  
                if (output->overwrite!=NULL)
                    output->overwrite->DataMutex.unlock();
                continue;
            }
            //We are only interested in allocation... so only with pointer NULL
            if (!this->gpuMemAlloc(&(output->GPUPointer[this->deviceNo]),output->size)) 
            {
                noMoreAlloc=true;
                if (output->overwrite!=NULL)
                    output->overwrite->DataMutex.unlock();
                continue;
            }
            output->underCalculation=true;
            totalAlloc+=output->size;
            if (output->overwrite!=NULL)
            {
               //Ojk we need to copy from overwrite cache
                output->cacheToGPUStartEvent[this->deviceNo]=this->eventPool.requestEvent();
                output->cacheToGPUEndEvent[this->deviceNo]=this->eventPool.requestEvent();
                
                
                checkCudaError(cudaEventRecord(*(output->cacheToGPUStartEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
                checkCudaError(cudaMemcpyAsync(output->GPUPointer[this->deviceNo],output->overwrite->cache, output->size,cudaMemcpyHostToDevice,this->memoryStream),"Error while issue memcpy()");
                checkCudaError(cudaEventRecord(*(output->cacheToGPUEndEvent[this->deviceNo]),this->memoryStream),"Unable to record event");
                output->overwrite->MemCpyCacheToForeign[this->deviceNo]=true;
                output->overwrite->DataMutex.unlock();
            }
            
        
        }
        else
        {
            totalAlloc+=output->size;
        }
            
    }
    
    //Last thing ... buffer..if we can allocate it
    if (noMoreAlloc) return;
    
    //bufferMode is not important here
    if (desc->buffer.GPUPointer==NULL)
    {
        if (desc->buffer.size!=0)
        {
            if (!this->gpuMemAlloc(&(desc->buffer.GPUPointer),desc->buffer.size))
            {
                    noMoreAlloc=true;
                    return;
            
            }
            totalAlloc+=desc->buffer.size;
        }
    }
    //If we arrive up to here then it means that all memory for this operator is allocated, 
    
    
    
}
/*
void CUDADeviceManager::ArrayOperatorSubmit(OperationDescriptor* desc)
{
    GPUSubmissionData data;
    data.stream=this->kernelStream;
    data.postExecutePointer=&(desc->data);
    data.noOfInputs=desc->noOfInputs;
    data.noOfOutputs=desc->noOfOutputs;
    GPUArrayDescriptor inputs[data.noOfInputs];
    GPUArrayDescriptor outputs[data.noOfOutputs];
    data.inputs=inputs;
    data.outputs=outputs;
   // data.params=desc->params;
    //Processing inputs
    for (int i=0;i<data.noOfInputs;i++)
    {
            inputs[i].dim=desc->inputs[i]->dim;
            inputs[i].type=desc->inputs[i]->type;
            inputs[i].pointer=desc->inputs[i]->GPUPointer[this->deviceNo];
    }
    //Processing outputs
    for (int i=0;i<data.noOfOutputs;i++)
    {
            outputs[i].dim=desc->outputs[i]->dim;
            outputs[i].type=desc->outputs[i]->type;
            outputs[i].pointer=desc->outputs[i]->GPUPointer[this->deviceNo];
    }
    //Everything is populated except the start event...
    this->logDebug(other,desc->arrayOperator,"Creating start event");
    checkCudaError(cudaEventCreate(&(desc->eventStart)),"Unable to create event");
    checkCudaError(cudaEventRecord(desc->eventStart,this->kernelStream),"Unable to record event");
           
    
    data.startEvent=&(desc->eventStart);
    
    //Everything is populated.. Now instruct the operator to load itself to the GPU
    logDebug(other,desc->arrayOperator,"Submitting to GPU");
    desc->arrayOperator->submitToGPU(data) ;
    logDebug(other,desc->arrayOperator,"Creating end event");
    checkCudaError(cudaEventCreateWithFlags(&(desc->eventEnd),cudaEventBlockingSync),"Unable to create event");
    checkCudaError(cudaEventRecord(desc->eventEnd,this->kernelStream),"Unable to record event");
}
*/
bool CUDADeviceManager::checkIfReadyForSubmit(OperationDescriptor* desc)
{
    //OK check that all inputs copies have been done
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *input=desc->inputs[i];
        input->DataMutex.lock();
        if (input->MemCpyCacheToGPUInProgress[this->deviceNo])
        {
           //Ok check if it is still being done
            cudaError_t err=cudaEventQuery(*(input->cacheToGPUEndEvent[this->deviceNo]));
            if (err==cudaErrorNotReady)
            {
                input->DataMutex.unlock();
                return false;
            }
            checkCudaError(err,"Unexpected error on querying end event");
            //Good copy is ready
            input->MemCpyCacheToGPUInProgress[this->deviceNo]=false;
            this->eventPool.checkInEvent(input->cacheToGPUStartEvent[this->deviceNo]);
            this->eventPool.checkInEvent(input->cacheToGPUEndEvent[this->deviceNo]);
            input->cacheToGPUStartEvent[this->deviceNo]=NULL;
            input->cacheToGPUEndEvent[this->deviceNo]=NULL;
        }   
        input->DataMutex.unlock(); //This input is all right    
        
    }
    //outputs.. we mainly have to take care of overwrites
    for (unsigned int i=0;i<desc->noOfInputs;i++)
    {
        DataDescriptor *output=desc->outputs[i];
        if (output->overwrite==NULL) continue; // no point even in creating the mutex
        output->DataMutex.lock();
        output->overwrite->DataMutex.lock();
        //Two checks 
        if (output->overwrite->GPUPointer[this->deviceNo]==output->GPUPointer[this->deviceNo])
        {
            //there is still the link.. we have to wait
            output->overwrite->DataMutex.unlock();
            output->DataMutex.unlock();
            
            return false;
        }
        //Ok check if we have a copy in place
        if (output->overwrite->MemCpyCacheToForeign[this->deviceNo])
        {
           // check if it is still being done
            cudaError_t err=cudaEventQuery(*output->cacheToGPUEndEvent[this->deviceNo]);
            if (err==cudaErrorNotReady)
            {
                output->overwrite->DataMutex.unlock();
                output->DataMutex.unlock();
                return false;
            }
            checkCudaError(err,"Unexpected error on querying end event");
            //Good copy is ready
            output->overwrite->MemCpyCacheToForeign[this->deviceNo]=false;
            this->eventPool.checkInEvent(output->cacheToGPUEndEvent[this->deviceNo]);
            this->eventPool.checkInEvent(output->cacheToGPUStartEvent[this->deviceNo]);
            output->cacheToGPUEndEvent[this->deviceNo]=NULL;
            output->cacheToGPUStartEvent[this->deviceNo]=NULL;
            
            
        }   
        output->overwrite->DataMutex.unlock();
        output->DataMutex.unlock(); //This output is all right    
        
    }
    //No checks for buffers
    return true;
}

bool CUDADeviceManager::gpuMemAlloc(void ** pointer,size_t size)
{
    
    cudaError_t err=cudaMalloc(pointer,size);
    if (err==cudaErrorMemoryAllocation) //Ok memeory filled up
    {
         return false;
    }
    checkCudaError(err,"Error returned during cudaMalloc()");
    this->gpuMemoryAllocated+=size;
    return true;
}
void CUDADeviceManager::gpuMemFree(void ** pointer,size_t size)
{
    if (*pointer==NULL){
        this->logWarn(other,"Request to free a NULL on GPU. Ignored ");
        return;
    }
    checkCudaError(cudaFree(*pointer),"Unable to free GPU memory");
    *pointer=NULL;
    //if (err==cudaSuccess)
    this->gpuMemoryAllocated-=size;
    
}
void CUDADeviceManager::submitDataDescriptorForReview(DataDescriptor *desc)
{
    desc->SubmitForGPUReviewCounter[this->deviceNo]++;
    this->allocDataQueue.push(desc);
}
    