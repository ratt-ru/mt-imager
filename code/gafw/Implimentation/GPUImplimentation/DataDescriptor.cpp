/* DataDescriptor.cpp:  Implementation of the DataDescriptor class.
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

#include <boost/thread/pthread/mutex.hpp>
#include <boost/thread/pthread/condition_variable_fwd.hpp>

#include "GPUafw.h"
using namespace GAFW;
using namespace GAFW::GPU;
        
DataDescriptor::DataDescriptor(int noOfCudaDevices):noOfCudaDevices(noOfCudaDevices)
{
        // GAFW::ArrayDimensions dim; set as defualt
        this->specialActions=NoAction;
        this->type=GAFW::GeneralImplimentation::StoreTypeUnknown;
        this->size=0;
        
        
        this->GPUToCacheStartEvent=NULL;
        this->GPUToCacheEndEvent=NULL;
        this->GPUToSystemStartEvent=NULL;
        this->GPUToSystemEndEvent=NULL;
        
        this->inNonLinkedList=new bool[noOfCudaDevices] ;
        
        this->GPUPointer=new void*[noOfCudaDevices];  
       
        this->cacheToGPUStartEvent=new cudaEvent_t*[noOfCudaDevices];
        this->cacheToGPUEndEvent=new cudaEvent_t*[noOfCudaDevices];
        this->cachingDeviceNo=-1; 
        
        this->MemCpyGPUToCacheInProgress=false;
        this->MemCpyGPUToSystemInProgress=false;
        this->MemCpyCacheToGPUInProgress=new bool[noOfCudaDevices];
        this->MemCpyCacheToForeign=new bool[noOfCudaDevices];
        this->linkedForOverwrite=new bool[noOfCudaDevices];
        this->linkedForOverwriteMutex=new boost::mutex[noOfCudaDevices];
        this->linkedForOverWriteCondition=new boost::condition_variable[noOfCudaDevices];
        
        
        this->GPUKnownOperations=new int[noOfCudaDevices];
        this->GPUKnownReadyOperations=new int[noOfCudaDevices];
        this->cache=NULL; 
        this->requireImmediateCache=false;
        this->cacheDataStore=NULL;
        this->copyTo=NULL; 
        this->copyToReady=true; 
        this->copyToSystemDeviceNo=-1;
        this->reusable=false; 
        this->overwrite=NULL; 
        this->forOverwrite=false;
        //this->forOverwrite=true; 
        this->relatedOperationsCounter=0;
        this->dealocateEverything=false;
        this->resultDataStore=NULL;
        this->validationReady=false;
        this->snapshot_no=0;
        
        this->engineKnownRelatedOperations=0;
        this->array=NULL;
        //this->deleted=false;
        this->sentToEngineDataQueueCounter=0;
        this->underCalculation=false;
        this->SubmitForGPUReviewCounter=new int[this->noOfCudaDevices];
        //Set up all arrays;
        for (int i=0;i<noOfCudaDevices;i++)
        {
            this->inNonLinkedList[i]=false; 
            this->GPUPointer[i]=NULL;  
            this->MemCpyCacheToForeign[i]=false;
            this->MemCpyCacheToGPUInProgress[i]=false;
            this->linkedForOverwrite[i]=false;
            this->GPUKnownOperations[i]=0;
            this->GPUKnownReadyOperations[i]=0;
            this->cacheToGPUStartEvent[i]=NULL;
            this->cacheToGPUEndEvent[i]=NULL;
            this->SubmitForGPUReviewCounter[i]=0;
        }
        
}
DataDescriptor::~DataDescriptor()
{
    delete[] this->inNonLinkedList;
    delete[] this->GPUPointer;  
    delete[] this->cacheToGPUStartEvent;
    delete[] this->cacheToGPUEndEvent;
    delete[] this->MemCpyCacheToForeign;
    delete[] this->linkedForOverwrite;
    delete[] this->GPUKnownOperations;
    delete[] this->GPUKnownReadyOperations;
    delete[] this->linkedForOverwriteMutex;;
    delete[] this->linkedForOverWriteCondition;
    delete[] this->MemCpyCacheToGPUInProgress;
        
    this->deleteCache();
    
    
}
void DataDescriptor::deleteCache()
{
    if ((this->cacheDataStore==NULL)&&(this->cache!=NULL))
        cudaFreeHost(this->cache);
    if (this->cacheDataStore!=NULL)
    {
        cacheDataStore->deleteSnapshot(this->snapshot_no);
        if (cacheDataStore->canDelete()) delete cacheDataStore;
    }
}


void DataDescriptor::unsetLinkForOverWrite(int device_no)
{
    boost::mutex::scoped_lock lock(this->linkedForOverwriteMutex[device_no]);
    this->linkedForOverwrite[device_no]=false;
    this->linkedForOverWriteCondition[device_no].notify_all();
}
void DataDescriptor::wait_for_unsetLinkForOverwrite(int device_no)
{
    boost::mutex::scoped_lock lock(this->linkedForOverwriteMutex[device_no]);
    while(this->linkedForOverwrite[device_no])
        this->linkedForOverWriteCondition[device_no].wait(lock);
}


 