/* DataDescriptor.h:  Definition of the DataDescriptor class.
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

#ifndef DATADESCRIPTOR_H
#define	DATADESCRIPTOR_H
#include <boost/thread.hpp>
namespace GAFW { namespace GPU
{
    
    class DataDescriptor {
    private:
        DataDescriptor(const DataDescriptor& orig):noOfCudaDevices(0){};
    public:
        enum {
           NoAction,
           ThreadShutdown,
           Cache_and_deallocate
        } specialActions;
        const int noOfCudaDevices;
        GAFW::ArrayDimensions dim;
        GAFW::GeneralImplimentation::StoreType type;
        size_t size;
        
        boost::mutex DataMutex; //This mutex is to be invoked everytime teh data descripor is accessed/changed
        
        bool *inNonLinkedList;
        int *SubmitForGPUReviewCounter;
        void **GPUPointer; //array as large as how nmuch CUDA devices exist 
        //bool GPUDataReady[];
        cudaEvent_t **cacheToGPUStartEvent;
        cudaEvent_t **cacheToGPUEndEvent;
        cudaEvent_t *GPUToCacheStartEvent;
        cudaEvent_t *GPUToCacheEndEvent;
        cudaEvent_t *GPUToSystemStartEvent;
        cudaEvent_t *GPUToSystemEndEvent;
       
        int cachingDeviceNo; //The device no that is or has cached 
        
        bool MemCpyGPUToCacheInProgress;
        bool MemCpyGPUToSystemInProgress;
        bool *MemCpyCacheToGPUInProgress;
        bool *MemCpyCacheToForeign;
        
        bool *linkedForOverwrite;
        boost::mutex *linkedForOverwriteMutex;
        boost::condition_variable *linkedForOverWriteCondition; 
        bool underCalculation;
        
        
        int *GPUKnownOperations;
        void *cache; //storage on host
        //bool isCacheValid;
        bool requireImmediateCache; // ajknd of signal to other GPus that I need to cache
        GAFW::GPU::GPUDataStore *cacheDataStore; //NOT NULL if is a datastore .. ie an input array
        void *copyTo; //if /not NULL then once answer known it must be copied to copyTo location... (ie a result)
        bool  copyToReady;  //always true f copyTo is NULL,, fasle when copying has still not finished 0r not initiated
        int copyToSystemDeviceNo;
        //we need info on Result.. as to inform that result is ready
        bool reusable; //we should not delete if counter goes to zero.. vaildation might remove reusability after  
        DataDescriptor *overwrite;  //When not NULL it means that when this Array is an output then we have to copy from overwrite
        bool forOverwrite; 
        //bool *overwriteCacheCopy;
        int relatedOperationsCounter;//set initially by validation..descreases while an operation is ready and sent tpo the ready buffer  
        bool dealocateEverything;
        int * GPUKnownReadyOperations;
        GAFW::GPU::GPUDataStore *resultDataStore;
        bool validationReady;
        int engineKnownRelatedOperations;
        int sentToEngineDataQueueCounter;
        long long snapshot_no;
        GAFW::Array * array;
        //bool deleted;
        void deleteCache();
        void unsetLinkForOverWrite(int device_no);
        void wait_for_unsetLinkForOverwrite(int device_no);
        
        DataDescriptor(int noOfCudaDevices);
        
        
        virtual ~DataDescriptor();
    private:

    };
}};
#endif	/* DATADESCRIPTOR_H */

