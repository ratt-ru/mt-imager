/* GPUEngine.h:  Definition of the GPUEngine class. 
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

#ifndef __GPUENGINE_H__
#define	__GPUENGINE_H__

#include <vector>
#ifndef __CUDACC__
#include <boost/thread.hpp>
#endif
namespace GAFW { namespace GPU
{
    
    #ifdef __CUDACC__
    template<> class SynchronousQueue<std::vector<OperationDescriptor *> *>;
    template<> class SynchronousQueue<OperationDescriptor * >;
    class CUDADeviceManager;
    namespace boost
    {
    class thread;
    }
    #endif

    class GPUEngine: public GAFW::GeneralImplimentation::Engine {
    private:
        
        GPUEngine(const GPUEngine& orig){};
    protected:
        int noOfDevices;
        int nextDeviceSubmitTo;
        long long int snapshot_no; 
        CUDADeviceManager ** devices;
        GPUFactory *factory;
        boost::thread *threadCalcScheduling;
        boost::thread *threadOperationReady;
        boost::thread *threadHandleDataDescriptors;
        GAFW::FactoryStatisticsOutput *statisticsOutput;
        
        
        std::map<GAFW::GeneralImplimentation::Array *, DataDescriptor *> arrayDataDescriptorMap;
        std::vector<OperationDescriptor *> *calcVector;
        std::map<GAFW::GeneralImplimentation::Array*, DataDescriptor *> activeReusablesMap;
        std::map<GAFW::GeneralImplimentation::Array*, DataDescriptor *> toPublishReusablesMap;
        std::vector<GAFW::GeneralImplimentation::Array *> toDeleteAsReusable;
        
        SynchronousQueue<std::vector<OperationDescriptor *> *> * calculationRequestQueue;
        SynchronousQueue<OperationDescriptor * > * operationReadyQueue;
        SynchronousQueue<void *> * broadcastRequestQueue;
        SynchronousQueue<DataDescriptor *> * dataDescriptorQueue;
        void calculationSchedulingThread();
        void operatorReadyThread();
        void handleDataDescriptorsThread();
        DataDescriptor * validate_Array(GAFW::GeneralImplimentation::Array * array);
        void validate_Operator(GPUArrayOperator *arrayOperator);
        void validate_inputArray(GAFW::GeneralImplimentation::Array *array);
        void validate_bindedArray(GAFW::GeneralImplimentation::Array *array);
        DataDescriptor * validate_outputArray(GAFW::GeneralImplimentation::Array *array);
        size_t calculateArraySize(GAFW::GeneralImplimentation::ArrayDimensions &dim, GAFW::GeneralImplimentation::StoreType &type);
        
    public:
        GAFW::CalculationId calculate(GAFW::GeneralImplimentation::Result *r);
        virtual inline GPUDeviceDescriptor &getDeviceDescriptor(int device_no);
        void submitOperationReady(OperationDescriptor *);
        void submitDataDescriptorReview(DataDescriptor *);
        
        GPUEngine(GPUFactory * factory,GAFW::FactoryStatisticsOutput * statisticOutput);
        ~GPUEngine();
        
        
        
    
};

inline GPUDeviceDescriptor & GPUEngine::getDeviceDescriptor(int device_no)
{
        return *(new GAFW::GPU::GPUDeviceDescriptor(device_no));
}
}}
#endif	/* GPUENGINE2_H */

