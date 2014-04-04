/* GPUafw.h: Main include file for the GPU implementation of the GAFW. 
 * It includes all other related header files   
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
#ifndef __GPUAFW_H__
#define	__GPUAFW_H__
#include "gafw.h"
#include "gafw-impl.h"
#include <vector>
#include <cuda_runtime.h>
#include <queue>

#define checkCudaError(x,msg)  \
        {    cudaError_t _err_=x ; \
             if (_err_!=cudaSuccess) \
                throw CudaException(msg,_err_); \
        }


namespace GAFW { namespace GPU {
class GPUFactory;
class GPUDataStore;
class ProcessMaps;
class CudaEventPool;
class DataDescriptor;
class GPUEngine;

class CUDADeviceManager;
class OperationDescriptor;
template <class T> class RingBuffer;
template <class T, class Container = std::deque<T> > class SynchronousQueue;
template <class T> class ThreadEntry;
namespace Buffer
{
    enum BufferType
    {
        Buffer_Normal, //allocted and transferred to submission 
        Buffer_DeallocBeforeSubmit, //deallocated before submition but allocationm continue after submission
        Buffer_UnkownSize //buffer size is not known and hadled during submission.. defragmantation and etc will happen before
    };
};
class GPUArrayOperator;
class GPUDeviceDescriptor;
class GPUAFWCudaException;

class GPUSubmissionData;
typedef struct
{
    void * pointer;
    GAFW::ArrayDimensions dim;
    GAFW::GeneralImplimentation::StoreType type;
        
} GPUArrayDescriptor;

} }

#include "GPUSubmissionData.h"
#include "GPUCudaException.h"

#include "GPUDeviceDescriptor.h"
#include "GPUFactory.h"
#ifndef __CUDACC__
#include "GPUDataStore.h"
#endif
#ifndef __CUDACC__
#include "SynchronousObject.h"
#include "SynchronousQueue.h"
#include "CudaEventPool.h"
#include "RingBuffer.h"
#include "ThreadEntry.h"
#include "DataDescriptor.h"
#include "CUDADeviceManager.h"
#include "OperationDescriptor.h"
#include "GPUEngineOperatorStatistic.h"
#endif
#include "ValidationData.h"
#include "GPUEngine.h"

#include "GPUArrayOperator.h"
#endif	/* GPUMFW_H */

