/* ValueChangeDetect.cu:  CUDA implementation of the ValueChangeDetect operator 
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
#include "ValueChangeDetect.h"
#include "cuda_runtime.h"
#include "cuComplex.h"
namespace GAFW{ namespace GPU { namespace StandardOperators {
    namespace ValueChangeDetect_kernels
    {
        template<class T> 
        __global__ void changedetect(T* input,int* output,uint no_of_records)
        {
            int totalthreads=gridDim.x*blockDim.x;

            for(uint myEntry=blockIdx.x*blockDim.x+threadIdx.x;
                    myEntry<no_of_records;
                    myEntry+=totalthreads
                    )
            {
                if (myEntry==0) output[0]=1; //First entry is noew
                else output[myEntry]=(input[myEntry]!=input[myEntry-1]);
            }

        }


    }
}}}
using namespace GAFW::GPU::StandardOperators::ValueChangeDetect_kernels;
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;


void ValueChangeDetect::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    uint no_of_elements=data.inputs[0].dim.getTotalNoOfElements();
    dim3 threadsPerBlock;
    dim3 blocks;
    threadsPerBlock.x=1024;
    threadsPerBlock.y=1;
    threadsPerBlock.z=1;
    blocks.x=32;
    blocks.y=1;
    blocks.z=1;
    
    
    cudaEventRecord(*data.startEvent,data.stream);
    switch (data.inputs[0].type)
    {
        case real_int:
            changedetect<int> <<<blocks,threadsPerBlock,0,data.stream>>> ((int *)data.inputs[0].pointer,(int*)data.outputs[0].pointer,no_of_elements);
            break;
        case real_uint:
            changedetect<uint> <<<blocks,threadsPerBlock,0,data.stream>>> ((uint *)data.inputs[0].pointer,(int*)data.outputs[0].pointer,no_of_elements);      
            break;
        case real_float:
            changedetect<float> <<<blocks,threadsPerBlock,0,data.stream>>> ((float *)data.inputs[0].pointer,(int*)data.outputs[0].pointer,no_of_elements);
            break;
        default:
            throw GeneralException("Not yet implemented");
        
    }
    
}
