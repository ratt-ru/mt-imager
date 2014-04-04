/* Zeros.cu:  CUDA implementation of the Zeros operator 
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
#include "Zeros.h"
#include "cuda_runtime.h"
#include "cuComplex.h" 
namespace GAFW { namespace GPU { namespace StandardOperators
{
    namespace Zeros_kernels
    {
        __global__ void zeros(int no_of_elements, float *ans)
        {
           int i=blockIdx.x * blockDim.x + threadIdx.x;
           int totalthreads=gridDim.x*blockDim.x;

           for (;i<no_of_elements;i+=totalthreads)
           {
             float * ans_i=ans+i;
                *ans_i=0.0;

           }

        }
    }
}}}

using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GPU::StandardOperators::Zeros_kernels;
using namespace GAFW::GeneralImplimentation;

    
void Zeros::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    int no_of_elements=data.outputs[0].dim.getTotalNoOfElements();
    int no_of_floats;
    switch (data.outputs[0].type)
    {
        case real_float:
            no_of_floats=no_of_elements;
            break;
        case complex_float:
            no_of_floats=no_of_elements*2;
            break;
        default:
            throw GeneralException("Unknown type")
    }
    
    //A function to put the array values all the a specific value exits but is not streamed
    // we use a kernel instead
    //We treat the array as a vector no_of_floats long
    dim3 threadsPerBlock;
    dim3 blocks;
    //data.deviceDescriptor->getBestThreadsAndBlocksDim(1,no_of_floats,blocks,threadsPerBlock);
    threadsPerBlock.x=1024;
    threadsPerBlock.y=1;
    threadsPerBlock.z=1;
    blocks.x=32;
    blocks.y=1;
    blocks.z=1;
    
    zeros <<<blocks,threadsPerBlock,0,data.stream>>> (no_of_floats,((float*)data.outputs[0].pointer));
}


