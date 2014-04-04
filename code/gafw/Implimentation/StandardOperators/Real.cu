/* Real.cu:  CUDA implementation of the Real operator 
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
#include "Real.h"
#include "cuda_runtime.h"
#include "cuComplex.h"
namespace GAFW { namespace GPU { namespace StandardOperators
{   namespace Real_kernels
    {
        __global__ void real(cuComplex *input,float *output,int no_of_elements);

        __global__ void real(cuComplex *input,float *output,int no_of_elements)
        {
           int i=blockIdx.x * blockDim.x + threadIdx.x;
           int totalthreads=gridDim.x*blockDim.x;

           for (;i<no_of_elements;i+=totalthreads)
           {
                cuComplex * input_i=input+i;
                float * output_i=output+i;
                *output_i=input_i->x;
           }     

        }

    }
}}}
using namespace GAFW::GPU::StandardOperators::Real_kernels;
using namespace GAFW::GPU::StandardOperators;

void Real::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    int no_of_elements=data.outputs[0].dim.getTotalNoOfElements();
    dim3 threadsPerBlock;
    dim3 blocks;
    
    threadsPerBlock.x=1024;
    threadsPerBlock.y=1;
    threadsPerBlock.z=1;
    blocks.x=32;
    blocks.y=1;
    blocks.z=1;
    real <<<blocks,threadsPerBlock,0,data.stream>>> ((cuComplex *)data.inputs[0].pointer,(float*)data.outputs[0].pointer,no_of_elements);
}

