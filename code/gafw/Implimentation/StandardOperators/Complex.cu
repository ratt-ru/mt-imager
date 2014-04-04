/* Complex.cu:  CUDA implementation of the Complex operator 
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
#include "Complex.h"
#include "cuda_runtime.h"
#include "cuComplex.h"

namespace GAFW { namespace GPU { namespace StandardOperators  {  namespace Complex_kernels { 
    
    __global__ void complex(float *input,cuComplex *output,int no_of_elements);
    __global__ void complex(double *input,cuDoubleComplex *output,int no_of_elements);
    


    __global__ void complex(float *input,cuComplex *output,int no_of_elements)
    {
       int i=blockIdx.x * blockDim.x + threadIdx.x;

       float * input_i=input+i;
       cuComplex * output_i=output+i;
        if (i<no_of_elements)  
        {    
            *output_i=make_float2(*input_i,0.0f);

        }
    }
    __global__ void complex(double *input,cuDoubleComplex *output,int no_of_elements)
    {
       int i=blockIdx.x * blockDim.x + threadIdx.x;

       double * input_i=input+i;
       cuDoubleComplex * output_i=output+i;
        if (i<no_of_elements)  
        {    
            *output_i=make_double2(*input_i,0.0f);

        }
    }
    }
}}}

using namespace GAFW::GeneralImplimentation;
using namespace GAFW::GPU::StandardOperators::Complex_kernels;
void GAFW::GPU::StandardOperators::Complex::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    int no_of_elements=data.outputs[0].dim.getTotalNoOfElements();
    
    //A function to put the array values all the a specific value exits but is not streamed
    // we use a kernel instead
    //We treat the array as a vector no_of_floats long
    dim3 threadsPerBlock;
    dim3 blocks;
    data.deviceDescriptor->getBestThreadsAndBlocksDim(1,no_of_elements,blocks,threadsPerBlock);
    if (data.inputs[0].type==real_float)
    complex <<<blocks,threadsPerBlock,0,data.stream>>> ((float *)data.inputs[0].pointer,(cuComplex*)data.outputs[0].pointer,no_of_elements);
    else
    complex <<<blocks,threadsPerBlock,0,data.stream>>> ((double *)data.inputs[0].pointer,(cuDoubleComplex*)data.outputs[0].pointer,no_of_elements);
        
}
