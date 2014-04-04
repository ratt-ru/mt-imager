/* SingleMultiply.cu:  CUDA implementation of the SingleMultiply operator 
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
#include "SingleMultiply.h"
#include "cuda_runtime.h"
#include "cuComplex.h"


namespace GAFW { namespace GPU { namespace StandardOperators
{   namespace SingleMultiply_kernels
{
    __device__ __inline__ cuComplex multiply(cuComplex value1,cuComplex value2);
    __device__ __inline__ cuComplex multiply(cuComplex value1,float value2);
    __device__ __inline__ cuDoubleComplex multiply(cuDoubleComplex value1,double value2);
    __device__ __inline__ cuComplex multiply(float value1,cuComplex value2);
    __device__ __inline__ float multiply(float value1,float value2);
    __device__ __inline__ cuDoubleComplex multiply(double value1,cuDoubleComplex value2);
    template <class InputType,class MultiplierType,class OutputType>
    __global__ void single_multiply(InputType * input,MultiplierType * multiplier, OutputType *output,int elements);
    template <class InputType,class MultiplierType,class OutputType>
    __global__ void single_multiply(InputType * input,MultiplierType multiplier_value, OutputType *output,int elements);

    __device__ __inline__ cuComplex multiply(cuComplex value1,cuComplex value2)
    {
        return make_float2(value1.x*value2.x-value1.y*value2.y,value1.x*value2.y+value1.y*value2.x);
    }
    __device__ __inline__ cuComplex multiply(cuComplex value1,float value2)
    {
        return make_float2(value1.x*value2,value1.y*value2);
    }
    __device__ __inline__ cuDoubleComplex multiply(cuDoubleComplex value1,double value2)
    {
        return make_double2(value1.x*value2,value1.y*value2);
    }


    __device__ __inline__ cuComplex multiply(float value1,cuComplex value2)
    {
        return make_float2(value2.x*value1,value2.y*value1);
    }
    __device__ __inline__ float multiply(float value1,float value2)
    {
        return value1*value2;
    }
    __device__ __inline__ cuDoubleComplex multiply(double value1,cuDoubleComplex value2)
    {
        return make_double2(value2.x*value1,value2.y*value1);
    }

    template <class InputType,class MultiplierType,class OutputType>
    __global__ void single_multiply(InputType * input,MultiplierType * multiplier, OutputType *output,int elements)
    {
        int idx=blockIdx.x * blockDim.x + threadIdx.x;
       MultiplierType multiplier_value=*multiplier;

        for (int elno=idx;elno<elements;elno+=gridDim.x*blockDim.x)
        {
            InputType input_value=*(input+elno);
            *(output+elno)=multiply(input_value,multiplier_value);
        }

    }

    template <class InputType,class MultiplierType,class OutputType>
    __global__ void single_multiply(InputType * input,MultiplierType multiplier_value, OutputType *output,int elements)
    {
        int idx=blockIdx.x * blockDim.x + threadIdx.x;

        for (int elno=idx;elno<elements;elno+=gridDim.x*blockDim.x)
        {
            InputType input_value=*(input+elno);
            *(output+elno)=multiply(input_value,multiplier_value);
        }

    }
}}}}

using namespace GAFW::GPU::StandardOperators::SingleMultiply_kernels;
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;

#define complex_to_cuComplex(value) make_float2(value.real(),value.imag())

void SingleMultiply::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{

     dim3 threadsPerBlock;
     dim3 blocks;
     //data.deviceDescriptor->getBestThreadsAndBlocksDim(1,data.inputs[0].dim.getTotalNoOfElements(),blocks,threadsPerBlock);
     threadsPerBlock.x=1024;
     threadsPerBlock.y=1;
     threadsPerBlock.z=1;
     blocks.y=1;
     blocks.x=32;
     blocks.z=1;
     if (data.noOfInputs==1)
     {
         if (data.params.getPropertyType("multiplier.value")==GAFW::Tools::CppProperties::Properties::Complex)
         {
             if(data.inputs[0].type==complex_float)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer,complex_to_cuComplex(data.params.getComplexProperty("multiplier.value")),(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
             else
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((float*)data.inputs[0].pointer,complex_to_cuComplex(data.params.getComplexProperty("multiplier.value")),(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
                 
             }
         }
         else
         {
             if(data.inputs[0].type==complex_float)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer,data.params.getFloatProperty("multiplier.value"),(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
                 
             }
             else if (data.inputs[0].type==complex_double)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((cuDoubleComplex*)data.inputs[0].pointer,(double)data.params.getFloatProperty("multiplier.value"),(cuDoubleComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
             else
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((float*)data.inputs[0].pointer,data.params.getFloatProperty("multiplier.value"),(float*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
                 
             }
         }
     }
     else
     {
         if (data.inputs[1].type==complex_float)
         {
             if(data.inputs[0].type==complex_float)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer,(cuComplex*)data.inputs[1].pointer,(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
             else
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((float*)data.inputs[0].pointer,(cuComplex*)data.inputs[1].pointer,(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
         }
         
         else if(data.inputs[1].type==complex_double)
         {
             if(data.inputs[0].type==real_double)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((double*)data.inputs[0].pointer,(cuDoubleComplex*)data.inputs[1].pointer,(cuDoubleComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
             else
                 throw GeneralException("Unsupported");
         }
         
         else
         {
             if(data.inputs[0].type==complex_float)
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer,(float*)data.inputs[1].pointer,(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
             else
             {
                single_multiply<<<blocks,threadsPerBlock,0,data.stream>>>((float*)data.inputs[0].pointer,(float*)data.inputs[1].pointer,(float*)data.outputs[0].pointer,data.inputs[0].dim.getTotalNoOfElements());
             }
         }
     }
         

}

