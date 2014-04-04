/* ZeroPad.cu:  CUDA implementation of the ZeroPad operator 
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
#include "ZeroPad.h"
#include "cuda_runtime.h"
#include "cuComplex.h"

namespace GAFW { namespace GPU { namespace StandardOperators
{
    namespace ZeroPad_kernels
    {
        template<class Type>
        __global__ void zeropad(Type *input, int2 inputDim,Type *output, int2 outputDim );
        template <class A> 
       __device__ __inline__ A zero();
        template <>
       __device__ __inline__ cuComplex zero();
       template <>
       __device__ __inline__ cuDoubleComplex zero();
       
       template <class A> 
       __device__ __inline__ A zero()
       {
       return 0;
       }
       template <>
       __device__ __inline__ cuComplex zero()
       {
           return make_float2(0.0f,0.0f);
       }
       template <>
       __device__ __inline__ cuDoubleComplex zero()
       {
           return make_double2(0.0,0.0);
       }

        template<class Type>
        __global__ void zeropad(Type *input, int2 inputDim,Type *output, int2 outputDim )
        {

            int2 loc=make_int2(blockIdx.x * blockDim.x + threadIdx.x,blockIdx.y * blockDim.y + threadIdx.y);
            int2 coordinate;
            int2 inputNegativeEdge=make_int2(-inputDim.x/2,-inputDim.y/2);
            int2 inputPositiveEdge=make_int2(inputDim.x/2-inputDim.x%2,inputDim.y/2-inputDim.y%2);


            for (;loc.y<outputDim.y;loc.y+=gridDim.y*blockDim.y)
            {
                coordinate.y=loc.y-outputDim.y/2;

                for (;loc.x<outputDim.x;loc.x+=gridDim.x*blockDim.x)
                {
                    coordinate.x=loc.x-outputDim.x/2;
                    if ((coordinate.x<inputNegativeEdge.x)||(coordinate.x>inputPositiveEdge.x)||(coordinate.y<inputNegativeEdge.y)||(coordinate.y>inputPositiveEdge.y))
                        *(output+loc.y*outputDim.x+loc.x)=zero<Type>();
                    else
                        *(output+loc.y*outputDim.x+loc.x)=*(input+(coordinate.y+inputDim.y/2)*inputDim.x+(coordinate.x+inputDim.x/2));

                }
            }
        }

    }
}
}}

using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GPU::StandardOperators::ZeroPad_kernels;
using namespace GAFW::GeneralImplimentation;

void ZeroPad::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    ArrayDimensions &d=data.outputs[0].dim;
    ArrayDimensions &din=data.inputs[0].dim;
   
    int2 inputDim;
    int2 outputDim;
    inputDim.x=din.getNoOfColumns();
    inputDim.y=din.getNoOfRows();
    outputDim.x=d.getNoOfColumns();
    outputDim.y=d.getNoOfRows();
    
    
    //A function to put the array values all the a specific value exits but is not streamed
    // we use a kernel instead
    //We treat the array as a vector no_of_floats long
    dim3 threadsPerBlock;
    dim3 blocks;
    data.deviceDescriptor->getBestThreadsAndBlocksDim(outputDim.y,outputDim.y,blocks,threadsPerBlock);
    switch (data.inputs[0].type)
    {
        case real_float:
            zeropad<float> <<<blocks,threadsPerBlock,0,data.stream>>> ((float*)data.inputs[0].pointer,inputDim,(float*)data.outputs[0].pointer,outputDim);
            break;
        case real_double:
            zeropad<double> <<<blocks,threadsPerBlock,0,data.stream>>> ((double*)data.inputs[0].pointer,inputDim,(double*)data.outputs[0].pointer,outputDim);
            break;
        case real_int:
            zeropad<int> <<<blocks,threadsPerBlock,0,data.stream>>> ((int*)data.inputs[0].pointer,inputDim,(int*)data.outputs[0].pointer,outputDim);
            break;
        case real_uint:
            zeropad<uint> <<<blocks,threadsPerBlock,0,data.stream>>> ((uint*)data.inputs[0].pointer,inputDim,(uint*)data.outputs[0].pointer,outputDim);
            break;
        case real_longint:
            zeropad<long> <<<blocks,threadsPerBlock,0,data.stream>>> ((long*)data.inputs[0].pointer,inputDim,(long*)data.outputs[0].pointer,outputDim);
            break;
        case real_ulongint:
            zeropad<unsigned long> <<<blocks,threadsPerBlock,0,data.stream>>> ((unsigned long*)data.inputs[0].pointer,inputDim,(unsigned long*)data.outputs[0].pointer,outputDim);
            break;
        case real_shortint:
            zeropad<short> <<<blocks,threadsPerBlock,0,data.stream>>> ((short*)data.inputs[0].pointer,inputDim,(short*)data.outputs[0].pointer,outputDim);
            break;
        case real_ushortint:
            zeropad<unsigned short> <<<blocks,threadsPerBlock,0,data.stream>>> ((unsigned short*)data.inputs[0].pointer,inputDim,(unsigned short*)data.outputs[0].pointer,outputDim);
            break;
        case complex_double:
            zeropad<cuDoubleComplex> <<<blocks,threadsPerBlock,0,data.stream>>> ((cuDoubleComplex*)data.inputs[0].pointer,inputDim,(cuDoubleComplex*)data.outputs[0].pointer,outputDim);
            break;
        case complex_float:
            zeropad<cuComplex> <<<blocks,threadsPerBlock,0,data.stream>>> ((cuComplex*)data.inputs[0].pointer,inputDim,(cuComplex*)data.outputs[0].pointer,outputDim);
            break;
        default:
            throw GeneralException("I don't know what to do with type");
    }  
                    
}