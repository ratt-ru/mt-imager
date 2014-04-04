/* ConvFunctionSupportFind.cu:  CUDA implementation of the ConvFunctionSupportFind operator 
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

#include "ConvFunctionSupportFind.h"
#include "cuComplex.h"

namespace mtimager { namespace ConvFunctionSupportFind_kernels
{
    template <class  T,class T2>
    __global__ void conv_function_support_find(T *input,float *output,int sampling,int dim,T2 takezeroratio);
    
    template <class  T,class T2>
    __global__ void conv_function_support_find(T *input,float *output,int sampling,int dim,T2 takezeroratio)
    {

        //zero assumed at centre.. ie dim/2  and a square convction
        //dim is always odd
        __shared__ int lastnonzeroloc;
        if ((threadIdx.x==0)&&(threadIdx.y==0)) lastnonzeroloc=-1;
        __syncthreads();
        int plane=(int)blockIdx.x;
        int planepos=plane*dim*dim;

        //WE need to get the value of the centre for comparison
        T centre =*(input+(dim/2)*dim+(dim/2));
        T2 takezero=takezeroratio; //(centre.x*centre.x+centre.y*centre.y)*takezeroratio;
        for (int loc=dim-threadIdx.x-1;loc>=dim/2-threadIdx.x;loc-=blockDim.x)
        {
            if (loc>=dim/2)
            {
                int pos;
                if (threadIdx.y==0) pos=planepos+(dim/2)*dim+loc;
                if (threadIdx.y==1) pos=planepos+loc*dim+dim/2;
                T2 magnitudepow2=(input+pos)->x*(input+pos)->x+(input+pos)->y*(input+pos)->y;
                if (magnitudepow2>takezero) //to be changed
                {   
                    int lastnonzeroloc_local=lastnonzeroloc;
                    while (lastnonzeroloc_local<loc)
                    {
                        atomicCAS(&lastnonzeroloc,lastnonzeroloc_local,loc);
                        lastnonzeroloc_local=lastnonzeroloc;
                    }
                }
            }//Not usre if I have to put  a threadfence here:(
            __syncthreads();
            if (lastnonzeroloc>-1) break;

        }
        if ((threadIdx.x==0)&&(threadIdx.y==0))
        {
            //Ok just need to calculate support
            if (lastnonzeroloc>-1)
                    *(output+plane)=((lastnonzeroloc-dim/2+(sampling+1)/2)/sampling)*2+1;
                    //To review sampling + 1 thingy////

            else
                    *(output+plane)=0;
        }
        //*(output+plane)=lastnonzeroloc;

    }
}}
using namespace mtimager;
using namespace mtimager::ConvFunctionSupportFind_kernels;

void ConvFunctionSupportFind::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     
    dim3 threadsPerBlock;
    dim3 blocks;
  //  data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
    int sampling=data.params.getIntProperty("ConvolutionFunction.sampling");
    threadsPerBlock.x=512;
    threadsPerBlock.y=2;
     threadsPerBlock.z=1;
     blocks.x=(data.inputs[0].dim.getNoOfDimensions()==2)?1:data.inputs[0].dim.getZ();
     blocks.y=1;
     blocks.z=1;
      checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    
    if (data.inputs[0].type==GAFW::GeneralImplimentation::complex_float)
    {
        conv_function_support_find<cuComplex,float> <<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer,(float*)data.outputs[0].pointer,sampling,data.inputs[0].dim.getNoOfColumns(),pow(data.params.getFloatProperty("ConvolutionFunction.takeaszero"),2));
    }
    else
        conv_function_support_find<cuDoubleComplex,double> <<<blocks,threadsPerBlock,0,data.stream>>>((cuDoubleComplex*)data.inputs[0].pointer,(float*)data.outputs[0].pointer,sampling,data.inputs[0].dim.getNoOfColumns(),pow((double)data.params.getFloatProperty("ConvolutionFunction.takeaszero"),2));
    //else
       // throw std::exception();
    checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;

}
