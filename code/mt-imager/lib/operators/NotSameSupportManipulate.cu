/* NotSameSupportManipulate.cu:  CUDA implementation of the NotSameSupportManipulate operator 
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

#include "NotSameSupportManipulate.h"
#include "common.hcu"



namespace mtimager { namespace NotSameSupportManipulate_kernels
{
    __global__ void not_same_support_manipulate(
        int2 * in_blockDataIndx,
        int no_of_records,
        int * out_not_same_support
        );
    __global__ void not_same_support_manipulate(
            int2 * in_blockDataIndx,
            int no_of_records,
            int * out_not_same_support
            )
    {
        //out_not_same_support should have already been initialized.. we only need to change to1 s some 0
        __shared__ uint threadsPerRange;
        __shared__ int2 indexes[1025];
        __shared__ uint lastEntryIndex;

        if (threadIdx.x==0)
        {
            lastEntryIndex=in_blockDataIndx[no_of_records].x;
            uint lastEntry=in_blockDataIndx[lastEntryIndex].x;
            uint averageRecords=(lastEntry)/(lastEntryIndex+1);
            uint threadsPerRange_temp=(averageRecords/TOTALLOADS);

            //ensure that it is a power of 2
            uint powerof2;
            for (powerof2=1;threadsPerRange_temp>powerof2;powerof2<<=1);

            threadsPerRange=powerof2;
            if (threadsPerRange>blockDim.x) threadsPerRange=blockDim.x;
        }
        __syncthreads();
       // printf("%d\n",threadsPerRange);

        int thread0entry=(blockIdx.x*blockDim.x)/threadsPerRange;
        if (thread0entry>=lastEntryIndex) return; //useless block
        int entry=(blockIdx.x*blockDim.x)+threadIdx.x;
        int myEntry=entry/threadsPerRange;
        uint internalEntry=threadIdx.x/threadsPerRange;
        //uint myOffset=threadIdx.x%threadsPerRange*max_same_support;
        if ((entry%threadsPerRange)==0)
        {    if (myEntry<=lastEntryIndex)
            {
            //We need to load data in this case
                  indexes[internalEntry]=in_blockDataIndx[myEntry];
            }
            else
            {
                  indexes[internalEntry]=make_int2(0,0);  
            }

        }

        if (threadIdx.x==(blockDim.x-1))
        {
            if ((myEntry+1)<no_of_records)
                indexes[internalEntry+1]=in_blockDataIndx[myEntry+1];

        }
        __syncthreads();
        //Ok all data loaded....
        int2 begin=indexes[internalEntry];
        int2 end=indexes[internalEntry+1];
        int max_same_support;
        if (begin.y<16) max_same_support=TOTALLOADS_UNDER16;
        if (begin.y>16) max_same_support=TOTALLOADS;

        for (int to1ify=begin.x+(threadIdx.x%threadsPerRange)*max_same_support;to1ify<end.x;to1ify+=(max_same_support*threadsPerRange))
        {   
            out_not_same_support[to1ify]=1;  
        }
    }
}}

using namespace mtimager;
using namespace mtimager::NotSameSupportManipulate_kernels;

void NotSameSupportManipulate::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
       
     dim3 threadsPerBlock;
     dim3 blocks;
     
     int records=data.inputs[0].dim.getNoOfColumns();
     
     threadsPerBlock.x=1024;
     threadsPerBlock.y=1;
     threadsPerBlock.z=1;
     blocks.x=records/threadsPerBlock.x;
     blocks.x++;
     blocks.y=1;
     blocks.z=1;
      checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    

     cudaError_t err=cudaMemcpyAsync(data.outputs[0].pointer,data.inputs[0].pointer,data.inputs[0].dim.getNoOfColumns()*sizeof(int),cudaMemcpyDeviceToDevice,data.stream);
     if (err!=cudaSuccess)
          throw CudaException("Error while initializing output",err);
     not_same_support_manipulate<<<blocks,threadsPerBlock,0,data.stream>>>(
        (int2*)data.inputs[1].pointer,
             records,
        (int*)data.outputs[0].pointer
      );
      checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}
