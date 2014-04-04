/* CreateBlockIndex.cu:  CUDA implementation of the CreateBlockIndex operator 
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


#include "CreateBlockIndex.h"

namespace mtimager { namespace CreateBlockIndex_kernels
{
    __global__ void create_blockDataIndx_with_support(
        int * not_same_support,
        int * not_same_support_accumulated,
        int * support,
        int no_of_records,
        int2  * out_blockDataIndex
        );
    __global__ void create_blockDataIndx_without_support(
        int * not_same_support,
        int * not_same_support_accumulated,
        int * support,
        int no_of_records,
        int  * out_blockDataIndex
        );

    __global__ void create_blockDataIndx_with_support(
            int * not_same_support,
            int * not_same_support_accumulated,
            int * support,
            int no_of_records,
            int2  * out_blockDataIndex
            )
    {   
        int thread0Entry=blockIdx.x*blockDim.x;
        int myEntry=thread0Entry+threadIdx.x;
        if (myEntry<no_of_records)
        {
            if (not_same_support[myEntry]==1)
            {
                int index=not_same_support_accumulated[myEntry];
                int newsupport=support[myEntry];
                out_blockDataIndex[index].x=myEntry;
                out_blockDataIndex[index].y=newsupport;

            }
        }
        else if (myEntry==no_of_records)
        {   int index=not_same_support_accumulated[no_of_records-1];
            out_blockDataIndex[no_of_records].x=not_same_support_accumulated[no_of_records-1];
            if (support[no_of_records-1]!=0)
            {    out_blockDataIndex[index].x=no_of_records;
                 out_blockDataIndex[index].y=0;
            }
       } 
    }
    __global__ void create_blockDataIndx_without_support(
            int * not_same_support,
            int * not_same_support_accumulated,
            int * support,
            int no_of_records,
            int  * out_blockDataIndex
            )
    {   
        int thread0Entry=blockIdx.x*blockDim.x;
        int myEntry=thread0Entry+threadIdx.x;
        if (myEntry<no_of_records)
        {
            if (not_same_support[myEntry]==1)
            {
                int index=not_same_support_accumulated[myEntry];
                //int newsupport=support[myEntry];
                out_blockDataIndex[index]=myEntry;

            }
        }
        else if (myEntry==no_of_records)
        {   
            int index=not_same_support_accumulated[no_of_records-1];
            out_blockDataIndex[no_of_records]=index;
            if (support[no_of_records-1]!=0)
                out_blockDataIndex[index]=no_of_records;

        }

    }
}}
using namespace mtimager;
using namespace mtimager::CreateBlockIndex_kernels;

void CreateBlockIndex::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
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
     //Inputs 
     //int * not_same_support,
      //  int * not_same_support_accumulated,
       // int no_of_records,
      //outputs
       //      int * out_blockDataIndex
     
      checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    

     if (params.getBoolProperty("with_support")==true)
     {
     cudaError_t err=cudaMemsetAsync(data.outputs[0].pointer,0,(records+1)*sizeof(int)*2,data.stream);
         if (err!=cudaSuccess)
             throw CudaException("Error with zerofying outputs",err);  
     create_blockDataIndx_with_support<<<blocks,threadsPerBlock,0,data.stream>>>(
        (int*)data.inputs[0].pointer,
        (int*)data.inputs[1].pointer,
             (int*)data.inputs[2].pointer,
             records,
        (int2*)data.outputs[0].pointer
      );
     }
     else
     {
        cudaError_t err=cudaMemsetAsync(data.outputs[0].pointer,0,(records+1)*sizeof(int),data.stream);
         if (err!=cudaSuccess)
             throw CudaException("Error with zerofying outputs",err);  
     create_blockDataIndx_without_support<<<blocks,threadsPerBlock,0,data.stream>>>(
        (int*)data.inputs[0].pointer,
        (int*)data.inputs[1].pointer,
        (int*)data.inputs[2].pointer,
        records,
        (int*)data.outputs[0].pointer
      );
     
     }
      checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}