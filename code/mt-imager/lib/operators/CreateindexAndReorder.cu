/* CreateIndexAndReorder.cu:  CUDA implementation of the CreateIndexAndReOrder operator 
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


#include "CreateIndexAndReorder.h"

namespace mtimager { namespace CreateIndexAndReorder_kernels
{
      __global__ void create_index_and_reorder(
            int * in_gridBool,
            int * in_grid_accumulated,
            int * in_support,  
            int * in_convIndex, 
            int no_of_records,
            int * out_index,
            int * out_support,  
            int * out_convIndex 
            );
  
    __global__ void create_index_and_reorder(
            int * in_gridBool,
            int * in_grid_accumulated,
            int * in_support,  
            int * in_convIndex, 
            int no_of_records,
            int * out_index,
            int * out_support,  
            int * out_convIndex 
            )
    {

        int thread0Entry=blockIdx.x*blockDim.x;
        int myEntry=thread0Entry+threadIdx.x;

        if (myEntry<no_of_records)
        {
            if (in_gridBool[myEntry]==1)
            {
                int index=in_grid_accumulated[myEntry];
                out_index[index]=myEntry;
                out_support[index]=in_support[myEntry];
                out_convIndex[index]=in_convIndex[myEntry];
            }
        }

    }
}}
using namespace mtimager;
using namespace mtimager::CreateIndexAndReorder_kernels;
void CreateIndexAndReorder::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
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
     //Input s
        //int * in_gridBool,
        //int * in_grid_accumulated,
        //int * in_support,  
        //int * in_same_support, 
     //outputs
       // int * out_index,
       // int * out_support,  
       // int * out_same_support, 
     //initialize everything to 0
      
    

     for (int x=0;x<3;x++)
     {
         cudaError_t err=cudaMemsetAsync(data.outputs[x].pointer,0,records*sizeof(int),data.stream);
         if (err!=cudaSuccess)
             throw CudaException("Error with zerofying outputs",err);
     }
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
     create_index_and_reorder<<<blocks,threadsPerBlock,0,data.stream>>>(
        (int*)data.inputs[0].pointer,
        (int*)data.inputs[1].pointer,
        (int*)data.inputs[2].pointer,
        (int*)data.inputs[3].pointer,
             records,
        (int*)data.outputs[0].pointer,
        (int*)data.outputs[1].pointer,
        (int*)data.outputs[2].pointer
             );
      checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}
 