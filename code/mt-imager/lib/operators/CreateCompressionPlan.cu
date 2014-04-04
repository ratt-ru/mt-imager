/* CreateCompressionPlan.cu:  CUDA implementation of the CreateCompressionPlan operator 
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

#include "CreateCompressionPlan.h"

namespace mtimager { namespace CreateCompressionPlan_kernels
{
    __global__ void create_compression_plan(
    int * in_accumulated_grid,
    int * in_toCompress,
    int no_of_records,
    int  * out_compressTo
    );


    __global__ void create_compression_plan(
        int * in_accumulated_grid,
        int * in_toCompress,
        int no_of_records,
        int  * out_compressTo
        )
    {   
        int thread0Entry=blockIdx.x*blockDim.x;
        int myEntry=thread0Entry+threadIdx.x;
        if (myEntry<(no_of_records-1))
        {
            if ((in_toCompress[myEntry]==1)&&(in_toCompress[myEntry+1]==0))
            {
                int index=in_accumulated_grid[myEntry]-1;
                //int newsupport=support[myEntry];
                out_compressTo[index]=myEntry;

            }
        }
        else if (myEntry==(no_of_records-1))
        {   
            if (in_toCompress[myEntry]==1)
            {
                int index=in_accumulated_grid[myEntry]-1;
                //int newsupport=support[myEntry];
                out_compressTo[index]=myEntry;
            }

        }
        else if (myEntry==no_of_records)
        {
            out_compressTo[no_of_records]=in_accumulated_grid[no_of_records-1]+1; 
            //In reality we do not know if there is an extra record

        }
    }

//end of namespace
}}
using namespace mtimager;
using namespace mtimager::CreateCompressionPlan_kernels;

void CreateCompressionPlan::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
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
     //int * gridIndex,
      //int * accumulated_grid,
       // int *to_compress,
      //outputs
       //      int * compressTo
     
      checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    

        cudaError_t err=cudaMemcpyAsync(data.outputs[0].pointer,
             data.inputs[0].pointer,
             records*sizeof(int),
             cudaMemcpyDeviceToDevice,data.stream);
         if (err!=cudaSuccess)
             throw CudaException("Error with copying",err);  

        create_compression_plan<<<blocks,threadsPerBlock,0,data.stream>>>(
                (int*)data.inputs[1].pointer,
                (int*)data.inputs[2].pointer,
                records,
                (int*)data.outputs[0].pointer
                );
        checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}



