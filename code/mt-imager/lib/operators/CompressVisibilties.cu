/* CompressVisibilities.cu:  CUDA implementation of the CompressVisibilities operator 
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

#include "CompressVisibilties.h"
#include <cuComplex.h>
#include "common.hcu"
namespace mtimager { namespace CompressVisibilities_kernels
{
    template <class PolVisType,class PolFloatType,class PolFlagsType> 
    __global__ void compressVisibilities(
        int * in_dataIndx,
        int * in_compressIndx,
        PolVisType * in_visibilities,
        float * img_mult,
        PolFloatType *in_weights,
        PolFlagsType * in_flags,
        int no_of_records,
        PolVisType * out_visibilities
        );




    template <class PolVisType,class PolFloatType,class PolFlagsType> 
    __global__ void compressVisibilities(
            int * in_dataIndx,
            int * in_compressIndx,
            PolVisType * in_visibilities,
            float * img_mult,
            PolFloatType *in_weights,
            PolFlagsType * in_flags,
            int no_of_records,
            PolVisType * out_visibilities
            )
    {   
        //For now I am keeping things simple but there needs to be better handling 
        // for when compression is too much
        int lastEntry;
        lastEntry=in_compressIndx[no_of_records];
        int myEntry=blockDim.x*blockIdx.x+threadIdx.x;
        if (myEntry>=lastEntry) return; //useless
        int begin_entry=in_dataIndx[myEntry]; //two floats4
        int end_entry=in_compressIndx[myEntry];
        PolVisType vis;
        zerofyvis(vis);
        //if (end_entry-begin_entry > 100) printf ("%d %d\n",begin_entry,end_entry);
        for (int addEntry=begin_entry;addEntry<=end_entry;addEntry++)
        {
            PolVisType toAddVis=in_visibilities[addEntry];
            PolFloatType myWeights=in_weights[addEntry];
            float myImg_mult=img_mult[addEntry];
            PolFloatType myFlags;
            tofloat(in_flags[addEntry],myFlags);
            visadd(vis,toAddVis,myImg_mult,myWeights,myFlags);
        }
        out_visibilities[myEntry]=vis;

    }
}}
using namespace mtimager;
using namespace mtimager::CompressVisibilities_kernels;
    
template <class PolVisType,class PolWeightType,class PolFlagsType>
void CompressVisibilities::kernel_launch_wrapper(dim3 &blocks,dim3 &threadsPerBlock,GAFW::GPU::GPUSubmissionData &data,int &records)
{
    checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
        
    compressVisibilities<PolVisType,PolWeightType,PolFlagsType><<<blocks,threadsPerBlock,0,data.stream>>>(
           (int*)data.inputs[0].pointer,
           (int*)data.inputs[1].pointer,
           (PolVisType*) data.inputs[2].pointer,
           (float*)data.inputs[3].pointer,
           (PolWeightType *) data.inputs[4].pointer,
           (PolFlagsType *) data.inputs[5].pointer,
            records,
           (PolVisType*)data.outputs[0].pointer
           );
     
    checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;

}

void CompressVisibilities::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
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
     
     
     switch (data.inputs[4].dim.getNoOfColumns()) //NoOfpolarizations
     {
         case 1:
             this->kernel_launch_wrapper<pol1vis_type,pol1float_type, pol1flags_type>
                     (blocks,threadsPerBlock,data,records);
             break;
         case 2:
             this->kernel_launch_wrapper<pol2vis_type,pol2float_type, pol2flags_type>
                     (blocks,threadsPerBlock,data,records);
             break;
         case 4:
             this->kernel_launch_wrapper<pol4vis_type,pol4float_type, pol4flags_type>
                     (blocks,threadsPerBlock,data,records);
             break;
             
         default:
             throw GAFW::GeneralException("BUG: Unknown Polarisation");
                    
           
     }
}