/* ConvFunctionReOrganize.cu:  CUDA implementation of the ConvFunctionReOrganize operator 
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

#include <cuComplex.h>
#include "ConvFunctionReOrganize.h"

namespace mtimager { namespace ConvFunctionReOrganize_kernels
{
    __global__ void conv_function_reorganize_odd_sampling(cuComplex *input,  cuComplex *output,int dim,int support, int sampling);
    __global__ void conv_function_reorganize_even_sampling(cuComplex *input,  cuComplex *output,int dim,int support, int sampling);
    


    __global__ void conv_function_reorganize_odd_sampling(cuComplex *input,  cuComplex *output,int dim,int support, int sampling)
    {
        //re-oraginaztion 
        int2 loc=make_int2(blockDim.x*blockIdx.x+ threadIdx.x,blockDim.y*blockIdx.y+ threadIdx.y);

        int cutoutborder=dim/2-((support/2)*sampling+sampling/2);

        if ((loc.x<support*sampling)&&(loc.y<support*sampling))
        {
            int2 inputloc=make_int2(loc.x+cutoutborder,loc.y+cutoutborder);
            int2 support_pos=make_int2(loc.x/sampling,loc.y/sampling);
            int2 sampling_pos=make_int2(loc.x%sampling,loc.y%sampling);
            int2 newloc=make_int2(sampling_pos.x*support*support,sampling_pos.y*support*support*sampling);
            int sup=support_pos.y*support+support_pos.x;
                *(output+newloc.y+newloc.x+sup)=*(input+inputloc.y*dim+inputloc.x);

        }
    }

    __global__ void conv_function_reorganize_even_sampling(cuComplex *input,  cuComplex *output,int dim,int support, int sampling)
    {
        //re-oraginaztion 

        int2 loc=make_int2(blockDim.x*blockIdx.x+ threadIdx.x,blockDim.y*blockIdx.y+ threadIdx.y);

        //To check if to do a +1

        int cutoutborder=dim/2-((support/2)*sampling+sampling/2);

        //printf (" %d %d",cutoutborder,dim);
        if ((loc.x<(support)*(sampling+1))&&(loc.y<(support)*(sampling+1)))
        {
            int2 support_pos=make_int2(loc.x/(sampling+1),loc.y/(sampling+1));
            int2 sampling_pos=make_int2(loc.x%(sampling+1),loc.y%(sampling+1));
            loc.x-=support_pos.x;
            loc.y-=support_pos.y; 
            int2 inputloc=make_int2(loc.x+cutoutborder,loc.y+cutoutborder);
             int2 newloc=make_int2(sampling_pos.x*support*support,sampling_pos.y*support*support*(sampling+1));
            int sup=support_pos.y*support+support_pos.x;
            *(output+newloc.y+newloc.x+sup)=*(input+inputloc.y*dim+inputloc.x);
        }
    }
}}

using namespace mtimager;
using namespace mtimager::ConvFunctionReOrganize_kernels;

void ConvFunctionReOrganize::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     
    dim3 threadsPerBlock;
    dim3 blocks;
    int support=data.params.getIntProperty("ConvolutionFunction.support");
    int sampling=data.params.getIntProperty("ConvolutionFunction.sampling");
    checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    if (sampling%2)
    {
        
        data.deviceDescriptor->getBestThreadsAndBlocksDim(support*sampling,support*sampling,blocks,threadsPerBlock);
        int oldsupport=data.inputs[0].dim.getX()/sampling;
        conv_function_reorganize_odd_sampling <<<blocks,threadsPerBlock,0,data.stream>>> ((cuComplex*)data.inputs[0].pointer,(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getNoOfColumns(),data.params.getIntProperty("ConvolutionFunction.support"),data.params.getIntProperty("ConvolutionFunction.sampling")); 
    }
    else
    {
        data.deviceDescriptor->getBestThreadsAndBlocksDim(support*(sampling+1),support*(sampling+1),blocks,threadsPerBlock);
    
        int oldsupport=(data.inputs[0].dim.getX()-1)/sampling;
        conv_function_reorganize_even_sampling <<<blocks,threadsPerBlock,0,data.stream>>> ((cuComplex*)data.inputs[0].pointer,(cuComplex*)data.outputs[0].pointer,data.inputs[0].dim.getNoOfColumns(),data.params.getIntProperty("ConvolutionFunction.support"),data.params.getIntProperty("ConvolutionFunction.sampling")); 
    }
     
    checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;

}  
