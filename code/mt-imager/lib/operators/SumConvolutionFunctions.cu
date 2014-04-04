/* SumConvolutionFunctions.cu:  CUDA implementation of the SumConvolutionFunctions operator 
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


#include "SumConvolutionFunctions.h"
#include "common.hcu"
#include "mtimager.h"
#include <cuComplex.h>

namespace mtimager { namespace SumConvolutionFunctions_kernels
{
    __global__  void ConvSumFind(cuComplex * in_convData,int in_convInfo[][2], int no_of_Planes, int convFunctionsPerPlane, float * out_convSum);
    __global__  void ConvSumFind(cuComplex * in_convData,int in_convInfo[][2], int no_of_Planes, int convFunctionsPerPlane, float * out_convSum)
    {


        //Each thread calculates one value
        int id=blockIdx.x*blockDim.x+threadIdx.x;
        int total_conv_functions=no_of_Planes*convFunctionsPerPlane;
        for (int convId=id;convId<total_conv_functions;convId+=gridDim.x*blockDim.x)
        {
            int plane=convId/convFunctionsPerPlane;
            int particular_id=convId%convFunctionsPerPlane;

            int support=in_convInfo[plane][0];
            int beginIndx=in_convInfo[plane][1];
            beginIndx+=particular_id*support*support;
            int endIndx=beginIndx+support*support; //position described not included
            float value=0.0;
            for (int currentIndx=beginIndx;currentIndx<endIndx;currentIndx++)
            {
                value+=in_convData[currentIndx].x;
            }
            //if (plane==0) printf ("particular_id %d value %f \n",particular_id,value);

            //value calculated
            //Now store
            out_convSum[plane*convFunctionsPerPlane+particular_id]=value;
        }
    }
}}
using namespace mtimager;
using namespace mtimager::SumConvolutionFunctions_kernels;

void SumConvolutionFunctions::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
       
     dim3 threadsPerBlock;
     dim3 blocks;
    
     int totalConvFunctions=data.inputs[0].dim.getY()*data.outputs[0].dim.getX();
     threadsPerBlock.x=1024;
     threadsPerBlock.y=1;
     threadsPerBlock.z=1;
     blocks.x=totalConvFunctions/threadsPerBlock.x+1;
     blocks.y=1;
     blocks.z=1;
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
       
        ConvSumFind<<<blocks,threadsPerBlock,0,data.stream>>>
                ((cuComplex*)data.inputs[1].pointer,
                (int(*)[2])data.inputs[0].pointer, 
                data.inputs[0].dim.getY(),
                data.outputs[0].dim.getX(),
                (float *) data.outputs[0].pointer);
        checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}

    
