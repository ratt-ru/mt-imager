/* SincCorrector.cu:  CUDA implementation of the SincCorrector operator 
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

#include "SincCorrector.h"
#include "common.hcu"
#include "mtimager.h"
#include <cuComplex.h>

namespace mtimager { namespace SincCorrector_kernels
{
    __global__ void sinc_correction(float *input, float *output,int sampling,int planes, int rows, int cols);
    
    __global__ void sinc_correction(float *input, float *output,int sampling,int planes, int rows, int cols)
    {
        int i=blockIdx.x * blockDim.x + threadIdx.x;
        int j=blockIdx.y * blockDim.y + threadIdx.y;
        int el=(j*cols) + i;

        //float * ans_el=ans+el;
        for (int plane=0;plane<planes;plane++)
        {
            float * input_el=input+plane*rows*cols+el;
            float * output_el=output+plane*rows*cols+el;
            if (i<cols && j<rows)  
            {    
                //x represents columns, y rows
                float2 offset=make_float2(float(i-(cols/2)),float(j-(rows/2)));
                float2 x;

                x.x=float(offset.x)/(float(cols)*(float)sampling);
                x.y=float(offset.y)/(float(rows)*(float)sampling);
                float2 invsinc;
                invsinc.x=3.14159265359*x.x/sinpif(x.x);
                invsinc.y=3.14159265359*x.y/sinpif(x.y);
                if (x.x==0.0f) invsinc.x=1.0f;
                if (x.y==0.0f) invsinc.y=1.0f;
                *(output_el)=*(input_el)*(invsinc.x*invsinc.y);

            }
        }

    }
}}
using namespace mtimager;
using namespace mtimager::SincCorrector_kernels;

void SincCorrector::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    dim3 threadsPerBlock;
    dim3 blocks;
    int sampling=data.params.getIntProperty("sampling");
    int planes=data.inputs->dim.getTotalNoOfElements()/(data.inputs->dim.getNoOfRows()*data.inputs->dim.getNoOfColumns());
    data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
    sinc_correction <<<blocks,threadsPerBlock,0,data.stream>>> ((float* )data.inputs->pointer,(float* )data.outputs->pointer,sampling,planes,data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns()); 
}







