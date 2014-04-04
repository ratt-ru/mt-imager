/* PreIFFTUVCorrection.cu:  CUDA implementation of the PreIFFTUVCorrection operator 
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

#include "PreIFFTUVCorrection.h"
#include "common.hcu"
#include "mtimager.h"
#include <cuComplex.h>
namespace mtimager { namespace PreIFFTUVCorrection_kernels
{
    __global__ void preifftUVCorrection(cuComplex *input,  cuComplex *output,int3 dim);

    __global__ void preifftUVCorrection(cuComplex *input,  cuComplex *output,int3 dim)
    {
        //Two corrections...
        // 1) insert conjugate of visibilities
        // 2) Put origin at edge as required by FFT
        // 3) re-order polarizations
       //CATERING ONLY FOR ODD FOR NOW 
        //int offset[2]={cols/2,rows/2};

        //get the points that I will take care of

        //IMPORTANT
        //Due to how polarizations are arranged in input dim.z= rows, dim.y=cols,dim.x=pols 
        //Hopefully the below declarations makes the code clearer
        int pols=dim.x;
        int rows=dim.z;
        int cols=dim.y;

        int3 outputdim=make_int3(dim.y,dim.z,dim.x); //The dim of output.... Polarization now is z


        int3 loc; ///Same argument as above


        loc.y=blockDim.x*blockIdx.x+ threadIdx.x;
        loc.z=blockDim.y*blockIdx.y+ threadIdx.y;
        //loc.x is not important for now
        //int2 sub=make_int2(cols%2-1,rows%2-1);
        //int2 otherloc=make_int2((-loc.x + 2* offset[0]+sub[0]), (-loc[1] + 2*offset[1]+sub[1]));

        if (loc.z>dim.z/2) return; //this need to change
        if (loc.y>=dim.y) return;


        int3 otherloc;

        otherloc.y=dim.y-loc.y-dim.y%2;
        otherloc.z=dim.z-loc.z-dim.z%2;
        if (otherloc.y==dim.y) otherloc.y=0;  //happens when dim is even since position 0 does not have "a mirror"
        if (otherloc.z==dim.z) otherloc.z=0;




        for (loc.x=0,otherloc.x=0;loc.x<dim.x;loc.x++,otherloc.x=loc.x)
        {

            int3 newloc;
            newloc.z=loc.x; //polarization
            newloc.y=OriginToCorner1D(loc.z, dim.z);
            newloc.x=OriginToCorner1D(loc.y, dim.y);

            int3 newotherloc;
            newotherloc.z=otherloc.x; //polarization
            newotherloc.y=OriginToCorner1D(otherloc.z, dim.z);
            newotherloc.x=OriginToCorner1D(otherloc.y, dim.y);

            cuComplex first=*(input+loc_2_pos3D(loc,dim));
            cuComplex second=*(input+loc_2_pos3D(otherloc,dim));
            *(output+loc_2_pos3D(newloc,outputdim))=first;
            *(output+loc_2_pos3D(newotherloc,outputdim))=second;
        }
    }
}}
    
using namespace mtimager;
using namespace mtimager::PreIFFTUVCorrection_kernels;
void PreIFFTUVCorrection::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     
    dim3 threadsPerBlock;
    dim3 blocks;
    data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    preifftUVCorrection <<<blocks,threadsPerBlock,0,data.stream>>> ((cuComplex*)data.inputs[0].pointer,(cuComplex*)data.outputs[0].pointer,make_int3(data.inputs[0].dim.getX(),data.inputs[0].dim.getY(),data.inputs[0].dim.getZ())); 
     checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
} 

