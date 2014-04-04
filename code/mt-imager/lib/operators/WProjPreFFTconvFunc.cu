/* WProjPreFFTconvFunc.cu:  CUDA implementation of the WProjPreFFTconvFunc operator 
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
#include <stdio.h>
#include "WProjPreFFTconvFunc.h"
#include "common.hcu"
#include "mtimager.h"
#include <cuComplex.h>

namespace mtimager { namespace WProjPreFFTconvFunc_kernels
{
    template <class ConvType,class PrecisionType, int leavedge>
    __global__ void calculate_screen_and_correction(ConvType *convFunc, 
                PrecisionType* corMatrix, int support,int sampling,
                PrecisionType l_increment, PrecisionType m_increment,  
                PrecisionType w);


    template <class ConvType,class PrecisionType, int leavedge>
    __global__ void calculate_screen_and_correction(ConvType *convFunc, 
            PrecisionType* corMatrix, int support,int sampling,
            PrecisionType l_increment, PrecisionType m_increment,  
            PrecisionType w)
    {

        int2 loc=make_int2(blockIdx.x * blockDim.x + threadIdx.x,blockIdx.y * blockDim.y + threadIdx.y);
       // if ((loc.x==0)&&(loc.y==0)) printf ("W------------------ %f\n",w);

        //int planesize=(support+1)*sampling-1;
        int planesize=support*sampling;
        //if (!(sampling%2))
         //   planesize++;
        if ((loc.x<planesize)&&(loc.y<planesize))
        {
            int2 coordinate=make_int2(loc.x-planesize/2,loc.y-planesize/2);
            ConvType pointvalue;


           // printf ("%i\n",leavedge);
            if ((coordinate.y<-support/2)||(coordinate.y>(support/2-leavedge))||(coordinate.x<-support/2)||(coordinate.x>(support/2-leavedge)))
            {
                pointvalue.x=0;
                pointvalue.y=0;
            }
            else
            {
                PrecisionType twoW=2*w;
                PrecisionType msq=m_increment*(PrecisionType)coordinate.y;
                msq*=msq;  //(m squared ie m^2)
                PrecisionType lsq=l_increment*(PrecisionType)coordinate.x;  //(l squared ie l^2)
                lsq*=lsq;

                PrecisionType rsq=lsq+msq;
                if (rsq<1.0)
                {
                    PrecisionType phase=twoW*(sqrt(1.0-rsq)-1.0);//to check


                    pointvalue.x=cospi(phase);
                    pointvalue.y=sinpi(phase);
                    //Not sure of below
                    PrecisionType correction= *(corMatrix+((coordinate.y+support/2)*support +(coordinate.x+support/2)));
                    pointvalue.x*=correction;
                    pointvalue.y*=correction; 

                }
                else
                {
                    //printf ("rsq greater then 0");
                    PrecisionType correction= *(corMatrix+((coordinate.y+support/2)*support+(coordinate.x+support/2)));
                    pointvalue.x=correction;  //should be 0.0????? Do not kmow yet
                    pointvalue.y=0; //correction;

                }


            }
            int2 newloc;
            newloc.x=OriginToCorner1D(loc.x, planesize);
            newloc.y=OriginToCorner1D(loc.y, planesize);
         /*  if ((newloc.x==0)&&(newloc.y==0))
               printf("%f %f\n", pointvalue.x,pointvalue.y);*/
            //Which position should this thread update
            int position= newloc.y*planesize+newloc.x;
            (convFunc+position)->x=pointvalue.x;
            (convFunc+position)->y=pointvalue.y;

        }

    }
}}
using namespace mtimager;
using namespace mtimager::WProjPreFFTconvFunc_kernels;

void WProjPreFFTConvFunc::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     //We expect no data.inputs and one output
     
    //int wplanes;
    float w;
     int support,sampling;
     //float wsquarescale;
     double lincrement;
     double mincrement;
     double total_l;
     double total_m;
     //wplanes=data.params.getIntProperty("wplanes");
     //wsquarescale=data.params.getFloatProperty("wsquarescale");
     w=data.params.getFloatProperty("w");
     support=data.params.getIntProperty("support");
     sampling=data.params.getIntProperty("sampling");
     total_l=data.params.getDoubleProperty("total_l");
     total_m=data.params.getDoubleProperty("total_m");
     
     lincrement=total_l/(double)support;
     mincrement=total_m/(double)support;
     
     dim3 threadsPerBlock;
     dim3 blocks;
     
     data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
     switch (data.outputs[0].type)
     {
         case GAFW::GeneralImplimentation::complex_float:
             if (this->getBoolParameter("lwimager-removeedge"))
                calculate_screen_and_correction<cuComplex,float,1><<<blocks,threadsPerBlock,0,data.stream>>>
                ((cuComplex *)data.outputs[0].pointer, 
                (float*) data.inputs[0].pointer, 
                support,sampling,(float)lincrement,(float)mincrement,
                w);
             else
                 calculate_screen_and_correction<cuComplex,float,0><<<blocks,threadsPerBlock,0,data.stream>>>
                ((cuComplex *)data.outputs[0].pointer, 
                (float*) data.inputs[0].pointer, 
                support,sampling,(float)lincrement,(float)mincrement,
                w);
             
             break;
         case GAFW::GeneralImplimentation::complex_double:
             if (this->getBoolParameter("lwimager-removeedge"))
                calculate_screen_and_correction<cuDoubleComplex,double,1><<<blocks,threadsPerBlock,0,data.stream>>>
                ((cuDoubleComplex *)data.outputs[0].pointer, 
                (double*) data.inputs[0].pointer, 
                support,sampling,(double)lincrement,(double)mincrement,
                w);
             else
                 calculate_screen_and_correction<cuDoubleComplex,double,0><<<blocks,threadsPerBlock,0,data.stream>>>
                ((cuDoubleComplex *)data.outputs[0].pointer, 
                (double*) data.inputs[0].pointer, 
                support,sampling,(double)lincrement,(double)mincrement,
                w);
             break;
     }      
      
    checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;

             
}