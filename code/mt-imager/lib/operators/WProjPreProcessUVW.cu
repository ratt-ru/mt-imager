/* WProjPreProcessUVW.cu:  CUDA implementation of the WProjPreProcessUVW operator 
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

#include "WProjPreProcessUVW.h"
#include "common.hcu"
#include <cuComplex.h>
namespace mtimager { namespace WProjPreProcessUVW_kernels
{
    __device__ __inline__ void doIGrid(pol1float_type &flags,uint &doGrid);
    __device__ __inline__ void doIGrid(pol2float_type &flags,uint &doGrid);
    __device__ __inline__ void doIGrid(pol4float_type &flags,uint &doGrid);
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol1float_type &weights, pol1float_type &flags,uint &myEntry,uint &no_of_records);
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol2float_type &weights, pol2float_type &flags,uint &myEntry,uint &no_of_records);
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol4float_type &weights, pol4float_type &flags,uint &myEntry,uint &no_of_records);
    template <class PolTypeFloat, class PolTypeFlags>
    __global__ void preprocess_uvw(
            //Input: uvw
            float * uvw,
            float *frequency,
            PolTypeFloat * weights,
            PolTypeFlags * flags,
            uint no_of_records,
            //Input: convolution support per plane and indexes.. 
            int wplanes, float wpow2increment, 
            uint2 *convData, // x is support for plane and y is index of plane
            float *in_convSums,
            int sampling,
            //Input image data
            int dim_row,int dim_col, float inc_row, float inc_col,
            //Outputs
            int4 * out_data, 
            int * out_convIndx,
            int * out_gridBool, 
            int * out_compressed,
            int * out_support,  float *out_img_multiply, float *out_convSum,
            bool compressEnabled);
        __global__ void preprocess_uvw_bounderies(
            uint block_length,
            uint no_of_records,
            int4 * data, //5x  
            int *convDataIndx,
            int * gridBool, 
            int * compressed);
    




    __device__ __inline__ void doIGrid(pol1float_type &flags,uint &doGrid)
    {
        if (flags==0.0f) doGrid=1;

    }
    __device__ __inline__ void doIGrid(pol2float_type &flags,uint &doGrid)
    {
        if (flags.x==0.0f) doGrid=1;
        if (flags.y==0.0f) doGrid=1;

    }
    __device__ __inline__ void doIGrid(pol4float_type &flags,uint &doGrid)
    {
        if (flags.x==0.0f) doGrid=1;
        if (flags.y==0.0f) doGrid=1;
        if (flags.z==0.0f) doGrid=1;
        if (flags.w==0.0f) doGrid=1;

    }
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol1float_type &weights, pol1float_type &flags,uint &myEntry,uint &no_of_records)
    {
        out_convSum[myEntry]=convSum*weights*(1.0-flags);

    }
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol2float_type &weights, pol2float_type &flags,uint &myEntry,uint &no_of_records)
    {
        int indx=myEntry;
        out_convSum[indx]=convSum*weights.x*(1.0-flags.x);
        indx+=no_of_records;
        out_convSum[indx]=convSum*weights.y*(1.0-flags.x);

    }
    __device__ __inline__ void calculate_convSum(float *&out_convSum,float & convSum, pol4float_type &weights, pol4float_type &flags,uint &myEntry,uint &no_of_records)
    {
        int indx=myEntry;
        out_convSum[indx]=convSum*weights.x*(1.0-flags.x);
        indx+=no_of_records;
        out_convSum[indx]=convSum*weights.y*(1.0-flags.y);
        indx+=no_of_records;
        out_convSum[indx]=convSum*weights.z*(1.0-flags.z);
        indx+=no_of_records;
        out_convSum[indx]=convSum*weights.w*(1.0-flags.w);

    }


    template <class PolTypeFloat, class PolTypeFlags>
    __global__ void preprocess_uvw(
            //Input: uvw
            float * uvw,
            float *frequency,
            PolTypeFloat * weights,
            PolTypeFlags * flags,
            uint no_of_records,
            //Input: convolution support per plane and indexes.. 
            int wplanes, float wpow2increment, 
            uint2 *convData, // x is support for plane and y is index of plane
            float *in_convSums,
            //float4 *visibilities,
            int sampling,
            //Input image data
            int dim_row,int dim_col, float inc_row, float inc_col,
            //Outputs
            int4 * out_data, 
            int * out_convIndx,
            int * out_gridBool, 
            int * out_compressed,
            int * out_support,  float *out_img_multiply, float *out_convSum,bool compressEnabled)
    {

        __shared__ uint convIndx_shared[1025];
        __shared__ uint upper_corner_pos_shared[1025]; //These are required for "data compression"
        float u,v,w; //data to load
        int4 loc2;
        int convIndx;
        uint toCompress=0; //we assume we do not compress
        uint toGrid; //we assume that we have to grid
        float img_multiply=1.0f;  //leave imaginary part of complex visibility the same
        int support;
        uint indx;
        uint thread0Entry=blockIdx.x*blockDim.x;
        uint myEntry=thread0Entry+threadIdx.x;
        float freq;
        float convSum=0;
        PolTypeFloat myFlags;
        if (myEntry<no_of_records)
        {
            toGrid=0; // Assume for now that we do not grid
            //Load flags first and decide if to grid
            tofloat(flags[myEntry],myFlags);
            doIGrid(myFlags,toGrid);
            //toGrid=1;
            //u=-u;
            //v=-v;
            if (toGrid)
            {
            //Let;'s load our u,v,w data... uvw is 3 x no_of_records that is u is in row 0, v in row 1 etc
                indx=myEntry;
                u=uvw[indx];
                indx+=no_of_records;
                v=uvw[indx];
                indx+=no_of_records;
                w=uvw[indx];
                freq=frequency[myEntry];

                if (w<0.0f) {
                    v=-v;
                    u=-u;
                 //   w=-w;
                    img_multiply=-1.0f;
                }

                            //Don't worry about changing w as we need it's square



                uint wplane=round(sqrt(abs((w*freq)/wpow2increment))); //still need to understand the +1 abd offset.. I beleive it is 0
                if (wplane>(wplanes-1)) wplane=wplanes-1; //When this happens it is no good
                //We can now know load the index of convFunction and support
                uint2 myConvData=convData[wplane]; /// This has to be reviewed for performance
                support=myConvData.x;
                convIndx=myConvData.y; //The answer is not yet complete
              //  printf("%d %f %d %f %f\n",wplane,w,support,freq,wpow2increment);
                float pos_v=(v*freq/(inc_row))+float(dim_row/2);     //THIS IS VERYDANGEROUS (the -) >>> I NEED TO THINK MORE ABOUT IT..It can heavily effect W-projection
                float pos_u=(u*freq/(inc_col))+float(dim_col/2);

                float2 loc=make_float2(round(pos_u),round(pos_v));
                int2 loc_corner=make_int2((int)loc.x-support/2,((int)loc.y-support/2));

                loc2=make_int4(loc_corner.x,loc_corner.y*dim_col+loc_corner.x,-(loc_corner.x%support),-(loc_corner.y%support));

                if (((loc_corner.y)<0) || ((loc_corner.y)>((dim_row-1)-support)) || ((loc_corner.x)<0) || ((loc_corner.x)>(dim_col-1-support)))
                {    
                    toGrid=0; //do not grid
                }
                else
                {
                    //printf ("Entry %i \n",myEntry);
                   //for now we will still grid but there is a second desicion to be taken
                    //toGrid would have already been set to 1 when considering flags
                    //Let's continue calculation of convIndx 
                   int2 offset=make_int2(round(((loc.x-pos_u)*(float)sampling)),round((loc.y-pos_v)*(float)sampling));
                   //For odd sampling if abs(loc-pos)=0.5 the we get an "erronoues" offset, below corrects the problem 

                   //we now transform  bit by bit  this offset to an index
                   //First move.. transform to a an index that begins from 0 

                   offset.x+=sampling/2; //Works for odd and even sampling 
                   offset.y+=sampling/2;  


                   //For odd sampling if abs(loc-pos)=0.5 the we get an "erronoues" offset, below corrects the problem 
                   // TO REVIEW method
                   if (sampling%2)
                   {
                       if (offset.x<0) offset.x=0;
                       if (offset.y<0) offset.y=0;
                       if (offset.x==sampling) offset.x--;
                       if (offset.y==sampling) offset.y--;
                   }
                   int totalsamples;
                   if (sampling%2) totalsamples=sampling;
                   else totalsamples=sampling+1;

                   convSum=in_convSums[wplane*totalsamples*totalsamples+offset.y*totalsamples+offset.x];
                   //NOTE: convSum still needs to be multiplied by weight

                   //In each plane we store we do all offsets of x one after each other for each offset of y
                   //NOTE: this complication is required such that we access coalised memory (not fully as we cannot completely evade segment issues
                   //mem is the whole cuComplex associated with each y offset 
                   int mult=support*support;
                   offset.x*=mult;
                   mult*=totalsamples;
                   offset.y*=mult;
                   convIndx+=offset.y+offset.x; //+1; //FINAL ANSWER //an extra 1
                   //if (w<0) convIndx=-convIndx;
                 }
            }
            if (toGrid==0)
            {
                support=0;
                convIndx=0;
                loc2=make_int4(0,0,0,0);
                convSum=0;
            }
            //Ok time to save

            out_data[myEntry]=loc2;
            out_convIndx[myEntry]=convIndx;
            convIndx_shared[threadIdx.x]=convIndx;
            upper_corner_pos_shared[threadIdx.x]=loc2.y;
            out_support[myEntry]=support;
            out_img_multiply[myEntry]=img_multiply;
            //The convSum needs to be multiplied by weight available for each polarization
            indx=myEntry;

            calculate_convSum(out_convSum,convSum, weights[myEntry], myFlags,myEntry,no_of_records);
        }


        __syncthreads();

       //out_gridBool and out_compressed are finalized next

        //Just check if I am the same as my before element
        if (myEntry<no_of_records)
        { //BUGS: THERE ARE HODDEN BUGS HERE. compression for non griddable points
           if ((threadIdx.x!=0)&&(toGrid==1))
           {  
               if ((convIndx==convIndx_shared[threadIdx.x-1])
                      &&(loc2.y==upper_corner_pos_shared[threadIdx.x-1])&&compressEnabled)
              {
                  toGrid=0;
                  toCompress=1;

              }
           }
           out_gridBool[myEntry]=toGrid;
           out_compressed[myEntry]=toCompress;
        }

    }
    __global__ void preprocess_uvw_bounderies(
            uint block_length,
            uint no_of_records,
            int4 * data, //5x  
            int *convDataIndx,
            int * gridBool, 
            int * compressed)
    {
        int pos_upper_corner;
        int convIndx;
        int4 loc2; 
        int pos_upper_corner_before;
        int convIndx_before;


        int myEntry=(blockIdx.x*blockDim.x+threadIdx.x+1)*block_length;  //+1 is becuase we ignore entry 0

        if (myEntry<no_of_records)
        {
            if (gridBool[myEntry]) //ie we want to grid
            {

                loc2=data[myEntry];
                pos_upper_corner=loc2.y;
                convIndx=convDataIndx[myEntry];
                /*
                if ((pos_upper_corner==pos_upper_corner_before)
                        &&(convIndx==convIndx_before))
                {*/
                if ((data[myEntry].y==data[myEntry-1].y)&&(convDataIndx[myEntry]==convDataIndx[myEntry-1]))
                {
                    compressed[myEntry]=1;
                    gridBool[myEntry]=0;
                }

            }
        }
    }
}}
using namespace mtimager;
using namespace mtimager::WProjPreProcessUVW_kernels;

template<class PolFloatType, class PolFlagType>
void WProjPreProcessUVW::kernel_launch_wrapper(GAFW::GPU::GPUSubmissionData &data,dim3 &blocks, dim3 &threadsPerBlock, int& records)
{
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    

    preprocess_uvw<PolFloatType,PolFlagType> <<<blocks,threadsPerBlock,0,data.stream>>>(
           (float*)data.inputs[2].pointer,
            (float*)data.inputs[3].pointer,
            (PolFloatType*)data.inputs[4].pointer,
            (PolFlagType*)data.inputs[5].pointer,

            records,
            //Input: convolution support per plane and indexes.. 
           data.inputs[0].dim.getNoOfRows(), 
           data.params.getFloatProperty("ConvolutionFunction.wsquareincrement"),

           (uint2*)data.inputs[0].pointer,
           (float*)data.inputs[1].pointer,

           //(float4*)data.outputs[1].pointer,
           (float)data.params.getIntProperty("ConvolutionFunction.sampling"),

           //Input image data
           //int dim_row,int dim_col,
           data.params.getIntProperty("uvImage.rows"),
           data.params.getIntProperty("uvImage.columns"),
           data.params.getFloatProperty("uvImage.v_increment"),data.params.getFloatProperty("uvImage.u_increment"),

           //Outputs

           (int4*)data.outputs[0].pointer,
           (int*)data.outputs[1].pointer,
           (int*)data.outputs[2].pointer,
           (int*)data.outputs[3].pointer,
           (int*)data.outputs[4].pointer,
           (float*)data.outputs[5].pointer,
           (float*) data.outputs[6].pointer,true);

    blocks.x/=1024;
    blocks.x++;

    if (true)
    {
    preprocess_uvw_bounderies<<<blocks,threadsPerBlock,0,data.stream>>>(
             1024,records,
            (int4*)data.outputs[0].pointer,
            (int*)data.outputs[1].pointer,
            (int*)data.outputs[2].pointer,
            (int*)data.outputs[3].pointer
      );  
    }
    checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
    data.endEventRecorded=true;
}
void WProjPreProcessUVW::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
       
     dim3 threadsPerBlock;
     dim3 blocks;
     
     int records=data.inputs[2].dim.getNoOfColumns();
     
     threadsPerBlock.x=1024;
     threadsPerBlock.y=1;
     threadsPerBlock.z=1;
     blocks.x=records/threadsPerBlock.x;
     blocks.x++;
     blocks.y=1;
     blocks.z=1;
      //Input 0 convolution functions data
     //input 1 uvw records
     
     //output 0 gridding data
     //output 1 to/not to grid data
     //output 2 toCompress
     //output 3 convolution function support
     //output 4 sameSupport
     switch (data.inputs[4].dim.getNoOfColumns())
     {
         case 1:
            this->kernel_launch_wrapper<pol1float_type,pol1flags_type>(data, blocks,threadsPerBlock, records);
            break;
         case 2:
            this->kernel_launch_wrapper<pol2float_type,pol2flags_type>(data, blocks,threadsPerBlock, records);
            break;
         case 4:
            this->kernel_launch_wrapper<pol4float_type,pol4flags_type>(data, blocks,threadsPerBlock, records);
            break;
     }
}
