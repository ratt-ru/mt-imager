/* ConvolutionGridder.cu:  CUDA implementation of the ConvolutionGridder operator 
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

#include "ConvolutionGridder.h"
#include <cuComplex.h>
#include "common.hcu"

namespace mtimager { namespace ConvolutionGridder_kernels
{
      template <class PolVisType>
    __device__ void convgrid_load_data(int &myPlanEntry,int2 *workPlan,
              int *dataIdx, 
        int4 *gridData,
        PolVisType *visibilities, 
        int *convIndx,
        int &shared_lastPlanEntry,
        int &shared_loads_per_block,
        int &shared_load_begin_entry,
        int &shared_support,
        int &shared_load_end_entry, //Entry not included
        int &shared_loaded_entries,
        int4 *shared_loc,
        int *shared_convPointers,
        PolVisType * shared_visibilities,
        int totBlockDim
        );
        __device__ __inline__ void visAtomicAdd(pol1vis_type& out,pol1vis_type& toAdd);
        __device__ __inline__ void visAtomicAdd(pol2vis_type& out,pol2vis_type& toAdd);
        __device__ __inline__ void visAtomicAdd(pol4vis_type& out,pol4vis_type& toAdd);
        __device__ __inline__ void visWeightedAdd(pol1vis_type& to,cuComplex &wt,pol1vis_type &visToAdd);
        __device__ __inline__ void visWeightedAdd(pol2vis_type& to,cuComplex &wt,pol2vis_type &visToAdd);
        __device__ __inline__ void visWeightedAdd(pol4vis_type& to,cuComplex &wt,pol4vis_type &visToAdd);
        template <class PolVisType>
        __global__ void convgrid_under16(
                int2 *workPlan,   
            int workPlanMaximumRecords, 
            int *dataIdx, 
            int4 *gridData,
            PolVisType *visibilities, 
            cudaTextureObject_t texObj,
            int *convIndx, 
            PolVisType * uvImage, 
            int dim_col, int totBlockDim);
        template <class PolVisType>
        __global__ void convgrid( 
            int2 *workPlan,   
            int workPlanMaximumRecords, 
            int *dataIdx, 
            int4 *gridData,
            PolVisType *visibilities, 
            cudaTextureObject_t texObj,
            int *convIndx, 
            PolVisType * uvImage, 
            int dim_col,int totBlockDim);

        
    template <class PolVisType>
    __device__ void convgrid_load_data(int &myPlanEntry,int2 *workPlan,
        int *dataIdx, 
        int4 *gridData,
        PolVisType *visibilities, 
        int *convIndx,
        int &shared_lastPlanEntry,
        int &shared_loads_per_block,
        int &shared_load_begin_entry,
        int &shared_support,
        int &shared_load_end_entry, //Entry not included
        int &shared_loaded_entries,
        int4 *shared_loc,
        int *shared_convPointers,
        PolVisType * shared_visibilities,
        int totBlockDim
        )    
    {
        for (int myInternalEntry=threadIdx.y*blockDim.x+threadIdx.x;myInternalEntry<shared_loaded_entries;myInternalEntry+=totBlockDim)
        {
            int myEntry=shared_load_begin_entry+myInternalEntry;
            int entryInGridData=dataIdx[myEntry];
            shared_loc[myInternalEntry]=gridData[entryInGridData];
            shared_convPointers[myInternalEntry]=convIndx[myEntry];
            shared_visibilities[myInternalEntry]=visibilities[myEntry];
        }
        __syncthreads();
    }

    __device__ __inline__ void visAtomicAdd(pol1vis_type& out,pol1vis_type& toAdd)
    {


        atomicAdd(&(out.vis.x),toAdd.vis.x);
        atomicAdd(&(out.vis.y),toAdd.vis.y);
    }
    __device__ __inline__ void visAtomicAdd(pol2vis_type& out,pol2vis_type& toAdd)
    {
        atomicAdd(&(out.vis_X_Y.x),toAdd.vis_X_Y.x);
        atomicAdd(&(out.vis_X_Y.y),toAdd.vis_X_Y.y);
        atomicAdd(&(out.vis_X_Y.z),toAdd.vis_X_Y.z);
        atomicAdd(&(out.vis_X_Y.w),toAdd.vis_X_Y.w);
    }
    __device__ __inline__ void visAtomicAdd(pol4vis_type& out,pol4vis_type& toAdd)
    {


        atomicAdd(&(out.vis_XX_XY.x),toAdd.vis_XX_XY.x);
        atomicAdd(&(out.vis_XX_XY.y),toAdd.vis_XX_XY.y);
        atomicAdd(&(out.vis_XX_XY.z),toAdd.vis_XX_XY.z);
        atomicAdd(&(out.vis_XX_XY.w),toAdd.vis_XX_XY.w);
        atomicAdd(&(out.vis_YX_YY.x),toAdd.vis_YX_YY.x);
        atomicAdd(&(out.vis_YX_YY.y),toAdd.vis_YX_YY.y);
        atomicAdd(&(out.vis_YX_YY.z),toAdd.vis_YX_YY.z);
        atomicAdd(&(out.vis_YX_YY.w),toAdd.vis_YX_YY.w);

    }
    __device__ __inline__ void visWeightedAdd(pol1vis_type& to,cuComplex &wt,pol1vis_type &visToAdd)
    {
        to.vis.x+=visToAdd.vis.x* wt.x;
        to.vis.y+=visToAdd.vis.y* wt.x;
        to.vis.x-=visToAdd.vis.y* wt.y;   
        to.vis.y+=visToAdd.vis.x* wt.y;
    }
    __device__ __inline__ void visWeightedAdd(pol2vis_type& to,cuComplex &wt,pol2vis_type &visToAdd)
    {
        //Pol 0
        to.vis_X_Y.x+=visToAdd.vis_X_Y.x* wt.x;
        to.vis_X_Y.y+=visToAdd.vis_X_Y.y* wt.x;
        to.vis_X_Y.x-=visToAdd.vis_X_Y.y* wt.y;   
        to.vis_X_Y.y+=visToAdd.vis_X_Y.x* wt.y;

        //Pol 1
        to.vis_X_Y.z+=visToAdd.vis_X_Y.z* wt.x;
        to.vis_X_Y.w+=visToAdd.vis_X_Y.w* wt.x;
        to.vis_X_Y.z-=visToAdd.vis_X_Y.w* wt.y;   
        to.vis_X_Y.w+=visToAdd.vis_X_Y.z* wt.y;


    }
    __device__ __inline__ void visWeightedAdd(pol4vis_type& to,cuComplex &wt,pol4vis_type &visToAdd)
    {
        //Pol 0
        to.vis_XX_XY.x+=visToAdd.vis_XX_XY.x* wt.x;
        to.vis_XX_XY.y+=visToAdd.vis_XX_XY.y* wt.x;
        //Pol 1
        to.vis_XX_XY.z+=visToAdd.vis_XX_XY.z* wt.x;
        to.vis_XX_XY.w+=visToAdd.vis_XX_XY.w* wt.x;
        //Pol 2
        to.vis_YX_YY.x+=visToAdd.vis_YX_YY.x* wt.x;
        to.vis_YX_YY.y+=visToAdd.vis_YX_YY.y* wt.x;
        //Pol 3
        to.vis_YX_YY.z+=visToAdd.vis_YX_YY.z* wt.x;
        to.vis_YX_YY.w+=visToAdd.vis_YX_YY.w* wt.x;

        //pol-0

        to.vis_XX_XY.x-=visToAdd.vis_XX_XY.y* wt.y;   
        to.vis_XX_XY.y+=visToAdd.vis_XX_XY.x* wt.y;
        //pol1
        to.vis_XX_XY.z-=visToAdd.vis_XX_XY.w* wt.y;   
        to.vis_XX_XY.w+=visToAdd.vis_XX_XY.z* wt.y;
        //pol2
        to.vis_YX_YY.x-=visToAdd.vis_YX_YY.y* wt.y;   
        to.vis_YX_YY.y+=visToAdd.vis_YX_YY.x* wt.y;
        //pol3
        to.vis_YX_YY.z-=visToAdd.vis_YX_YY.w* wt.y;   
        to.vis_YX_YY.w+=visToAdd.vis_YX_YY.z* wt.y;
    }

    template <class PolVisType>
    __global__ void convgrid_under16( 
            int2 *workPlan,   
            int workPlanMaximumRecords, 
            int *dataIdx, 
            int4 *gridData,
            PolVisType *visibilities, 
            //const  cuComplex * __restrict__ convFunc,
            cudaTextureObject_t texObj,

            int *convIndx, 
            PolVisType * uvImage, 
            int dim_col, int totBlockDim)
    {
        __shared__ int shared_lastPlanEntry;
        __shared__ int shared_loads_per_block;
        __shared__ int shared_load_begin_entry;
        __shared__ int shared_support;
        __shared__ int shared_load_end_entry; //Entry not included
        __shared__ int shared_loaded_entries;
        __shared__ int beginEntry; // used only in thread0

        __shared__ int4 shared_loc[TOTALLOADS_UNDER16];
        __shared__ int shared_convPointers[TOTALLOADS_UNDER16];
        __shared__ PolVisType shared_visibilities[TOTALLOADS_UNDER16];
        //First thing we must make decisions
        if ((threadIdx.x==0)&&(threadIdx.y==0))
        {   
            shared_lastPlanEntry=workPlan[workPlanMaximumRecords].x;

            shared_loads_per_block=shared_lastPlanEntry/gridDim.x + 1;
            beginEntry=blockIdx.x*shared_loads_per_block;


        }

        __syncthreads();
        /*for (int loadno=0;loadno<shared_loads_per_block;loadno++)*/

        for (int myPlanEntry=blockIdx.x;myPlanEntry<shared_lastPlanEntry;myPlanEntry+=gridDim.x)
        {

            __syncthreads();

            if ((threadIdx.x==0)&&(threadIdx.y==0))
            {

                //IN here there is a rare BUG... 
                //what if all workPlan consumes all entries? Difficult to happen.. To check we can set TOTALLOADS to 1
                int2 plan_begin=workPlan[myPlanEntry];
                shared_support=plan_begin.y;


                int plan_end=workPlan[myPlanEntry+1].x;
                shared_load_begin_entry=plan_begin.x;
                shared_load_end_entry=plan_end;
                shared_loaded_entries=plan_end-plan_begin.x;
                //if (shared_loaded_entries<0) shared_loaded_entries=0;
            }
            __syncthreads();
            if (shared_support>16) continue;

            convgrid_load_data<PolVisType>(myPlanEntry,workPlan,dataIdx,gridData,visibilities,convIndx,
                    shared_lastPlanEntry,shared_loads_per_block,
                    shared_load_begin_entry,shared_support,shared_load_end_entry,
                    shared_loaded_entries,shared_loc,shared_convPointers,
                    shared_visibilities,totBlockDim);
            if ((threadIdx.x>=shared_support)||(threadIdx.y>=shared_support)) continue;

            int box_pos_u=threadIdx.x;
            int box_pos_v=threadIdx.y;


            PolVisType myPoint;
            zerofyvis(myPoint);
            int last_poss;
            int index;
            float2 wt;
            //position the box on 1st location
            {
                int4 loc=shared_loc[0];
                int2 supportIndex=make_int2(loc.z+box_pos_u,loc.w+box_pos_v);
                if (supportIndex.x<0) supportIndex.x+=shared_support;
                if (supportIndex.y<0) supportIndex.y+=shared_support;
                index=supportIndex.x+supportIndex.y*shared_support;
                    //int index=supportIndex.y*support;
                last_poss=((loc.y+supportIndex.x)+supportIndex.y*dim_col);
                //last_poss=0;
            }

            for (int internalEntry=0;internalEntry<shared_loaded_entries;internalEntry++)
            {

                    __prof_trigger(0);
                    int4 loc=shared_loc[internalEntry];

                    //supportIndex.x=loc.z+box_pos_u;
                    //supportIndex.y=loc.w+box_pos_v;
                    int2 supportIndex=make_int2(loc.z+box_pos_u,loc.w+box_pos_v);
                    if (supportIndex.x<0) supportIndex.x+=shared_support;
                    if (supportIndex.y<0) supportIndex.y+=shared_support;


                    index=supportIndex.x+supportIndex.y*shared_support;//-1;
                    /*if (shared_convPointers[internalEntry]<0)
                    {

                        int convIndex=-shared_convPointers[internalEntry]+index;
                            //convFunc[shared_convPointers[internalEntry]+index];
                            //wt=tex1Dfetch(supportTexture, convIndex);
                            wt=tex1Dfetch<cuComplex>(texObj,convIndex);
                            wt.y=-wt.y;

                    }
                    else
                    {*/

                            int convIndex=shared_convPointers[internalEntry]+index;
                            //convFunc[shared_convPointers[internalEntry]+index];
                            //wt=tex1Dfetch(supportTexture, convIndex);
                            wt=tex1Dfetch<cuComplex>(texObj,convIndex);
                    //}
                    int poss=((loc.y+supportIndex.x)+supportIndex.y*dim_col);
                    if (last_poss!=poss)
                    {
                        visAtomicAdd(uvImage[last_poss],myPoint);
                        zerofyvis(myPoint);
                        last_poss=poss;
                    }


                    visWeightedAdd(myPoint,wt,shared_visibilities[internalEntry]);


           }
           visAtomicAdd(uvImage[last_poss],myPoint);
        }
    }



    template <class PolVisType>
    __global__ void convgrid( 
            int2 *workPlan,   
            int workPlanMaximumRecords, 
            int *dataIdx, 
            int4 *gridData,
            PolVisType *visibilities, 
            cudaTextureObject_t texObj,

            int *convIndx, 
            PolVisType * uvImage, 
            int dim_col,int totBlockDim)
    {
        __shared__ int shared_lastPlanEntry;
        __shared__ int shared_loads_per_block;
        __shared__ int shared_load_begin_entry;
        __shared__ int shared_support;
        __shared__ int shared_load_end_entry; //Entry not included
        __shared__ int shared_loaded_entries;
        __shared__ int beginEntry; // used only in thread0

        __shared__ int4 shared_loc[TOTALLOADS];
        __shared__ int shared_convPointers[TOTALLOADS];
        __shared__ PolVisType shared_visibilities[TOTALLOADS];
        //First thing we must make decisions
        if ((threadIdx.x==0)&&(threadIdx.y==0))
        {   
            shared_lastPlanEntry=workPlan[workPlanMaximumRecords].x;

            shared_loads_per_block=shared_lastPlanEntry/gridDim.x + 1;
            beginEntry=blockIdx.x*shared_loads_per_block;


        }

        __syncthreads();

        for (int myPlanEntry=blockIdx.x;myPlanEntry<shared_lastPlanEntry;myPlanEntry+=gridDim.x)
        {


            __syncthreads();

            if ((threadIdx.x==0)&&(threadIdx.y==0))
            {

                //IN here there is a rare BUG... 
                //what if all workPlan consumes all entries? Difficult to happen.. To check we can set TOTALLOADS to 1
                int2 plan_begin=workPlan[myPlanEntry];
                shared_support=plan_begin.y;


                int plan_end=workPlan[myPlanEntry+1].x;
                shared_load_begin_entry=plan_begin.x;
                shared_load_end_entry=plan_end;
                shared_loaded_entries=plan_end-plan_begin.x;
                //if (shared_loaded_entries<0) shared_loaded_entries=0;
            }
            __syncthreads();
            if (shared_support<16) continue;
            convgrid_load_data<PolVisType>(myPlanEntry,workPlan,dataIdx,gridData,visibilities,convIndx,
                    shared_lastPlanEntry,shared_loads_per_block,
                    shared_load_begin_entry,shared_support,shared_load_end_entry,
                    shared_loaded_entries,shared_loc,shared_convPointers,
                    shared_visibilities,totBlockDim);





            for (int box_pos_u=threadIdx.x;box_pos_u<shared_support;box_pos_u+=(int)blockDim.x)
                for (int box_pos_v=threadIdx.y;box_pos_v<shared_support;box_pos_v+=(int)blockDim.y)
                {


                    PolVisType myPoint;
                    zerofyvis(myPoint);
                    int last_poss;
                    int index;
                    float2 wt;
                    //position the box on 1st location
                    {
                        int4 loc=shared_loc[0];
                        int2 supportIndex=make_int2(loc.z+box_pos_u,loc.w+box_pos_v);
                        if (supportIndex.x<0) supportIndex.x+=shared_support;
                        if (supportIndex.y<0) supportIndex.y+=shared_support;
                        index=supportIndex.x+supportIndex.y*shared_support;
                        last_poss=((loc.y+supportIndex.x)+supportIndex.y*dim_col);
                    }

                    for (int internalEntry=0;internalEntry<shared_loaded_entries;internalEntry++)
                    {
                            __prof_trigger(0);
                            int4 loc=shared_loc[internalEntry];

                            int2 supportIndex=make_int2(loc.z+box_pos_u,loc.w+box_pos_v);
                            if (supportIndex.x<0) supportIndex.x+=shared_support;
                            if (supportIndex.y<0) supportIndex.y+=shared_support;


                            index=supportIndex.x+supportIndex.y*shared_support;//-1;
                                    int convIndex=shared_convPointers[internalEntry]+index;
                                    wt=tex1Dfetch<cuComplex>(texObj,convIndex);
                            int poss=((loc.y+supportIndex.x)+supportIndex.y*dim_col);
                            if (last_poss!=poss)
                            {
                                visAtomicAdd(uvImage[last_poss],myPoint);
                                zerofyvis(myPoint);
                                last_poss=poss;
                            }


                            visWeightedAdd(myPoint,wt,shared_visibilities[internalEntry]);


                   }
                   visAtomicAdd(uvImage[last_poss],myPoint);

                }
        }
    }
}}

using namespace mtimager;
using namespace mtimager::ConvolutionGridder_kernels;

template <class PolVisType> 
void ConvolutionGridder::kernel_launch_wrapper(dim3 &blocks,dim3 &threadsPerBlock,GAFW::GPU::GPUSubmissionData &data, int records,Texture *tex)
{
         
     cudaFuncSetCacheConfig( convgrid_under16<PolVisType>, cudaFuncCachePreferShared );
     cudaFuncSetCacheConfig( convgrid<PolVisType>, cudaFuncCachePreferShared );
    
     checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
        convgrid<PolVisType><<<blocks,threadsPerBlock,0,data.stream>>>(
                 (int2*)data.inputs[0].pointer,   
        records, //int workPlanMaximumRecords, 
                (int*) data.inputs[1].pointer, //int dataIdx[], 
                (int4*)data.inputs[2].pointer, //int4 gridData[],
                (PolVisType*) data.inputs[3].pointer, //float4 visibilities[][2], 
                //(cuComplex*)data.inputs[4].pointer, //cuComplex convFunc[],
                tex->object,
                (int*) data.inputs[5].pointer,  //int convIndx[], 
                (PolVisType*) data.outputs[0].pointer, //cuComplex  uvImage[][4], 
                data.outputs[0].dim.getNoOfRows(), //int dim_col)
                threadsPerBlock.x*threadsPerBlock.y
                );
        threadsPerBlock.x=16;
        threadsPerBlock.y=16;
        convgrid_under16<PolVisType><<<blocks,threadsPerBlock,0,data.stream>>>(
                 (int2*)data.inputs[0].pointer,   
        records, //int workPlanMaximumRecords, 
                (int*) data.inputs[1].pointer, //int dataIdx[], 
                (int4*)data.inputs[2].pointer, //int4 gridData[],
                (PolVisType*) data.inputs[3].pointer, //float4 visibilities[][2], 
                //(cuComplex*)data.inputs[4].pointer, //cuComplex convFunc[],
                tex->object,
                (int*) data.inputs[5].pointer,  //int convIndx[], 
                (PolVisType*) data.outputs[0].pointer, //cuComplex  uvImage[][4], 
                data.outputs[0].dim.getNoOfRows(), //int dim_col)
                threadsPerBlock.x*threadsPerBlock.y
                );
        
        checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
     

}
    
  void ConvolutionGridder::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     
      if (data.params.getBoolProperty("initialize_output"))
      {
         cudaError_t err=cudaMemsetAsync(data.outputs[0].pointer,0,data.outputs[0].dim.getTotalNoOfElements()*sizeof(cuComplex),data.stream);
         this->logDebug(other,"Memset to 0");
         if (err!=cudaSuccess)
             throw CudaException("Error with zerofying outputs",err);
      }
      
      
     dim3 threadsPerBlock;
     dim3 blocks;
     
     int records=data.inputs[0].dim.getNoOfRows()-1;
     threadsPerBlock.x=32;
     threadsPerBlock.y=32;
     threadsPerBlock.z=1;
     blocks.x=4000;
     blocks.y=1;
     blocks.z=1;
     
     Texture *tex=new Texture;
     // Specify texture
     
     memset(&(tex->resDesc), 0, sizeof(tex->resDesc));
     tex->resDesc.resType = cudaResourceTypeLinear;
     tex->resDesc.res.linear.devPtr = data.inputs[4].pointer;
     tex->resDesc.res.linear.sizeInBytes=data.inputs[4].dim.getTotalNoOfElements()*sizeof(cuComplex);
     tex->resDesc.res.linear.desc=cudaCreateChannelDesc<float2>();
     
     memset(&tex->texDesc, 0, sizeof(tex->texDesc));
     tex->texDesc.readMode = cudaReadModeElementType;
     tex->texDesc.addressMode[0]=cudaAddressModeClamp;
     tex->texDesc.addressMode[1]=cudaAddressModeClamp;
     tex->texDesc.addressMode[2]=cudaAddressModeClamp;
     tex->texDesc.normalizedCoords = 0;
     tex->texDesc.sRGB=0;
     cudaCreateTextureObject(&tex->object, &tex->resDesc, &tex->texDesc, NULL);
     *data.postExecutePointer=(void*)tex;
     
     switch (data.inputs[3].dim.getNoOfColumns()) //every 2 columns is 1 polarisation 
     {
         case 2:  // 1 Polarization
             this->kernel_launch_wrapper<pol1vis_type>(blocks,threadsPerBlock,data,records,tex);
             break;
         case 4:  // 1 Polarization
             this->kernel_launch_wrapper<pol2vis_type>(blocks,threadsPerBlock,data,records,tex);
             break;
         case 8:  // 1 Polarization
             this->kernel_launch_wrapper<pol4vis_type>(blocks,threadsPerBlock,data,records,tex);
             break;
         default:
             throw GAFW::GeneralException("Unknown no of polarisations");
     }
     
}  
    
  
