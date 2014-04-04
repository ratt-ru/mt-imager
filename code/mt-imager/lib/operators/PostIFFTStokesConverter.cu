/* PostIFFTStokesConverter.cu:  CUDA implementation of the PostIFFTStokesConverter operator 
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

#include "PostIFFTStokesConverter.h"
#include "common.hcu"
#include "mtimager.h"
#include <cuComplex.h>
namespace mtimager { namespace PostIFFTStokeConverter_kernels
{
    template<class DataType,int pols>
    struct DataHolder
    {
        DataType data[pols];
    };
    template<int pols,PolarizationType::Type polType>
    __device__ void stokesTransform(DataHolder<cuComplex,pols> &indata,DataHolder<float,pols> &outdata );
    template <class NormalizerType, int pols>
    __device__ void normalize(DataHolder<cuComplex,pols> &indata,NormalizerType &norm);
    __device__ void inverse_normalizers(float4 &normalizer);
    __device__ void inverse_normalizers(float2 &normalizer);
    __device__ void inverse_normalizers(float &normalizer);
    template<int pols,enum PolarizationType::Type polType, class NormalizerType>
    __global__ void StokesConv(cuComplex * input, float * output, NormalizerType *normalizer, uint2 dim);

    
    template<int pols,PolarizationType::Type polType>
    __device__ void stokesTransform(DataHolder<cuComplex,pols> &indata,DataHolder<float,pols> &outdata )
    {
        for (int i=0;i<pols;i++)
        {
            outdata.data[i]=-1.0; 
        }
    }
    //0 linear polarisation
    template<>
    __device__ void stokesTransform<4,PolarizationType::Linear>(DataHolder<cuComplex,4> &indata, DataHolder<float,4> &outdata )
    {

            outdata.data[0]=(indata.data[0].x+indata.data[3].x)*0.5;
            outdata.data[1]=(indata.data[0].x-indata.data[3].x)*0.5;
            outdata.data[2]=(indata.data[1].x+indata.data[2].x)*0.5;
            outdata.data[3]=(indata.data[1].y-indata.data[2].y)*0.5;
    }
    template<>
    __device__ void stokesTransform<2,PolarizationType::Linear>(DataHolder<cuComplex,2> &indata, DataHolder<float,2> &outdata)
    {
            outdata.data[0]=(indata.data[0].x+indata.data[1].x)*0.5;
            outdata.data[1]=(indata.data[0].x-indata.data[1].x)*0.5;
    }
    template<>
    __device__ void stokesTransform<1,PolarizationType::Linear>(DataHolder<cuComplex,1> &indata, DataHolder<float,1> &outdata )
    {
            outdata.data[0]=indata.data[0].x;
    }

    //circular
    template<>
    __device__ void stokesTransform<4,PolarizationType::Circular>(DataHolder<cuComplex,4> &indata, DataHolder<float,4> &outdata )
    {
            outdata.data[0]=(indata.data[0].x+indata.data[3].x)*0.5;
            outdata.data[1]=(indata.data[1].x+indata.data[2].x)*0.5;
            outdata.data[2]=(indata.data[2].y-indata.data[1].y)*0.5;
            outdata.data[3]=(indata.data[0].x-indata.data[3].x)*0.5;
    }
    template<>
    __device__ void stokesTransform<2,PolarizationType::Circular>(DataHolder<cuComplex,2> &indata, DataHolder<float,2> &outdata)
    {
            outdata.data[0]=(indata.data[0].x+indata.data[1].x)*0.5;
            outdata.data[1]=(indata.data[0].x-indata.data[1].x)*0.5;
    }
    template<>
    __device__ void stokesTransform<1,PolarizationType::Circular>(DataHolder<cuComplex,1> &indata, DataHolder<float,1> &outdata)
    {
            outdata.data[0]=(indata.data[0].x);

    }

    template <class NormalizerType, int pols>
    __device__ void normalize(DataHolder<cuComplex,pols> &indata,NormalizerType &norm)
    {
        //Don't do anything thsi function shoudl not be called

    }
    template <>
    __device__ void normalize<float4,4>(DataHolder<cuComplex,4> &indata,float4 &norm)
    {
        //Not all data is used so we only normalize the data that we use

        indata.data[0].x/=norm.x;
        indata.data[1].x/=norm.y;
        indata.data[1].y/=norm.y;
        indata.data[2].x/=norm.z;
        indata.data[2].y/=norm.z;
        indata.data[3].x/=norm.w;
    }
    template <>
    __device__ void normalize<float,1>(DataHolder<cuComplex,1> &indata,float &norm)
    {
        //Not all data is used so we only normalize the data that we use
        //indata.data[0].x=(float)(double(indata.data[0].x)/double(norm));
        indata.data[0].x=(indata.data[0].x)/(norm);
    }

    template <>
    __device__ void normalize<float2,2>(DataHolder<cuComplex,2> &indata,float2 &norm)
    {
        //Not all data is used so we only normalize the data that we use

        indata.data[0].x/=norm.x;
        indata.data[1].x/=norm.y;

    }
    __device__ void inverse_normalizers(float4 &normalizer)
    {
        normalizer.x=1.0/normalizer.x;
        normalizer.y=1.0/normalizer.y;
        normalizer.z=1.0/normalizer.z;
        normalizer.w=1.0/normalizer.w;

    }
    __device__ void inverse_normalizers(float2 &normalizer)
    {
        normalizer.x=1.0/normalizer.x;
        normalizer.y=1.0/normalizer.y;

    }

    __device__ void inverse_normalizers(float &normalizer)
    {  
        normalizer=1.0/normalizer;
    }


 
    template<int pols,enum PolarizationType::Type polType, class NormalizerType>
    __global__ void StokesConv(cuComplex * input, float * output, NormalizerType *normalizer, uint2 dim)
    {
        DataHolder<cuComplex,pols> indata;
        DataHolder<float,pols> outdata;
        NormalizerType normalizers=*normalizer;
        //inverse_normalizers(normalizers);
        uint2 loc;
        uint totalPoints=dim.x*dim.y;
        for (loc.y=threadIdx.y+blockIdx.y*blockDim.y;loc.y<dim.y;loc.y+=gridDim.y*blockDim.y)
            for (loc.x=threadIdx.x+blockIdx.x*blockDim.x;loc.x<dim.x;loc.x+=gridDim.x*blockDim.x)
            {

                uint inputloc=loc.y*dim.x+loc.x;
        #pragma unroll
                for (int i=0;i<pols;i++)
                {
                    indata.data[i]=input[inputloc+i*totalPoints];
                }
                normalize<NormalizerType,pols>(indata,normalizers);
                stokesTransform<pols,polType>(indata,outdata);
                uint outputloc=(dim.y-loc.y-1)*dim.x+loc.x; //invert m axes due to FITS
        #pragma unroll
                for (int i=0;i<pols;i++)
                {
                    output[outputloc+i*totalPoints]=outdata.data[i];
                }

            }
    }

}}

using namespace mtimager;
using namespace mtimager::PostIFFTStokeConverter_kernels;
template <enum PolarizationType::Type polType>
void PostIFFTStokesConverter_really_submit(GAFW::GPU::GPUSubmissionData &data)
{
     dim3 threadsPerBlock;
     dim3 blocks;
     uint2 dim;
     dim.x=data.inputs[0].dim.getNoOfColumns();
     dim.y=data.inputs[0].dim.getNoOfRows();
     threadsPerBlock.x=32;
     threadsPerBlock.y=32;
     threadsPerBlock.z=1;
     blocks.x=32;
     blocks.y=32;
     blocks.z=1;
      checkCudaError(cudaEventRecord(*data.startEvent,data.stream),"Unable to record event");
    

     switch (data.inputs[0].dim.getZ())
     {
         case 4:
                StokesConv<4,polType,float4><<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer, (float *) data.outputs[0].pointer,(float4*)data.inputs[1].pointer, dim);
                break;
         case 2:
                StokesConv<2,polType,float2><<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer, (float *) data.outputs[0].pointer,(float2*)data.inputs[1].pointer, dim);
                break;
         case 1:
                StokesConv<1,polType,float><<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer, (float *) data.outputs[0].pointer,(float*)data.inputs[1].pointer,  dim);
                break;
         
     }
     
       checkCudaError(cudaEventRecord(*data.endEvent,data.stream),"Unable to record event");
        data.endEventRecorded=true;
}
void PostIFFTStokesConverter::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    switch((enum PolarizationType::Type)data.params.getIntProperty("PolarizationType"))
    {
        case PolarizationType::Circular:
            PostIFFTStokesConverter_really_submit<PolarizationType::Circular>(data);
            break;
        case PolarizationType::Linear:
            PostIFFTStokesConverter_really_submit<PolarizationType::Linear>(data);
            break;

    }


}
