/* CasaGridSF.cu:  CUDA implementation of the CasaGridSF operator 
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
#include <cmath> 

#include "CasaGridSF.h"
#include <cuComplex.h>
#include "common.hcu"
namespace mtimager { namespace CasaGridSF_kernels
{
    template <class T>
    __inline__ __device__ T gridsf(T nu);
    __inline__ __device__  void gridsf_setReal(float *p,float value);
    __inline__ __device__  void gridsf_setReal(cuComplex *p,float value);
    __inline__ __device__  void gridsf_setReal(double *p,double value);
    __inline__ __device__  void gridsf_setReal(cuDoubleComplex *p,double value);
    template<class T,class V>
    __global__ void gridsf(T *ans,int size,int sampling);
    template <class T>
    __global__ void casacore_gridsf_correction_generate(T *ans,int ny, int nx);
    template <class T>
    __global__ void casacore_gridsf_correction(T *input, T *output,int planes, int rows, int cols);



    template <class T>
    __device__ T gridsf(T nu)
    {
         T P[2][5]= 
         { {8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
            {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2} };
         T Q[2][3]=
         { {1.0, 8.212018e-1, 2.078043e-1},
            {1.0, 9.599102e-1, 2.918724e-1}
         };
         int part;
         T nuendpow2;
         T nupow2=(nu*nu);
         if (nu<0.75)
         {
             part=0;
             nuendpow2=0.5625;
         }
         else
         {
             part=1;
             nuendpow2=1.0;
         }
         T top=P[part][0];
         T delnusq=nupow2 - nuendpow2;
         for (int k=1;k<5;k++)
         {
             top+=P[part][k]*pow(delnusq,k);
         }
        T bot=Q[part][0];
        for (int k=1;k<3;k++)
        {
            bot+=Q[part][k]*pow(delnusq,k);
        }

        if (bot!=0.0)
     {
             return top/bot; 

        }
        else
                     return 0.0;


    }

    __device__ __inline__ void gridsf_setReal(float *p,float value)
    {
        *p=value;
    }
    __device__ __inline__ void gridsf_setReal(cuComplex *p,float value)
    {
        *p=make_float2(value,0.0f);
    }


    __device__ __inline__ void gridsf_setReal(double *p,double value)
    {
        *p=value;
    }
    __device__ __inline__ void gridsf_setReal(cuDoubleComplex *p,double value)
    {
        *p=make_double2(value,0.0f);
    }


    template<class T,class V>
    __global__ void gridsf(T *ans,int size,int sampling)
    {
        //V is either float or double.. depending directly on T
        int i=blockIdx.x * blockDim.x + threadIdx.x;
        int j=blockIdx.y * blockDim.y + threadIdx.y;
        if ((i>=size)||(j>=size)) return; //useless thread

        int pos=j*size + i;

        //Position i and j such that centre is 0 and take abs
        i-=size/2;
        j-=size/2;
        i=abs(i);
        j=abs(j);
        int firstzeropoint=3*sampling;


        // This function re-codes the gridsf and ConvolutionGridder of lwimager
        //Following are comments found in the casacore function

        /***********************************************************************
     C
    CD Find Spheroidal function with M = 6, alpha = 1 using the rational
    C approximations discussed by Fred Schwab in 'Indirect Imaging'.
    C This routine was checked against Fred's SPHFN routine, and agreed
    C to about the 7th significant digit.
    C The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance
    C to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    C is now the distance to the edge of the image.
    C */ 

        V value=0.0f;
        if ((i<firstzeropoint)&&(j<firstzeropoint))
        {
            V nu_i=V(i)/V(firstzeropoint);
            V nu_j=V(j)/V(firstzeropoint);

            value=(1-nu_i*nu_i)*gridsf<V>(nu_i)*(1-nu_j*nu_j)*gridsf<V>(nu_j);
        }
        gridsf_setReal(ans+pos,value);

    }

    template <class T>
    __global__ void casacore_gridsf_correction_generate(T *ans,int ny, int nx)
    {
        int i=blockIdx.x * blockDim.x + threadIdx.x;
        int j=blockIdx.y * blockDim.y + threadIdx.y;
        int el=(j*nx) + i;

        //float * ans_el=ans+el;
        T * ans_el=ans+el;

        if (i<nx && j<ny)  
        {    
            int offset[2]={ j-(ny/2), i-(nx/2) };
            //int offset[2]={ j, i };
            //float offset[2]={float(j)- float(ny)/2.0, float(i)-float(nx)/2.0 }; 
            T nu[2] = { abs((T(offset[0]))/(T(ny/2))),
                            abs((T(offset[1]))/(T(nx/2)))
            };


            *(ans_el)= ((gridsf<T>(nu[0])* gridsf<T>(nu[1])));
        }
    }
    template <class T>
    __global__ void casacore_gridsf_correction(T *input, T *output,int planes, int rows, int cols)
    {
        int i=blockIdx.x * blockDim.x + threadIdx.x;
        int j=blockIdx.y * blockDim.y + threadIdx.y;
        int el=(j*cols) + i;

        //float * ans_el=ans+el;
        for (int plane=0;plane<planes;plane++)
        {
            T * input_el=input+plane*rows*cols+el;
            T * output_el=output+plane*rows*cols+el;
            if (i<cols && j<rows)  
            {    
                int offset[2]={ j-(rows/2), i-(cols/2)};
                T nu[2] = { abs((T(offset[0]))/(T(rows/2))),
                                abs((T(offset[1]))/(T(cols/2)))
                            };



                *(output_el)=*(input_el)/((gridsf<T>(nu[0])* gridsf<T>(nu[1])));
             }
        }

    }
//namespace end
}}

using namespace mtimager;
using namespace mtimager::CasaGridSF_kernels;

void CasaGridSF::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
     
    

    if (data.params.getBoolProperty("lmPlanePostCorrection")==false)
    {    
        dim3 threadsPerBlock;
        dim3 blocks;
        int size=data.outputs[0].dim.getNoOfRows();
        int sampling=data.params.getIntProperty("sampling");
        data.deviceDescriptor->getBestThreadsAndBlocksDim(size,size,blocks,threadsPerBlock);
        switch (data.outputs[0].type)
        {
            case GAFW::GeneralImplimentation::real_float:
                gridsf<float,float> <<<blocks,threadsPerBlock,0,data.stream>>> ((float* )data.outputs->pointer,size,sampling); 
                break;
            case GAFW::GeneralImplimentation::real_double:
                gridsf<double,double> <<<blocks,threadsPerBlock,0,data.stream>>> ((double* )data.outputs->pointer,size,sampling); 
                break;
            case GAFW::GeneralImplimentation::complex_float:
                gridsf<cuFloatComplex,float> <<<blocks,threadsPerBlock,0,data.stream>>> ((cuFloatComplex* )data.outputs->pointer,size,sampling); 
                break;
            case GAFW::GeneralImplimentation::complex_double:
                gridsf<cuDoubleComplex,double> <<<blocks,threadsPerBlock,0,data.stream>>> ((cuDoubleComplex* )data.outputs->pointer,size,sampling); 
                break;
        }
    }
    
    else
    {
        if (data.params.getBoolProperty("generate")==true)
        {
            dim3 threadsPerBlock;
            dim3 blocks;
            data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
            switch (data.outputs[0].type)
            {
                case GAFW::GeneralImplimentation::real_float:
                        casacore_gridsf_correction_generate<float> <<<blocks,threadsPerBlock,0,data.stream>>> ((float* )data.outputs->pointer,data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns()); 
                        break;
                case GAFW::GeneralImplimentation::real_double:
                        casacore_gridsf_correction_generate<double> <<<blocks,threadsPerBlock,0,data.stream>>> ((double* )data.outputs->pointer,data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns()); 
                        break;
            }
        }
        else
        {
            dim3 threadsPerBlock;
            dim3 blocks;
            data.deviceDescriptor->getBestThreadsAndBlocksDim(data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns(),blocks,threadsPerBlock);
            int planes=data.inputs->dim.getTotalNoOfElements()/(data.inputs->dim.getNoOfRows()*data.inputs->dim.getNoOfColumns());
            switch (data.outputs[0].type)
            {
                case GAFW::GeneralImplimentation::real_float:
                        casacore_gridsf_correction<float> <<<blocks,threadsPerBlock,0,data.stream>>> ((float* )data.inputs->pointer,(float* )data.outputs->pointer,planes,data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns()); 
                        break;
                case GAFW::GeneralImplimentation::real_double:
                        casacore_gridsf_correction<double> <<<blocks,threadsPerBlock,0,data.stream>>> ((double* )data.inputs->pointer,(double* )data.outputs->pointer,planes,data.outputs[0].dim.getNoOfRows(),data.outputs[0].dim.getNoOfColumns()); 
                        break;
            }
            
            
        }
    }
     
}