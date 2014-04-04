/* AccumulatedSum.cu:  CUDA implementation of the AccumulatedSum operator 
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
#include "GPUafw.h"
#include "AccumulatedSum.h"

#define ELEMENTS_PER_BLOCK 1024
#define LOG2_ELEMENTS_PER_BLOCK 10
#define INTERNAL_A4 4
#define LOG2_INTERNAL_A4 2
#define TWO_POW_INTERNAL_A4 16

namespace GAFW { namespace GPU { namespace StandardOperators { namespace AccumulatedSum_kernels
{
    template<class A>
    __device__ __inline__ void swap_and_add(A &num0,A &num1);
    
    __device__ __inline__ void set_to_zero(float4 &var);
    __device__ __inline__ void set_to_zero(int4 &var);
    __device__ __inline__ void set_to_zero(uint4 &var);
    
    template<class A,class A4>
    __device__ __inline__ void accumulate_load_in_registries(int &id, A4 myNumbers[INTERNAL_A4],A4* input,  int no_of_elements);
    
    template<class A,class A4>
    __global__ void accumulate_part1(A4 *input, A4*output,int no_of_elements);
    
    template<class A,class A4>
    __device__ __inline__ void accumulate_final_tail(A4 * input,A4 *output, int no_of_elements,A temp[ELEMENTS_PER_BLOCK]);
    
    template<class A,class A4>
    __global__ void accumulate_final_part(A4 * input, A4* output,int no_of_elements);
    
    template<class A> 
    __global__ void accumulate_middle_simple(A* data,int offset,int no_of_elements);



template<class A>
__device__ __inline__ void swap_and_add(A &num0,A &num1)
{
    A tempNumber;
     tempNumber=num0;
     num0=num1;
     num1+=tempNumber;

}

__device__ __inline__ void set_to_zero(float4 &var)
{
    var=make_float4(0,0,0,0);
}
__device__ __inline__ void set_to_zero(int4 &var)
{
    var=make_int4(0,0,0,0);
}
__device__ __inline__ void set_to_zero(uint4 &var)
{
    var=make_uint4(0,0,0,0);
}


template<class A,class A4>
__device__ __inline__ void accumulate_load_in_registries(int &id, A4 myNumbers[INTERNAL_A4],A4* input,  int no_of_elements)
{
    int soffset=blockDim.x*blockIdx.x*INTERNAL_A4; 

#pragma unroll
     for (int i=0;i<INTERNAL_A4;i++)
     {
         myNumbers[i]=input[soffset+INTERNAL_A4*id+i];
     }
  
}
template <class A,class A4>
__device__ __inline__ void accumulate_save_from_registries(int &id, A4 myNumbers[INTERNAL_A4], A4* output, int no_of_elements)
{
    int soffset=blockDim.x*blockIdx.x*INTERNAL_A4; 
    //Thsi fiunction is only called in blocks  
    
#pragma unroll
     for (int i=0;i<INTERNAL_A4;i++)
     {
         output[soffset+INTERNAL_A4*id+i]=myNumbers[i];
     }
 
     
}  


template<class A,class A4>
__global__ void accumulate_part1(A4 *input, A4*output,int no_of_elements)
{
     __shared__ A temp[ELEMENTS_PER_BLOCK]; 
    A4 myNumbers[INTERNAL_A4];
    
     
     int id = threadIdx.x;  
    
    accumulate_load_in_registries<A,A4>(id, myNumbers,input, no_of_elements);
   // if ((blockIdx.x==0)&&(threadIdx.x==0)) printf(" %d\n", myNumbers[0].w);
#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].y+=myNumbers[i].x;
#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].w+=myNumbers[i].z;
#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].w+=myNumbers[i].y; 
    myNumbers[1].w+=myNumbers[0].w;
    myNumbers[3].w+=myNumbers[2].w;
    myNumbers[3].w+=myNumbers[1].w;

     
     temp[id]=myNumbers[INTERNAL_A4-1].w;
     
     
     int offset2=1;
     for (int d = ELEMENTS_PER_BLOCK>>1; d > 0; d >>= 1)                    // build sum in place up the tree  
     {  
        __syncthreads();  
        if (id < d)  
        {  
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;
             temp[bi] += temp[ai];  
             
        }  
         offset2 *= 2;
       
     }
     __syncthreads();
     myNumbers[INTERNAL_A4-1].w=temp[id];
     
     
     accumulate_save_from_registries<A,A4>(id, myNumbers, output, no_of_elements);
}
template<class A,class A4>
__device__ void accumulate_final_tail(A4 * input,A4 *output, int no_of_elements,A temp[ELEMENTS_PER_BLOCK])
{
    //the last block of the final block needs to do all the process for the "tail" 
    //of the series
    int id = threadIdx.x;  
    int soffset=blockDim.x*blockIdx.x*INTERNAL_A4; 
    A4 myNumbers[INTERNAL_A4];
    
#pragma unroll
     for (int i=0;i<INTERNAL_A4;i++)
     {
        if (((soffset+INTERNAL_A4*id+i+1)*4)<=no_of_elements)
         myNumbers[i]=input[soffset+INTERNAL_A4*id+i];
        else
        {
            set_to_zero(myNumbers[i]);
            if (((soffset+INTERNAL_A4*id+i)*4)<no_of_elements)
                myNumbers[i].x=*(((A*)input)+(soffset+INTERNAL_A4*id+i)*4);
            
            if (((soffset+INTERNAL_A4*id+i)*4+1)<no_of_elements)
                myNumbers[i].y=*(((A*)input)+(soffset+INTERNAL_A4*id+i)*4+1);
            
            if (((soffset+INTERNAL_A4*id+i)*4+2)<no_of_elements)
                myNumbers[i].z=*(((A*)input)+(soffset+INTERNAL_A4*id+i)*4+2);
            
            if (((soffset+INTERNAL_A4*id+i)*4+3)<no_of_elements)
                myNumbers[i].w=*(((A*)input)+(soffset+INTERNAL_A4*id+i)*4+3);
            
        }
     }

#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].y+=myNumbers[i].x;
#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].w+=myNumbers[i].z;
#pragma unroll     
     for (int i=0;i<INTERNAL_A4;i++) myNumbers[i].w+=myNumbers[i].y; 

    myNumbers[1].w+=myNumbers[0].w;
    myNumbers[3].w+=myNumbers[2].w;
    myNumbers[3].w+=myNumbers[1].w;
     
     temp[id]=myNumbers[INTERNAL_A4-1].w;
     
     
     int offset2=1;
     for (int d = ELEMENTS_PER_BLOCK>>1; d > 0; d >>= 1)                    // build sum in place up the tree  
     {   
        __syncthreads();  
        if (id < d)  
        {  
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;
             temp[bi] += temp[ai];  
         }  
         offset2 *= 2;
     }
     if (id == 0) { temp[ELEMENTS_PER_BLOCK - 1] = *(((A*)output)+no_of_elements-1); } 
     
     for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) // traverse down tree & build scan  
    {  
         offset2 >>= 1;  
         __syncthreads();  
         if (id < d)                       
         {    
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;  
    A t = temp[ai];  
    temp[ai] = temp[bi];  
    temp[bi] += t;   
          }  
    }  
     __syncthreads(); 
     myNumbers[INTERNAL_A4-1].w=temp[id];
     
     
     swap_and_add(myNumbers[1].w,myNumbers[3].w);
     swap_and_add(myNumbers[0].w,myNumbers[1].w);
     swap_and_add(myNumbers[2].w,myNumbers[3].w);
    
#pragma unroll
    for (int i=0;i<INTERNAL_A4;i++)
    {
        
    swap_and_add(myNumbers[i].y,myNumbers[i].w);   
    swap_and_add(myNumbers[i].x,myNumbers[i].y);   
    swap_and_add(myNumbers[i].z,myNumbers[i].w);   
    
   }

#pragma unroll
     for (int i=0;i<INTERNAL_A4;i++)
     {
        if (((soffset+INTERNAL_A4*id+i+1)*4)<=no_of_elements)
         output[soffset+INTERNAL_A4*id+i]=myNumbers[i];
        else
        {
           // set_to_zero(myNumbers[i]);
            if (((soffset+INTERNAL_A4*id+i)*4)<no_of_elements)
                *(((A*)output)+(soffset+INTERNAL_A4*id+i)*4)=myNumbers[i].x;
            
            if (((soffset+INTERNAL_A4*id+i)*4+1)<no_of_elements)
                *(((A*)output)+(soffset+INTERNAL_A4*id+i)*4+1)=myNumbers[i].y;
            
            if (((soffset+INTERNAL_A4*id+i)*4+2)<no_of_elements)
                *(((A*)output)+(soffset+INTERNAL_A4*id+i)*4+2)=myNumbers[i].z;
            
            if (((soffset+INTERNAL_A4*id+i)*4+3)<no_of_elements)
                *(((A*)output)+(soffset+INTERNAL_A4*id+i)*4+3)=myNumbers[i].w;
            
        }
     }
     
}  



template <class A,class A4> 
__global__  void accumulate_final_part(A4 * input, A4* output,int no_of_elements)
{
     __shared__ A temp[ELEMENTS_PER_BLOCK]; 
    
     if (blockIdx.x==(gridDim.x-1))
         if (no_of_elements%(blockDim.x*4*ELEMENTS_PER_BLOCK))
         {
             accumulate_final_tail(input,output, no_of_elements,temp);
             return;
         }
     
     A4 myNumbers[INTERNAL_A4];
    
    
    int id = threadIdx.x;  
    
    accumulate_load_in_registries<A,A4>(id, myNumbers,output,  no_of_elements);

    temp[id]=myNumbers[INTERNAL_A4-1].w;
    __syncthreads();
    int offset2=ELEMENTS_PER_BLOCK;
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) // traverse down tree & build scan  
    {  
         offset2 >>= 1;  
         __syncthreads();  
         if (id < d)                       
         {    
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;  
    A t = temp[ai];  
    temp[ai] = temp[bi];  
    temp[bi] += t;   
          }  
    }  
     __syncthreads(); 
     myNumbers[INTERNAL_A4-1].w=temp[id];

     
     
     swap_and_add(myNumbers[1].w,myNumbers[3].w);
     swap_and_add(myNumbers[0].w,myNumbers[1].w);
     swap_and_add(myNumbers[2].w,myNumbers[3].w);
             

    
#pragma unroll
    for (int i=0;i<INTERNAL_A4;i++)
    {
        swap_and_add(myNumbers[i].y,myNumbers[i].w);
        swap_and_add(myNumbers[i].x,myNumbers[i].y);
        swap_and_add(myNumbers[i].z,myNumbers[i].w);
        
    
    }
    accumulate_save_from_registries<A,A4>(id, myNumbers, output, no_of_elements);
}

template<class A> 
__global__ void accumulate_middle_simple(A* data,int offset,int no_of_elements)
{
      __shared__ A temp[ELEMENTS_PER_BLOCK];
      int id=threadIdx.x;
      int myNumberPosition=(id+1)*offset-1;
      //printf ("%d fff\n ",myNumberPosition);
      if (myNumberPosition<no_of_elements)
          temp[id]=data[myNumberPosition];
      else 
          temp[id]=data[no_of_elements-1];
      
     
     
     int offset2=1;
     for (int d = ELEMENTS_PER_BLOCK>>1; d > 0; d >>= 1)                    // build sum in place up the tree  
     {   
        __syncthreads();  
        if (id < d)  
        {  
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;
             temp[bi] += temp[ai];  
         }  
         offset2 *= 2;
     }
     if (id == 0) { temp[ELEMENTS_PER_BLOCK - 1] = 0; } // clear the last element
    
     for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) // traverse down tree & build scan  
    {  
         offset2 >>= 1;  
         __syncthreads();  
         if (id < d)                       
         {    
             int ai = offset2*(2*id+1)-1;  
             int bi = offset2*(2*id+2)-1;  
    A t = temp[ai];  
    temp[ai] = temp[bi];  
    temp[bi] += t;   
          }  
    }  
     __syncthreads(); 
      if (myNumberPosition<no_of_elements)
          data[myNumberPosition]=temp[id];
      else if (myNumberPosition<(no_of_elements+offset))
          data[no_of_elements-1]=temp[id];
     
}

// End of namespace

} } }};

using namespace GAFW::GPU::StandardOperators::AccumulatedSum_kernels;
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;

void AccumulatedSum::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    int no_of_elements=data.inputs[0].dim.getTotalNoOfElements();
    
    //A function to put the array values all the a specific value exits but is not streamed
    // we use a kernel instead
    //We treat the array as a vector no_of_floats long
    dim3 threadsPerBlock;
    threadsPerBlock.x=1024;
    threadsPerBlock.y=1;
    threadsPerBlock.z=1;
    
    dim3 blocks;
    blocks.x=no_of_elements/(4096*INTERNAL_A4);
    if (no_of_elements%(4096*INTERNAL_A4))
        blocks.x++; 
    blocks.y=1;
    blocks.z=1;
   
    dim3 blocksfinal=blocks;
   
    
    //data.deviceDescriptor->getBestThreadsAndBlocksDim(1,1024,blocks,threadsPerBlock);
    cudaEventRecord(*data.startEvent,data.stream);
    switch (data.inputs[0].type)
    {
        case real_int:
            accumulate_part1<int,int4> <<<blocks,threadsPerBlock,0,data.stream>>> ((int4 *)data.inputs[0].pointer,(int4*)data.outputs[0].pointer,no_of_elements);
            accumulate_middle_simple<int><<<1,1024,0,data.stream>>>((int*)data.outputs[0].pointer,4096*INTERNAL_A4,no_of_elements);
            accumulate_final_part <int,int4> <<<blocksfinal,threadsPerBlock,0,data.stream>>> ((int4 *)data.inputs[0].pointer,(int4*)data.outputs[0].pointer,no_of_elements);
            break;
        case real_uint:
            accumulate_part1<uint,uint4> <<<blocks,threadsPerBlock,0,data.stream>>> ((uint4 *)data.inputs[0].pointer,(uint4*)data.outputs[0].pointer,no_of_elements);
            accumulate_middle_simple<uint><<<1,1024,0,data.stream>>>((uint*)data.outputs[0].pointer,4096*INTERNAL_A4,no_of_elements);
            accumulate_final_part <uint,uint4> <<<blocksfinal,threadsPerBlock,0,data.stream>>> ((uint4 *)data.inputs[0].pointer,(uint4*)data.outputs[0].pointer,no_of_elements);
            break;
        case real_float:
            accumulate_part1<float,float4> <<<blocks,threadsPerBlock,0,data.stream>>> ((float4 *)data.inputs[0].pointer,(float4*)data.outputs[0].pointer,no_of_elements);
            accumulate_middle_simple<float><<<1,1024,0,data.stream>>>((float*)data.outputs[0].pointer,4096*INTERNAL_A4,no_of_elements);
            accumulate_final_part <float,float4> <<<blocksfinal,threadsPerBlock,0,data.stream>>> ((float4 *)data.inputs[0].pointer,(float4*)data.outputs[0].pointer,no_of_elements);
            break;
    }
    cudaEventRecord(*data.endEvent,data.stream);
    data.endEventRecorded=true;
}

