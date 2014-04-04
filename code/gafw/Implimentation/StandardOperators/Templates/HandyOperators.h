/* HandyOperators.h:  Some CUDA device code that can be handy for the specialising 
 * the template operators
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
#ifndef __HANDYOPERATORS_H__
#define	__HANDYOPERATORS_H__
#include "SharedStructures.h"
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>
#include "gafw.h"

namespace GAFW { namespace GPU { namespace OperatorTemplates
{       //We do not support cuComplex as input
    #include "HelperCudaFunctions.hcu" 
    class InnerChi2
    {
    public:
        template<class InputType,class OutputType>
        __device__ __inline__ static void  InnerOperation(OutputType &ans,struct InputElements<InputType,2> &el)
        {
            OutputType tmp;
            tmp= (OutputType)el.inputs[0]-(OutputType)el.inputs[1];
            tmp*=tmp; // squared
            ans=tmp;
        }
    };
    
    class InnerMultiply
    {
    public:
        template<class InputType,class OutputType>
        __device__ __inline__ static void  InnerOperation(OutputType &ans,struct InputElements<InputType,2> &el)
        {
            ans= (OutputType)el.inputs[0]*(OutputType)el.inputs[1];
           
        }
    };
    class InnerOnlyCast //For reductions of 1 input
    {
    public:
        template<class InputType,class OutputType>
        __device__ __inline__ static void  InnerOperation(OutputType &ans,struct InputElements<InputType,1> &el)
        {
            ans= (OutputType)el.inputs[0];
           
        }
   
    };
    
    
   class OuterOperationAdd
   {
    public:
        template <class T>
        __device__ static __inline__ void OuterOperation(T& acc,T & num)
        {
            acc+=num;
        }
        __device__ static __inline__ void OuterOperation(cuComplex& acc,cuComplex & num)
        {
            acc.x+=num.x;
            acc.y+=num.y;
        }
        __device__  static __inline__ void OuterOperation(cuDoubleComplex& acc,cuDoubleComplex & num)
        {
            acc.x+=num.x;
            acc.y+=num.y;
        }
        template <class T>
        __device__ static __inline__ void setIdentity(T &value)
        {
            value=0;
        }
        __device__ static __inline__ void setIdentity(cuComplex &value)
        {
            value=make_float2(0.0f,0.0f);
        }
        __device__ static __inline__ void setIdentity(cuDoubleComplex &value)
        {
            value=make_double2(0.0,0.0);
        }
};
class OuterOperationMax
   {   //defined only for real ... no complex inputs
    public:
        template <class T>
        __device__ static __inline__ void OuterOperation(T& acc,T & num)
        {
            if (num>acc) acc=num;
        }
        template <class T>
        __device__ static inline void setIdentity(T&value)
        {
            value=0; //only good for unsigned
        }
        __device__ static inline void setIdentity(float &value)
        {
            value=-FLT_MAX;
        }
        __device__ static inline void setIdentity(double &value)
        {
            value=-DBL_MAX;
        }
        __device__ static inline void setIdentity(int &value)
        {
            value=-INT_MAX;
        }
        __device__ static inline void setIdentity(long int &value)
        {
            value=-LONG_MAX;
        }
        __device__ static inline void setIdentity(short int &value)
        {
            value=-SHRT_MAX;
        }

};


template <int NoOfInputs>
class SumSquares
{
    public:
        //no support for complex 
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,NoOfInputs> &el)
        {
            ans=zero<OutputType>();
    #pragma unroll
            for (int x=0;x<NoOfInputs;x++)
            {
                ans+=(OutputType)el.inputs[x]*(OutputType)el.inputs[x];
            }
        }
};
//Ensure efficibnecy for NoOfinpits=1
template <>
class SumSquares<1>
{
    public:
        //no support for complex 
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,1> &el)
        {
            ans=(OutputType)el.inputs[0]*(OutputType)el.inputs[0];
        }
};

template <int NoOfInputs>
class JustSum  //This reduces to a cast for No Of inputs=1 
{
   public:
       
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,NoOfInputs> &el)
        {
            ans=zero<OutputType>();
    #pragma unroll
            for (int x=0;x<NoOfInputs;x++)
            {
                ans+=(OutputType)el.inputs[x];
            }
        }
        //we have to complicate life for complex numbers but for now we do not support
};
class InnerOperatorTakeDifference  //Supprts onkly two inputs
{
   public:
       //For now we do not support complex
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,2> &el)
        {
            ans=el.inputs[0]-el.inputs[1];
        }
                
};

class InnerOperatorAbs  //Supprts onkly two inputs
{
   public:
       //For now we do not support complex
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,1> &el)
        {
                ans=abs(el.inputs[0]);
            
            
        }
                
};





//We ensure efficiency for NoOfInputs=1
template <>
class JustSum<1>  
{
   public:
        template<class InputType,class OutputType>
        __device__ __inline__ static void InnerOperation(OutputType &ans,struct InputElements<InputType,1> &el)
        {
            ans=(OutputType)el.inputs[0];
            
        }
        
};

class FinalOperationDoNothing
{
public:
    static  void submit(GAFW::GPU::GPUSubmissionData &data,uint &no_of_elements,int &no_of_planes) 
    {}    ;
};

class DivideOperation
{
public:
    template<class InputType, class OutputType>
    __device__ static void ScalarOperation(OutputType &ans,InputType &value,OutputType &divider)
    {
        ans=OutputType(value)/divider;
    }

};
class SquareRootOperation
{
public:
    template<class InputType, class OutputType>
    __device__ static __inline__ void ScalarOperation(OutputType &ans,InputType &value,OutputType &divider)
    {
        ans=1.0/sqrt(OutputType(value));
    }

};




template <class InputType,class OutputType,class ScalarOperationDefenition>
__global__ void GeneralScalarOperation(InputType * input,OutputType scalar, OutputType *output,int elements)
{
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int elno=idx;elno<elements;elno+=gridDim.x*blockDim.x)
    {
        InputType input_value=*(input+elno);
        ScalarOperationDefenition::ScalarOperation(*(output+elno),input_value,scalar);
    }
    
}




class FinalOperationDivideByNoOfElements
{
private:
    template<class T>
    static void really_submit(GAFW::GPU::GPUSubmissionData &data,uint &no_of_elements,int &no_of_planes)
    {
          dim3 threadsPerBlock;
          dim3 blocks;
          threadsPerBlock.x=1024;
          threadsPerBlock.y=1;
          threadsPerBlock.z=1;
          blocks.x=32;
          blocks.y=1;
          blocks.z=1;
          
          GeneralScalarOperation<T,T, DivideOperation> <<<blocks,threadsPerBlock,0,data.stream>>> ((T*) data.outputs[0].pointer,(T)no_of_elements, (T*) data.outputs[0].pointer,(T)no_of_planes);
    }
public:
    static void submit(GAFW::GPU::GPUSubmissionData &data,uint &no_of_elements,int &no_of_planes) 
    {      
          switch (data.outputs[0].type)
          {
              case GAFW::GeneralImplimentation::real_float:
                  really_submit<float>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_double:
                  really_submit<double>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_int:
                  really_submit<int>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_uint:
                  really_submit<unsigned int>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_longint:
                  really_submit<long>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_ulongint:
                  really_submit<unsigned long>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_shortint:
                  really_submit<short>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_ushortint:
                  really_submit<unsigned short>(data,no_of_elements,no_of_planes);
                  break;
              default:
                  throw GAFW::GeneralException2("BUG::Don't know what to do with given type!!",(void*)NULL);
          
          }
    };
   
};

class FinalOperationSquareRoot
{
private:
    template<class T>
    static void really_submit(GAFW::GPU::GPUSubmissionData &data,uint &no_of_elements,int &no_of_planes)
    {
          dim3 threadsPerBlock;
          dim3 blocks;
          threadsPerBlock.x=1024;
          threadsPerBlock.y=1;
          threadsPerBlock.z=1;
          blocks.x=32;
          blocks.y=1;
          blocks.z=1;
          
          GeneralScalarOperation<T,T, SquareRootOperation> <<<blocks,threadsPerBlock,0,data.stream>>> ((T*) data.outputs[0].pointer,(T)no_of_elements, (T*) data.outputs[0].pointer,(T)no_of_planes);
    }
public:
    static void submit(GAFW::GPU::GPUSubmissionData &data,uint &no_of_elements,int &no_of_planes) 
    {      
          switch (data.outputs[0].type)
          {
              case GAFW::GeneralImplimentation::real_float:
                  really_submit<float>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::GeneralImplimentation::real_double:
                  really_submit<double>(data,no_of_elements,no_of_planes);
                  break;
/*
              case GAFW::real_int:
                  really_submit<int>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::real_uint:
                  really_submit<unsigned int>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::real_longint:
                  really_submit<long>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::real_ulongint:
                  really_submit<unsigned long>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::real_shortint:
                  really_submit<short>(data,no_of_elements,no_of_planes);
                  break;
              case GAFW::real_ushortint:
                  really_submit<unsigned short>(data,no_of_elements,no_of_planes);
                  break;
*/
               default:
                  throw GAFW::GeneralException2("BUG::Don't know what to do with given type!!",(void*)NULL);
          
          }
    };
   
};

}}};





#endif	/* HANDYOPERATORS_H */

