/* ZeroToCentre2DShift.cu:  CUDA implementation of the ZeroToCentre2DShift operator 
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
#include "ZeroToCentre2DShift.h"
#include "cuda_runtime.h"
#include "cuComplex.h"

namespace GAFW { namespace GPU { namespace StandardOperators
{
    namespace ZeroToCentre2DShift_kernels
    {
        #define loc_2_pos3D(loc,dim) (loc.z*dim.x*dim.y+loc.y*dim.x+loc.x)

        __device__ __inline__ int  OriginToCentre1D(int& loc,int& dim)
        {
            int newloc;
            if (dim%2)  //odd number
            {
                if (loc<=dim/2) newloc=loc+dim/2;
                else newloc=loc-(dim/2)-1;
            }
            else    //even dimension
            {
               if (loc<(dim/2)) newloc=loc+(dim/2);
               else newloc=loc-(dim/2); 
            }
            return newloc;
        }
        __device__ __inline__ int OriginToCorner1D(int & loc, int &dim)
        {
            int newloc;
            if (dim%2)  //odd number
            {
                if (loc<dim/2) newloc=loc+dim/2+1;
                else newloc=loc-(dim/2);
            }
            else    //even dimension
            {
               if (loc<(dim/2)) newloc=loc+(dim/2);
               else newloc=loc-(dim/2); 
            }
            return newloc;
        }
    
    
        template <class T>
        __global__ void ZeroToCentre2D(T * input, T * output, int3 dim);
        
        template <class T>
        __global__ void ZeroToCentre2D(T * input, T * output, int3 dim)
        {
            int2 idx=make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
            int3 loc;
            int3 newloc;

            for (loc.z=0;loc.z<dim.z;loc.z++)  ///It might be worth putting z in the inner loop
            {
                newloc.z=loc.z;
                for (loc.y=idx.y;loc.y<dim.y;loc.y+=int(gridDim.y*blockDim.y))
                {

                    newloc.y=OriginToCentre1D(loc.y,dim.y);

                    for (loc.x=idx.x;loc.x<dim.x;loc.x+=int(gridDim.x*blockDim.x))
                    {
                        //No we calculate new positions
                        newloc.x=OriginToCentre1D(loc.x,dim.x);

                        *(output+loc_2_pos3D(newloc,dim))=*(input+loc_2_pos3D(loc,dim));

                    }
                }
            }
        }
    }}}}
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GPU::StandardOperators::ZeroToCentre2DShift_kernels;
using namespace GAFW::GeneralImplimentation;
 
void ZeroToCentre2DShift::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{

     dim3 threadsPerBlock;
     dim3 blocks;
     int3 dim;
     
     dim.y=data.inputs[0].dim.getY();
     dim.x=data.inputs[0].dim.getX();
     dim.z=data.inputs[0].dim.getTotalNoOfElements()/(dim.y*dim.x);
     
     data.deviceDescriptor->getBestThreadsAndBlocksDim(dim.y,dim.x,blocks,threadsPerBlock);
     logWarn(execution,"REMINDER: To change how to run this kernel and check rows and cols for DeviceDescriptor.. WE NEED TO SUPPORT for dimensions 8000x8000");
     switch (data.inputs->type)
     {
         case complex_float:
                ZeroToCentre2D<cuComplex><<<blocks,threadsPerBlock,0,data.stream>>>((cuComplex*)data.inputs[0].pointer, (cuComplex*)data.outputs->pointer, dim);
                break;
         case complex_double:
                ZeroToCentre2D<cuDoubleComplex><<<blocks,threadsPerBlock,0,data.stream>>>((cuDoubleComplex*)data.inputs[0].pointer, (cuDoubleComplex*)data.outputs->pointer, dim);
                break;
     }         
}