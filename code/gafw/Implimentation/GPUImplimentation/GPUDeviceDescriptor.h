/* GPUDeviceDescriptor.h:  Definition of the GPUDeviceDescriptor class. 
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

#ifndef __GPUDEVICEDESCRIPTOR_H__
#define	__GPUDEVICEDESCRIPTOR_H__
#include <cuda_runtime.h>
namespace GAFW { namespace GPU 
{
class GPUDeviceDescriptor : public GAFW::LogFacility,public GAFW::Identity {
    struct cudaDeviceProp deviceProperties;
public:
    GPUDeviceDescriptor(int device_no);
    virtual ~GPUDeviceDescriptor();
    int getBestThreadsAndBlocksDim(const int rows, const int columns, dim3 &blockDim, dim3 &threadDim);
    cudaDeviceProp getDeviceProperties();
private:

};
} }

#endif	/* GPUDEVICEDESCRIPTOR_H */

