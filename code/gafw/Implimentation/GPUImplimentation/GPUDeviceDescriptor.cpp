/* GPUDeviceDescriptor.cpp:  Implementation of the GPUDeviceDescriptor class. 
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
#include <iostream>
#include <sstream>
using namespace GAFW::GPU;
using namespace GAFW::GeneralImplimentation;
using namespace std;

GPUDeviceDescriptor::GPUDeviceDescriptor(int device_no):Identity("GPUDeviceDescriptor","GPUDeviceDescriptor")  {
    
    LogFacility::init();
    
    cudaGetDeviceProperties (&this->deviceProperties, device_no);
  
}

GPUDeviceDescriptor::~GPUDeviceDescriptor() {
    
    
}



int GPUDeviceDescriptor::getBestThreadsAndBlocksDim(const int rows, const int columns, dim3 &blockDim, dim3 &threadDim)
{
    
    stringstream ss;
    ss<<"getBestThreadsAndBlocksDim(): request with rows="<< rows << " columns="<<columns;
    logDebug(execution_submission,ss.str());
    
    float maxThreads=(float)this->deviceProperties.maxThreadsPerBlock;
    float ratio=((float)columns/(float(rows)));
    threadDim.y=floor(sqrt((maxThreads)/ratio ));
    if (threadDim.y==0) threadDim.y=1;
    if (threadDim.y>maxThreads) threadDim.y=maxThreads; // TO REVIEW
    threadDim.x=floor(maxThreads/((float)threadDim.y));
    if (threadDim.x==0) threadDim.x=1;
    threadDim.z=1;
    blockDim.x=ceil(((float)columns)/((float)threadDim.x));
    if (blockDim.x==0) blockDim.x=1;
    blockDim.y=ceil(((float)rows)/((float)threadDim.y));
    if (blockDim.y==0) blockDim.y=1;
    
    blockDim.z=1;
    stringstream ss2;
    ss2<< "getBestThreadsAndBlocksDim():Answer is" 
                << " threadDim.x=" << threadDim.x 
                << " threadDim.y=" << threadDim.y
                << " threadDim.z=" << threadDim.z
                << " blockDim.x=" << blockDim.x
                << " blockDim.x=" << blockDim.y
                << " blockDim.x=" << blockDim.z ;
    
    logDebug(execution_submission,ss2.str());
    return 0;        
}
cudaDeviceProp GPUDeviceDescriptor::getDeviceProperties()
{
    return this->deviceProperties;
}
