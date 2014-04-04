/* CUDAEventPool.cpp:  Implementation of the CUDAEventPool class
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
using namespace GAFW::GPU;

CudaEventPool::CudaEventPool() 
{


}


CudaEventPool::~CudaEventPool() 
{
    /* THIS FUNCTIONS O<METIMES GIVES trouble.. for now diasbled*/
    //Clear pool and destroy all events (that are in store)
    /*
    boost::mutex::scoped_lock lock(this->myMutex);
    while(!this->store.empty())
    {
        
        cudaEvent_t * event=this->store.top();
        this->store.pop();
        cudaEventDestroy(*event);
    }*/
}
cudaEvent_t * CudaEventPool::proper_pop()
{
    boost::mutex::scoped_lock lock(this->myMutex);
    if (this->store.empty()) return NULL;
    cudaEvent_t * ret=this->store.top();
    this->store.pop();
    return ret;
}
cudaEvent_t * CudaEventPool::createEvent()
{
    cudaEvent_t * ret=new cudaEvent_t;
    checkCudaError(cudaEventCreateWithFlags(ret,cudaEventBlockingSync),"Unable to create Event");
    return ret;
}
cudaEvent_t * CudaEventPool::requestEvent()
{
    cudaEvent_t * ret=this->proper_pop();
    if (ret==NULL) return this->createEvent();
    else return ret;
}



void CudaEventPool::checkInEvent(cudaEvent_t * in)
{
    boost::mutex::scoped_lock lock(this->myMutex);
    this->store.push(in);
}
