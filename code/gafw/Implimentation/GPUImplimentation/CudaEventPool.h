/* CUDAEventPool.h: Definition of the CUDAEventPool class 
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

#ifndef __CUDAEVENTPOOL_H__
#define	__CUDAEVENTPOOL_H__
#include <cuda_runtime.h>
#include <boost/thread.hpp>
#include <stack>
namespace GAFW { namespace GPU
{
    class CudaEventPool 
    {
    private:

        CudaEventPool(const CudaEventPool& orig){};
    protected:
        std::stack<cudaEvent_t *> store;
        boost::mutex myMutex;
        cudaEvent_t * proper_pop();
        cudaEvent_t * createEvent();
    public:

        CudaEventPool();
        virtual ~CudaEventPool();
        cudaEvent_t * requestEvent();
        void checkInEvent(cudaEvent_t *);
    };
}};
#endif	/* CUDAEVENTPOOL_H */

