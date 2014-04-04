/* GPUArrayOperator.cpp:  Definition of the GPUCudaException class which is thrown when a CUDA error occurs.
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
#ifndef GPUMFWCUDAEXCEPTION_H
#define	GPUMFWCUDAEXCEPTION_H
#include <exception>
#include <string>
#include "cuda_runtime.h"
#define CudaException(description,error) GAFW::GPU::GPUCudaException(description,error,__LINE__,__FILE__); 
namespace GAFW { namespace GPU 
{
    class GPUCudaException : public std::exception
    {
     protected:

         std::string description;
         std::string linenumber;
         std::string file;
         cudaError_t error;

    public:
        virtual const char* what() const throw();
        GPUCudaException(std::string desc,cudaError_t error,int linenumber, std::string file ) throw();

        virtual ~GPUCudaException() throw();


    };
} }

#endif	/* GPUMFWCUDAEXCEPTION_H */

