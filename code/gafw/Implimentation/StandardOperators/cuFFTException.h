/* cuFFTException.h:  Definition of the cuFFTException class. 
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
#ifndef __GPUAFWCUFFTEXCEPTION_H__
#define	__GPUAFWCUFFTEXCEPTION_H__
#include <exception>
#include <string>
#include "cufft.h"
#include "gafw.h"
#define FFTException(description,error) GAFW::GPU::StandardOperators::cuFFTException(description,error,this,__LINE__,__FILE__); 
#define FFTException2(description,error,pointer) GAFW::GPU::StandardOperators::cuFFTException(description,error,pointer,__LINE__,__FILE__); 
namespace GAFW { namespace GPU
{
    namespace StandardOperators
    { 
        class cuFFTException : public GAFW::GAFWException
        {
         protected:

             std::string description;
             std::string linenumber;
             std::string file;
             cufftResult error;
             std::string errorName;
             std::string object_name;
             std::string object_objectname;

        public:
            virtual const char* what() const throw();
            cuFFTException(std::string desc,cufftResult error,GAFW::Identity * object_pointer,int linenumber, std::string file );
            cuFFTException(std::string desc,cufftResult error,void * object_pointer,int linenumber, std::string file ) ;

            ~cuFFTException() throw();


        };
    }
}}
#endif	/* GPUMFWCUFFTEXCEPTION_H */

