/* MTImagerException.h:  Definition of the MTImagerException class and ImagerException 
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
#ifndef __IMAGEREXCEPTION_H__
#define	__IMAGEREXCEPTION_H__
#include <exception>
#include <string>
#include "gafw.h"
#define ImagerException(description) MTImagerException(description,this,__LINE__,__FILE__); 
#define ImagerException2(description,object) MTImagerException(description,object,__LINE__,__FILE__); 

namespace mtimager
{
class MTImagerException : public GAFW::GAFWException
{
 protected:
     std::string description;
     std::string linenumber;
     std::string file;
     std::string object_name;
     std::string object_objectName;
        
public:
    virtual const char* what() const throw();
    MTImagerException(std::string desc,GAFW::Identity *object_pointer, int linenumber, std::string file ) throw();
    MTImagerException(std::string desc,void * object_pointer,int linenumber, std::string file ) throw();
    virtual ~MTImagerException() throw();


};
} // end of namespace

#endif	/* MTIMAGEREXCEPTION_H */

