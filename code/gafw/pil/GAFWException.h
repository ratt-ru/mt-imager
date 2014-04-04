/* GAFWException.h:  Definition of the GAFWException class    
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */

#ifndef __GAFWEXCEPTION_H__
#define	__GAFWEXCEPTION_H__
#include <exception>
#include <string>

#define GeneralException(description) GAFWException(description,this,__LINE__,__FILE__); 
#define GeneralException2(description,object) GAFWException(description,object,__LINE__,__FILE__); 
namespace GAFW
{
class GAFWException : public std::exception
{
 protected:
     std::string description;
     std::string linenumber;
     std::string file;
     std::string object_name;
     std::string object_objectname;
        
public:
    virtual const char* what() const throw();
    GAFWException(std::string desc,Identity *object_pointer, int linenumber, std::string file ) throw();
    GAFWException(std::string desc,void * object_pointer,int linenumber, std::string file ) throw();
    virtual ~GAFWException() throw();


};
} // end of namespace
#endif	/* GMFWEXCEPTION_H */

