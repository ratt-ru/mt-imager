/* GAFWValidationException.h:  Header file for the definition of GAFWValidationException 
 * and macros ValidationException and ValidationException2
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

#ifndef __GEN_GMFWVALIDATIONEXCEPTION_H__
#define	__GEN_GMFWVALIDATIONEXCEPTION_H__
#include <exception>
#include <string>

#define ValidationException2(description,object_pointer) GAFWValidationException(description,object_pointer,__LINE__,__FILE__)


#define ValidationException(description) GAFW::GeneralImplimentation::GAFWValidationException(description,this,__LINE__,__FILE__);
namespace GAFW { namespace GeneralImplimentation 
{
    
class GAFWValidationException : public GAFWException
{
 protected:
     //std::string description;
     //std::string linenumber;
     //std::string file;
     //std::string object_name;
     //std::string object_nickname;
        
public:
    virtual const char* what() const throw();
    GAFWValidationException(std::string desc,void * object_pointer,int linenumber, std::string file );
    GAFWValidationException(std::string desc,Identity * object_pointer,int linenumber, std::string file );
    virtual ~GAFWValidationException() throw();


};

} }//End of namespace

#endif	/* GMFWVALIDATIONEXCEPTION_H */

