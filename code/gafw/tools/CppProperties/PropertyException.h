/* PropertyException.h: Header file for the PropertyException class, part of the GAFW CPPProperties Tool     
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
#ifndef __PROPERTYEXCEPTION_H__
#define	__PROPERTYEXCEPTION_H__
#include <string>

namespace GAFW {  namespace Tools { namespace CppProperties
{
    class PropertyException: public std::exception 
    {
        std::string * whatReturn; 
        std::string propertyName;
        std::string reason;
        std::string function;    
    public:
        PropertyException(std::string function,std::string name,std::string reason) throw();
        PropertyException(const PropertyException& orig) throw();
        ~PropertyException() throw();
        virtual const char* what() const throw();
    };
} } }
#endif	/* PROPERTYEXCEPTION_H */

