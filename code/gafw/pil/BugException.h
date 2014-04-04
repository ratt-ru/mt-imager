/* BugException.h: Definition of the BugException class   
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

#ifndef BUGEXCEPTION_H
#define	BUGEXCEPTION_H
#define Bug(description) BugException(description,this,__LINE__,__FILE__); 
#define Bug2(description,object) BugException(description,object,__LINE__,__FILE__); 

namespace GAFW 
{

    class BugException: public GAFWException {
    public:
        virtual const char* what() const throw();
        BugException(std::string desc,Identity *object_pointer, int linenumber, std::string file ) throw();
        BugException(std::string desc,void * object_pointer,int linenumber, std::string file ) throw();
        virtual ~BugException() throw();
    };
}


#endif	/* BUGEXCEPTION_H */

