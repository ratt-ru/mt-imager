/* SynchronousObject.h:  SynchronousObject template class. 
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

#ifndef SYNCHRONOUSOBJECT_H
#define	SYNCHRONOUSOBJECT_H
#include <boost/thread.hpp>
namespace GAFW { namespace GPU
{
    template <class T>
    class SynchronousObject
    {
        protected:
            boost::mutex myMutex;
            boost::condition_variable myCondVariable;
            T variable;
        public:
            SynchronousObject<T>(){}
            SynchronousObject<T>(T value)
            {
                variable=value;
            }
            template <class J> 
            SynchronousObject<T>& operator=(J value)
            {
                boost::mutex::scoped_lock lock(this->myMutex);
                variable=T(value);
                this->myCondVariable.notify_all();
                
                return *this;
            }
            
            template<class J>
            SynchronousObject<T>& operator+=(J value)
            {
                
                boost::mutex::scoped_lock lock(this->myMutex);
                variable+=T(value);
                this->myCondVariable.notify_all();
                return *this;
            }
            
            template<class J>
            SynchronousObject<T>& operator-=(J value)
            {
                
                boost::mutex::scoped_lock lock(this->myMutex);
                variable-=T(value);
                
                this->myCondVariable.notify_all();
                return *this;
            }
            template<class J>
            operator J()
            {    
                boost::mutex::scoped_lock(this->myMutex);
                return J(variable);
            }
            T wait_until_change(T oldvalue)  //returns the new value
            {
                boost::mutex::scoped_lock lock(this->myMutex);
                while(this->variable==oldvalue)
                {
                    this->myCondVariable.wait(lock);
                }
                return this->variable;
            }
        
    };        
}};

#endif	/* VARIABLEMANAGER_H */

