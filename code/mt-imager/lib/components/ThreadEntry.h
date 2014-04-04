/* ThreadEntry.cpp: ThreadEntry template class. 
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
#ifndef IMAGER_THREADENTRY_H
#define	IMAGER_THREADENTRY_H
#include <boost/atomic.hpp>
namespace mtimager
{
    template <class T>
    class ThreadEntry
    {
        public:
        typedef void (T::*FunctionPointer)() ;
    
        protected:
            T* obj;
           FunctionPointer entryPoint;
           
        public:
            
            ThreadEntry(T* obj,FunctionPointer entryPoint):obj(obj),entryPoint(entryPoint) 
            {
                boost::atomic_thread_fence(boost::memory_order_release);
            };
            void operator()()
            {
                (obj->*entryPoint)();
            }
        
    };
};

#endif	/* THREADENTRY_H */

