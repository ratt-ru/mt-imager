/* SynchronousQueue.h:  SynchronousQueue template class. 
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
#ifndef SYNCHRONOUSQUEUE_H
#define	SYNCHRONOUSQUEUE_H
#include <queue>
#include <boost/thread.hpp>
namespace GAFW { namespace GPU
{
    template<class T, class Container >
    class SynchronousQueue
    {
        protected:
            std::queue<T,Container> myQueue;
            boost::mutex myMutex;
            boost::condition_variable myCondition;
            
        public:
            SynchronousQueue()
            {
                
            }
            void push(T & value)
            {
                boost::mutex::scoped_lock lock(myMutex);
                myQueue.push(value);
                myCondition.notify_all();
            }
            bool pop(T &value)
            {
                boost::mutex::scoped_lock lock(myMutex);
                if (myQueue.size()==0)
                {
                    return false;
                }
                value=myQueue.front();
                myQueue.pop();
                
                return true;
            }
            void pop_wait(T& value)
            {
                boost::mutex::scoped_lock lock(myMutex);
                
                while(myQueue.size()==0)
                {
                    this->myCondition.wait(lock);
                }
                value=myQueue.front();
                myQueue.pop();
            }
            void wait()
            {
                boost::mutex::scoped_lock lock(myMutex);
                while(myQueue.size()==0)
                {
                    this->myCondition.wait(lock);
                }
            }
            size_t size()
            {
                boost::mutex::scoped_lock lock(myMutex);
                return myQueue.size();
            }
            
    };
    

}}

#endif	/* SYNCHRONOUSQUEUE_H */

