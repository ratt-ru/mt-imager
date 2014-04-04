/* RingBuffer.h:  RingBuffer template class 
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
#ifndef RINGBUFFER_H
#define	RINGBUFFER_H
#include "boost/atomic.hpp"
namespace GAFW { namespace GPU
{
    template<class T>
    class RingBuffer {
    protected:
        inline size_t next(size_t & current);
        const size_t size;
        T *ring;
        boost::atomic<size_t> head_position, tail_position;
        boost::atomic<size_t> noOfElements;
        
    public:
        
        RingBuffer(size_t size);
        ~RingBuffer();
        bool push(const T & value);
        bool pop(T & value);
        
        
    };
    template <class T>  
    RingBuffer<T>::RingBuffer(size_t size):size(size),noOfElements(0)
    {
        this->ring=new T[size];
        this->tail_position=0;
        this->head_position=0;
    }
    template <class T>  
    RingBuffer<T>::~RingBuffer()
    {
        delete[] this->ring;
    }
    
    template <class T>  
    bool RingBuffer<T>::push(const T &value)
    {
        if (this->size==noOfElements.load(boost::memory_order_relaxed))
            return false;
        size_t head = head_position.load(boost::memory_order_relaxed);
        size_t next_head = next(head);
        if (next_head == tail_position.load(boost::memory_order_acquire))
                return false;
        ring[head] = value;
        head_position.store(next_head, boost::memory_order_release);
        /*
         * int old_value=noOfElements.fetch_add(1,boost::memory_order_acquire);
        if (old_value==0)
        {
            //Then we expect the pop mutex to be locked.. so we wait until it is really locked 
            //and unlock ourselves
            
            
        }*/
        
        return true;
    }
    template <class T>  
    bool RingBuffer<T>::pop(T & value)
    {
        size_t tail = this->tail_position.load(boost::memory_order_relaxed);
        if (tail == this->head_position.load(boost::memory_order_acquire))
                return false;
        value = ring[tail];
        this->tail_position.store(next(tail), boost::memory_order_release);
        return true;
    }
    template <class T>  
    inline size_t RingBuffer<T>::next(size_t & current)
    {
        return (current + 1) % this->size;
    }
        
        
        

}} ;
#endif	/* RINGBUFFER_H */

