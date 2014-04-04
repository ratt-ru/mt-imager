/* StoreManager.h:  StoreManager template class. 
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
#ifndef __STOREMANAGER_H__
#define	__STOREMANAGER_H__
#include "cuComplex.h"
//helper class for GPUDataStore
namespace GAFW { namespace GPU
{
    
    
     class StoreManager
    {
     public:
      //   template <class TcomplexInput,class Toutput> static inline Toutput  cast(std::complex<TcomplexInput> &value);
         
             virtual size_t getSizeOfElement()=0;
             virtual void getElement(unsigned int i,float &element)=0;
             virtual void getElement(unsigned int i,double &element)=0;
             virtual void getElement(unsigned int i,std::complex<double> &element)=0;
             virtual void getElement(unsigned int i,std::complex<float> &element)=0;
             virtual void getElement(unsigned int i,int &element)=0;
             virtual void getElement(unsigned int i,unsigned int &element)=0;
             virtual void getElement(unsigned int i,long int &element)=0;
             virtual void getElement(unsigned int i,unsigned long int &element)=0;
             virtual void getElement(unsigned int i,short int &element)=0;
             virtual void getElement(unsigned int i,unsigned short int &element)=0;
             virtual void getElement(unsigned int i,GAFW::SimpleComplex<float> &element)=0;
             virtual void getElement(unsigned int i,GAFW::SimpleComplex<double> &element)=0;
             virtual void *getPositionPointer(unsigned int i)=0;
             
             virtual void setElement(unsigned int i,float &element)=0;
             virtual void setElement(unsigned int i,double &element)=0;
             virtual void setElement(unsigned int i,std::complex<double> &element)=0;
             virtual void setElement(unsigned int i,std::complex<float> &element)=0;
             virtual void setElement(unsigned int i,int &element)=0;
             virtual void setElement(unsigned int i,unsigned int &element)=0;
             virtual void setElement(unsigned int i,long int &element)=0;
             virtual void setElement(unsigned int i,unsigned long int &element)=0;
             virtual void setElement(unsigned int i,short int &element)=0;
             virtual void setElement(unsigned int i,unsigned short int &element)=0;
             virtual void setElement(unsigned int i,GAFW::SimpleComplex<float> &element)=0;
             virtual void setElement(unsigned int i,GAFW::SimpleComplex<double> &element)=0;
             
             
             
             
     };
     
     template <class T>
     class NumberHolder
     {
         public:
         T & num;
         //We assume T is not complex
         NumberHolder(T & num):num(num) {}
         inline T &  getReal()
         {
           return num;  
         }
         inline T getImag()
         {
             return 0;
         }
         inline void setReal(double number) { num=number; };
         inline void setReal(float number) { num=number; };
         inline void setReal(int number) { num=number; };
         inline void setReal(unsigned int number) { num=number; };
         inline void setReal(short int number) { num=number; };
         inline void setReal(unsigned short int number) { num=number; };
         inline void setReal(long int number) { num=number; };
         inline void setReal(unsigned long int number) { num=number; };
         inline void setImag(double number) {  };
         inline void setImag(float number) { };
         inline void setImag(int number) {  };
         inline void setImag(unsigned int number) {  };
         inline void setImag(short int number) {  };
         inline void setImag(unsigned short int number) {  };
         inline void setImag(long int number) {  };
         inline void setImag(unsigned long int number) {  };
     };
     
     template<>
     class NumberHolder<std::complex<double> >
     {
         public:
         std::complex<double> & num;
         //We assume T is not complex
         NumberHolder(std::complex<double> & num):num(num) {}
         inline double getReal()
         {
           return num.real();  
         }
         inline double getImag()
         {
             return num.imag();
         }
         inline void setReal(double number) { num.real(number); };
         inline void setReal(float number) { num.real(number); };
         inline void setReal(int number) { num.real(number); };
         inline void setReal(unsigned int number) { num.real(number); };
         inline void setReal(short int number) { num.real(number); };
         inline void setReal(unsigned short int number) { num.real(number); };
         inline void setReal(long int number) { num.real(number); };
         inline void setReal(unsigned long int number) { num.real(number); };
         inline void setImag(double number) { num.imag(number); };
         inline void setImag(float number) { num.imag(number); };
         inline void setImag(int number) { num.imag(number); };
         inline void setImag(unsigned int number) { num.imag(number); };
         inline void setImag(short int number) { num.imag(number); };
         inline void setImag(unsigned short int number) { num.imag(number); };
         inline void setImag(long int number) { num.imag(number); };
         inline void setImag(unsigned long int number) { num.imag(number); };
         
     };
     template<>
     class NumberHolder<std::complex<float> >
     {
         public:
         std::complex<float> & num;
         //We assume T is not complex
         NumberHolder(std::complex<float> & num):num(num) {}
         inline float getReal()
         {
           return num.real();  
         }
         inline float getImag()
         {
             return num.imag();
         }
         inline void setReal(double number) { num.real(number); };
         inline void setReal(float number) { num.real(number); };
         inline void setReal(int number) { num.real(number); };
         inline void setReal(unsigned int number) { num.real(number); };
         inline void setReal(short int number) { num.real(number); };
         inline void setReal(unsigned short int number) { num.real(number); };
         inline void setReal(long int number) { num.real(number); };
         inline void setReal(unsigned long int number) { num.real(number); };
         inline void setImag(double number) { num.imag(number); };
         inline void setImag(float number) { num.imag(number); };
         inline void setImag(int number) { num.imag(number); };
         inline void setImag(unsigned int number) { num.imag(number); };
         inline void setImag(short int number) { num.imag(number); };
         inline void setImag(unsigned short int number) { num.imag(number); };
         inline void setImag(long int number) { num.imag(number); };
         inline void setImag(unsigned long int number) { num.imag(number); };
         
     };
     template<>
     class NumberHolder<cuFloatComplex >
     {
         public:
         cuFloatComplex & num;
         //We assume T is not complex
         NumberHolder(cuFloatComplex & num):num(num) {}
         inline float getReal()
         {
           return num.x;  
         }
         inline float getImag()
         {
             return num.y;
         }
         inline void setReal(double number) { num.x=number; };
         inline void setReal(float number) { num.x=number; };
         inline void setReal(int number) { num.x=number; };
         inline void setReal(unsigned int number) { num.x=number; };
         inline void setReal(short int number) { num.x=number; };
         inline void setReal(unsigned short int number) { num.x=number; };
         inline void setReal(long int number) { num.x=number; };
         inline void setReal(unsigned long int number) { num.x=number; };
         inline void setImag(double number) { num.y=number; };
         inline void setImag(float number) { num.y=number; };
         inline void setImag(int number) { num.y=number; };
         inline void setImag(unsigned int number) { num.y=number; };
         inline void setImag(short int number) { num.y=number; };
         inline void setImag(unsigned short int number) { num.y=number; };
         inline void setImag(long int number) { num.y=number; };
         inline void setImag(unsigned long int number) { num.y=number; };
         
     };
     template<>
     class NumberHolder<cuDoubleComplex >
     {
         public:
         cuDoubleComplex & num;
         //We assume T is not complex
         NumberHolder(cuDoubleComplex & num):num(num) {}
         inline float getReal()
         {
           return num.x;  
         }
         inline float getImag()
         {
             return num.y;
         }
         inline void setReal(double number) { num.x=number; };
         inline void setReal(float number) { num.x=number; };
         inline void setReal(int number) { num.x=number; };
         inline void setReal(unsigned int number) { num.x=number; };
         inline void setReal(short int number) { num.x=number; };
         inline void setReal(unsigned short int number) { num.x=number; };
         inline void setReal(long int number) { num.x=number; };
         inline void setReal(unsigned long int number) { num.x=number; };
         inline void setImag(double number) { num.y=number; };
         inline void setImag(float number) { num.y=number; };
         inline void setImag(int number) { num.y=number; };
         inline void setImag(unsigned int number) { num.y=number; };
         inline void setImag(short int number) { num.y=number; };
         inline void setImag(unsigned short int number) { num.y=number; };
         inline void setImag(long int number) { num.y=number; };
         inline void setImag(unsigned long int number) { num.y=number; };
         
     };
       template<>
     class NumberHolder<GAFW::SimpleComplex<float> >
     {
         public:
         GAFW::SimpleComplex<float> & num;
         //We assume T is not complex
         NumberHolder(GAFW::SimpleComplex<float> & num):num(num) {}
         inline float getReal()
         {
           return num.real;  
         }
         inline float getImag()
         {
             return num.imag
                     ;
         }
         inline void setReal(double number) { num.real=number; };
         inline void setReal(float number) { num.real=number; };
         inline void setReal(int number) { num.real=number; };
         inline void setReal(unsigned int number) { num.real=number; };
         inline void setReal(short int number) { num.real=number; };
         inline void setReal(unsigned short int number) { num.real=number; };
         inline void setReal(long int number) { num.real=number; };
         inline void setReal(unsigned long int number) { num.real=number; };
         inline void setImag(double number) { num.imag=number; };
         inline void setImag(float number) { num.imag=number; };
         inline void setImag(int number) { num.imag=number; };
         inline void setImag(unsigned int number) { num.imag=number; };
         inline void setImag(short int number) { num.imag=number; };
         inline void setImag(unsigned short int number) { num.imag=number; };
         inline void setImag(long int number) { num.imag=number; };
         inline void setImag(unsigned long int number) { num.imag=number; };
         
     };
     template<>
     class NumberHolder<GAFW::SimpleComplex<double> >
     {
         public:
         GAFW::SimpleComplex<double> & num;
         //We assume T is not complex
         NumberHolder(GAFW::SimpleComplex<double> & num):num(num) {}
         inline float getReal()
         {
           return num.real;  
         }
         inline float getImag()
         {
             return num.imag;
         }
         inline void setReal(double number) { num.real=number; };
         inline void setReal(float number) { num.real=number; };
         inline void setReal(int number) { num.real=number; };
         inline void setReal(unsigned int number) { num.real=number; };
         inline void setReal(short int number) { num.real=number; };
         inline void setReal(unsigned short int number) { num.real=number; };
         inline void setReal(long int number) { num.real=number; };
         inline void setReal(unsigned long int number) { num.real=number; };
         inline void setImag(double number) { num.imag=number; };
         inline void setImag(float number) { num.imag=number; };
         inline void setImag(int number) { num.imag=number; };
         inline void setImag(unsigned int number) { num.imag=number; };
         inline void setImag(short int number) { num.imag=number; };
         inline void setImag(unsigned short int number) { num.imag=number; };
         inline void setImag(long int number) { num.imag=number; };
         inline void setImag(unsigned long int number) { num.imag=number; };
         
     };
   
     
     
     
     
     
     
     template <class Tinput,class Toutput> inline Toutput cast_internal(Tinput &value)
     {
         Toutput outvalue;
         NumberHolder<Tinput> input(value);
         NumberHolder <Toutput> output(outvalue);
         output.setReal(input.getReal());
         output.setImag(input.getImag());
         return outvalue;
     }
     
     
     
     
     
    template<class T> class StoreManagerProper:public StoreManager
    {
        private:
            void *&store;
        public:
             StoreManagerProper(void *&store);
             size_t getSizeOfElement();
             template <class U> inline void internal_getElement(unsigned int i,U &element)
             {
                     element=cast_internal<T,U>(*(((T*)this->store)+i));
             }
             template <class U> inline void internal_setElement(unsigned int i,U &element)
             {
                *(((T*)this->store)+i)=cast_internal<U,T>(element);
             }
             virtual void *getPositionPointer(unsigned int i)
             {
                 return (void*)(((T*)this->store)+i);
             }
             virtual inline void getElement(unsigned int i,float &element);
             virtual inline void getElement(unsigned int i,double &element);
             virtual inline void getElement(unsigned int i,std::complex<double> &element);
             virtual inline void getElement(unsigned int i,std::complex<float> &element);
             virtual inline void getElement(unsigned int i,int &element);
             virtual inline void getElement(unsigned int i,unsigned int &element);
             virtual inline void getElement(unsigned int i,long int &element);
             virtual inline void getElement(unsigned int i,unsigned long int &element);
             virtual inline void getElement(unsigned int i,short int &element);
             virtual inline void getElement(unsigned int i,unsigned short int &element);
             virtual void getElement(unsigned int i,GAFW::SimpleComplex<float> &element);
             virtual void getElement(unsigned int i,GAFW::SimpleComplex<double> &element);
             
             virtual inline void setElement(unsigned int i,float &element);
             virtual inline void setElement(unsigned int i,double &element);
             virtual inline void setElement(unsigned int i,std::complex<double> &element);
             virtual inline void setElement(unsigned int i,std::complex<float> &element);
             virtual inline void setElement(unsigned int i,int &element);
             virtual inline void setElement(unsigned int i,unsigned int &element);
             virtual inline void setElement(unsigned int i,long int &element);
             virtual inline void setElement(unsigned int i,unsigned long int &element);
             virtual inline void setElement(unsigned int i,short int &element);
             virtual inline void setElement(unsigned int i,unsigned short int &element);
             virtual inline void setElement(unsigned int i,GAFW::SimpleComplex<float> &element);
             virtual inline void setElement(unsigned int i,GAFW::SimpleComplex<double> &element);
            
    };
    template<class T>
    inline void StoreManagerProper<T>::getElement(unsigned int i,float &element)
    {
        this->internal_getElement<float>(i,element);
    }
    template<class T>
    inline void StoreManagerProper<T>::getElement(unsigned int i,double &element)
    {
        this->internal_getElement<double>(i,element);
    }
    
template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,std::complex<double> &element)
    {
        this->internal_getElement<std::complex<double> >(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,std::complex<float> &element)
    {
        this->internal_getElement<std::complex<float> >(i,element);
    }
template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,GAFW::SimpleComplex<double> &element)
    {
        this->internal_getElement<GAFW::SimpleComplex<double> >(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,GAFW::SimpleComplex<float> &element)
    {
        this->internal_getElement<GAFW::SimpleComplex<float> >(i,element);
    }



    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,int &element)
    {
        this->internal_getElement<int>(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,unsigned int &element)
    {
        this->internal_getElement<unsigned int>(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,long int &element)
    {
        this->internal_getElement<long int>(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,unsigned long int &element)
    {
        this->internal_getElement<unsigned long int>(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,short int &element)
    {
        this->internal_getElement<short int>(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::getElement(unsigned int i,unsigned short int &element)
    {
        this->internal_getElement<unsigned short int>(i,element);
    }
   
    template<class T>
     inline void StoreManagerProper<T>::setElement(unsigned int i,float &element)
    {
        this->internal_setElement<float>(i,element);
    }
    template<class T>
    inline void StoreManagerProper<T>::setElement(unsigned int i,double &element)
    {
        this->internal_setElement<double>(i,element);
    }
    

     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,std::complex<double> &element)
    {
        this->internal_setElement<std::complex<double> >(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,std::complex<float> &element)
    {
        this->internal_setElement<std::complex<float> >(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,int &element)
    {
        this->internal_setElement<int>(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,unsigned int &element)
    {
        this->internal_setElement<unsigned int>(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,long int &element)
    {
        this->internal_setElement<long int>(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,unsigned long int &element)
    {
        this->internal_setElement<unsigned long int>(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,short int &element)
    {
        this->internal_setElement<short int>(i,element);
    }
    
     template<class T> inline void StoreManagerProper<T>::setElement(unsigned int i,unsigned short int &element)
    {
        this->internal_setElement<unsigned short int>(i,element);
    }
template<class T>
     inline void StoreManagerProper<T>::setElement(unsigned int i,GAFW::SimpleComplex<double> &element)
    {
        this->internal_setElement<GAFW::SimpleComplex<double> >(i,element);
    }
    template<class T>
     inline void StoreManagerProper<T>::setElement(unsigned int i,GAFW::SimpleComplex<float> &element)
    {
        this->internal_setElement<GAFW::SimpleComplex<float> >(i,element);
    }
    
    
    
    template<class T>
    StoreManagerProper<T>::StoreManagerProper(void *&store):store(store)
    {
                
    }
    
    
    template<class T>
     size_t StoreManagerProper<T>::getSizeOfElement()
    {
        return sizeof(T); 
    }
/*
    //Most casts are there but 
    template<class T,class U > inline void StoreManagerProper<T>::getElement<U>(unsigned int i, U& element)
    {
        element=(U)(((T*)this->store)+i);
    }
     template<class T,class U> inline void StoreManagerProper<T>::setElement<U>(unsigned int i, U& element)
    {
        *(((T*)this->store)+i)=T(element);
    }
    
  */  
    
    
    
    
} }//end of namespace
#endif	/* STOREMANAGER_H */

