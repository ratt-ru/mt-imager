/* DataType.cpp.  Template specialisations for GAFW::DataType.
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


#include "gafw-enums.h"
#include "DataType.h"
#include "SimpleComplex.h"
using namespace GAFW::GeneralImplimentation;
#include <complex>
namespace GAFW
{
    template <>
    class DataType<void>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<void>::DataType():DataTypeBase((int)StoreTypeUnknown)
    {}
   
    template <>
    class DataType<std::complex<float> >: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<std::complex<float> >::DataType():DataTypeBase((int)complex_float)
    {}
    
    template <>
    class DataType<std::complex<double> >: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<std::complex<double> >::DataType():DataTypeBase((int)complex_double)
    {}
    
    template<>
    class DataType<SimpleComplex<float> >: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<SimpleComplex<float> >::DataType():DataTypeBase((int)complex_float)
    {}
    
    template <>
    class DataType<SimpleComplex<double> >: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<SimpleComplex<double> >::DataType():DataTypeBase((int)complex_double)
    {}
    
    template <>
    class DataType<float>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<float >::DataType():DataTypeBase((int)real_float)
    {}
        
    
    template <>
    class DataType<double>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<double>::DataType():DataTypeBase((int)real_double)
    {}
    
    template <>
    class DataType<int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<int>::DataType():DataTypeBase((int)real_int)
    {}
    
    template <>
    class DataType<unsigned int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<unsigned int>::DataType():DataTypeBase((int)real_uint)
    {}
    
    template <>
    class DataType<long int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<long int>::DataType():DataTypeBase((int)real_longint)
    {}
    
    template <>
    class DataType<unsigned long int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<unsigned long int>::DataType():DataTypeBase((int)real_ulongint)
    {}
    
    template <>
    class DataType<short int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<short int>::DataType():DataTypeBase((int)real_shortint)
    {}
    
    template <>
    class DataType<unsigned short int>: public DataTypeBase
    {

    public:
        DataType();
    };
    DataType<unsigned short int>::DataType():DataTypeBase((int)real_ushortint)
    {}
    
    //A way how to cause compilation of the specialisations
    struct __dummy__
    {   DataType<void> a;
        DataType<std::complex<float> > b;
        DataType<std::complex<double> > c;
        DataType<SimpleComplex<float> > d;
        DataType<SimpleComplex<double> > e;
        DataType<double> f;
        DataType<float> g;
        DataType<int> h;
        DataType<unsigned int> i;
        DataType<long int> j;
        DataType<unsigned long int> k;
        DataType<short int> l;
        DataType<unsigned short int> m;
    };
    
} 