/* ValueWrapper.cpp:  Template code of ValueWrapper and  instantiation.
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


#include "gafw-impl.h"
#include <complex>
#include "TypeToIntConvert.h"
using namespace GAFW;
template <class T>
ValueWrapper<T>::ValueWrapper(T value):ValueWrapperBase(GAFW::GeneralImplimentation::typeToIntConvert<T>())
, value(value)
{
    
}
template <class T>
ValueWrapper<T>::ValueWrapper():ValueWrapperBase(GAFW::GeneralImplimentation::typeToIntConvert<T>())
{
    
}
template <class T>
ValueWrapper<T> & ValueWrapper<T>::setValue(T value)
{
    this->value=value;
    return *this;
}

template class ValueWrapper<std::complex<float> >;
template class ValueWrapper<std::complex<double> >;
template class ValueWrapper<SimpleComplex<float> >;
template class ValueWrapper<SimpleComplex<double> >;
template class ValueWrapper<double>;
template class ValueWrapper<float>;
template class ValueWrapper<int>;
template class ValueWrapper<unsigned int>;
template class ValueWrapper<short int>;
template class ValueWrapper<unsigned short int>;
template class ValueWrapper<long int>;
template class ValueWrapper<unsigned long int>;
