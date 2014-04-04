/* ParameterWrapper.cpp:  Template code of ParameterWrapper and  instantiation.
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
using namespace GAFW::GeneralImplimentation;
template <class T> 
ParameterWrapper<T>::ParameterWrapper(std::string name, T value):ParameterWrapperBase(ParameterTypeToInt<T>())
{
    this->setName(name);
    this->value=value;
}
template <class T>
ParameterWrapper<T>::ParameterWrapper():ParameterWrapperBase(ParameterTypeToInt<T>())
{
    
}
template <class T>
ParameterWrapper<T>::ParameterWrapper(std::string name):ParameterWrapperBase(ParameterTypeToInt<T>())
{
    this->setName(name);
}
template <class T>
inline ParameterWrapper<T> & ParameterWrapper<T>::setNameAndValue(std::string name,T value)
{
    this->setName(name);
    this->value=value;
}
template <class T>
inline ParameterWrapper<T> & ParameterWrapper<T>::setValue(T value)
{
    this->value=value;
}
template class ParameterWrapper<int>;
template class ParameterWrapper<float>;
template class ParameterWrapper<double>;
template class ParameterWrapper<std::complex<float> >;
template class ParameterWrapper<std::string>;
template class ParameterWrapper<void *>;
template class ParameterWrapper<bool>;


