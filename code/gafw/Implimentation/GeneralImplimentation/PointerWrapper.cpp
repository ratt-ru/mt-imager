/* PointerWrapper.cpp:  Template code of PointerWrapper and  instantiation.
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
PointerWrapper<T>::PointerWrapper(T * pointer):PointerWrapperBase(GAFW::GeneralImplimentation::typeToIntConvert<T>())
, pointer(pointer)
{
    
}
template class PointerWrapper<std::complex<float> >;
template class PointerWrapper<std::complex<double> >;
template class PointerWrapper<SimpleComplex<float> >;
template class PointerWrapper<SimpleComplex<double> >;
template class PointerWrapper<double>;
template class PointerWrapper<float>;
template class PointerWrapper<int>;
template class PointerWrapper<unsigned int>;
template class PointerWrapper<short int>;
template class PointerWrapper<unsigned short int>;
template class PointerWrapper<long int>;
template class PointerWrapper<unsigned long int>;
