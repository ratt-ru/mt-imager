/* TypeToIntConvert.cpp:  template specialisation of the of TypeToIntConvert() function
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
#include "TypeToIntConvert.h"
#include "SimpleComplex.h"
#include "Properties.h"
#include <complex>
namespace GAFW { namespace GeneralImplimentation 
{
    template  <> int typeToIntConvert<std::complex<float> >()
    {
        return (int) std_complex_float;
    }
    template  <> int typeToIntConvert<std::complex<double> >()
    {
        return (int) std_complex_double;
    }
    template  <> int typeToIntConvert<GAFW::SimpleComplex<float> >()
    {
        return (int) simple_complex_float;
    }
    template  <> int typeToIntConvert<GAFW::SimpleComplex<double> >()
    {
        return (int) simple_complex_double;
    }
    template  <> int typeToIntConvert<float>()
    {
        return (int) std_float;
    }
    template  <> int typeToIntConvert<double>()
    {
        return (int) std_double;
    }
    template  <> int typeToIntConvert<int>()
    {
        return (int) std_int;
    }
    template  <> int typeToIntConvert<unsigned int>()
    {
        return (int) std_unsigned_int;
    }
    template  <> int typeToIntConvert<short int>()
    {
        return (int) std_short_int;
    }
    template  <> int typeToIntConvert<unsigned short int>()
    {
        return (int) std_unsigned_short_int;
    }
    template  <> int typeToIntConvert<long int>()
    {
        return (int) std_long_int;
    }
    template  <> int typeToIntConvert<unsigned long int>()
    {
        return (int) std_unsigned_long_int;
    }
    
    
    template <> int ParameterTypeToInt<int>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Int;
    }
    template <> int ParameterTypeToInt<float>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Float;
    }
    template <> int ParameterTypeToInt<double>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Double;
    }
    template <> int ParameterTypeToInt<std::complex<float> >()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Complex;
    }
    template <> int ParameterTypeToInt<std::string>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::String;
    }
    template <> int ParameterTypeToInt<void *>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Pointer;
    }
    template <> int ParameterTypeToInt<bool>()
    {
        return (int) GAFW::Tools::CppProperties::Properties::Bool;
    }
    
    
    
    
} }
