/* globalVariables.cpp:  IInitialization  of global variables/arrays.
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
using namespace GAFW::GeneralImplimentation;
const StoreType GAFW::GeneralImplimentation::ValuePointerTypeMap[std_unsigned_short_int+1]
={      complex_float,complex_double,complex_float,complex_double,
        real_float,real_double,real_int,real_uint,real_longint,real_ulongint,
        real_shortint,   real_ushortint };

