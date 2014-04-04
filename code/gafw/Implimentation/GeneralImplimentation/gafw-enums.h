/* gafw-enum.h:  Definition of enumerations.
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
#ifndef __GAFW_ENUMS_H__
#define	__GAFW_ENUMS_H__
namespace GAFW { namespace  GeneralImplimentation { 
typedef enum {
        complex_float=0,
        complex_double,
        real_float,
        real_double,
        real_int,
        real_uint,
        real_longint,
        real_shortint,
        real_ulongint,
        real_ushortint,
        StoreTypeUnknown
   } StoreType; //This defines for DataType
   
typedef enum {
        std_complex_float=0,  //very useful to put to 0
        std_complex_double,
        simple_complex_float,
        simple_complex_double,
        std_float,
        std_double,
        std_int,
        std_unsigned_int,
        std_long_int,
        std_unsigned_long_int,
        std_short_int,
        std_unsigned_short_int
} ValuePointerType; //This defines type for Pointer and Value Wrapper data types,
// voids and unknown do not make sense... Pointers to void are also useless, 
// so we do not support
extern const StoreType ValuePointerTypeMap[std_unsigned_short_int+1]; // a map to make things easy


}}

#endif	/* GAFW_ENUMS_H */

