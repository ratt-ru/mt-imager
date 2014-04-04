/* DataStore.cpp:  DataStore General Implementation 
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
using namespace GAFW::GeneralImplimentation;
DataStore::DataStore(Factory * factory, std::string nickname,std::string name,DataTypeBase type,ArrayDimensions dim):FactoryAccess(factory),Identity(nickname,name),storetype(type)
{
    FactoryAccess::init();
    LogFacility::init();
    if (!dim.isWellDefined()) throw GeneralException("Dimensions of array not yet defined. Cannot create datastore");
    
    this->dim=dim;
   
}
