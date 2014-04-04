/* gafw-impl.h:  Main header file for the general implementation of the GAFW. 
 * It includes all other header files and some other definitions
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
#ifndef __GAFW_IMPL_H__
#define	__GAFW_IMPL_H__
#include <vector>
#include <queue>
#include "gafw.h"
#include "gafw-enums.h"

namespace GAFW { namespace GeneralImplimentation {
//A list of all classes
class Factory;

class Array;
class Result;
class DataStore;
class ArrayOperator;
class DataStoreSnapshotDescriptor;
class Engine;

class GAFWValidationException;
class ProxyResult;
class Registry;


using GAFW::LogFacility;
using GAFW::Identity;
using GAFW::GAFWException;
using GAFW::ArrayDimensions;
using GAFW::CalculationId;
using GAFW::DataTypeBase;

class DataTypeManual: public DataTypeBase
{
public:
    DataTypeManual(const int type):DataTypeBase(type){};
    DataTypeManual(const StoreType type):DataTypeBase((int)type){};
};

typedef struct {
    std::vector <GAFW::GeneralImplimentation::ArrayOperator*> input_to;  //Can be an input to many operators  
    std::vector <GAFW::GeneralImplimentation::Array*> preValidatorDependents;
    GAFW::GeneralImplimentation::ArrayOperator *output_of;   ///but we only allow to be an output of only one operator
    DataStore * store;
    GAFW::GeneralImplimentation::Result *result;
    GAFW::Result *result_Outputof;
  //  Factory *factory; 
    ArrayDimensions dim;
    StoreType type;
    PreValidator *preValidator;
    bool toReuse;
    bool toOverwrite;
    bool requireResult;
    bool removeReusability;
} ArrayInternalData;


}}// End of namespace

//Now we include all header files that define each class
#include "GAFWValidationException.h"
#include "Registry.h"
#include "GAFWValidationException.h"
#include "Factory.h"
#include "FactoryAccess.h"
#include "DataStore.h"
#include "Array.h"


#include "ArrayOperator.h"
#include "Result.h"
#include "DataStoreSnapshotDescriptor.h"
#include "Engine.h"
#include "ProxyResult.h"

#endif	

