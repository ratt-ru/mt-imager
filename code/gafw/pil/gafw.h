/* gafw.h:  Main Header file amalgamating all PIL header files     
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */

#ifndef __PIL_GAFW_H__
#define	__PIL_GAFW_H__

#define NICOLA_COMPATIBLE
#include <vector>
#include <string>
namespace GAFW {
//A list of all classes
class ArrayDimensions;
class Identity;
class Factory;
class Array;
class Result;
class ArrayOperator;
class PreValidator;
class GAFWException;
class Module;
class Structure;
class ProxyResult;
class LogFacility;
class FactoryHelper;
class Statistic;
class DataTypeBase;
class FactoryAccess;

typedef long long int CalculationId; 


class Statistic
{
public:
    virtual ~Statistic(){};
};



class FactoryStatisticsOutput
{
public:
    virtual void push_statistic(Statistic *)=0;
};



}
#include "SimpleComplex.h"
#include "DataType.h"
#include "PointerWrapper.h"
#include "ValueWrapper.h"
#include "ParameterWrapper.h"
#include "ParameterType.h"
#include "GAFWException.h"
#include "BugException.h"
#include "ArrayDimensions.h"
#include "Identity.h"



#include "LogFacility.h"
#include "FactoryHelper.h"
#include "Factory.h"
#include "FactoryAccess.h"
#include "PreValidator.h"
#include "Array.h"
#include "ArrayOperator.h"
#include "Result.h"
#include "Module.h"
#include "Structure.h"
#include "ProxyResult.h"

#endif	
