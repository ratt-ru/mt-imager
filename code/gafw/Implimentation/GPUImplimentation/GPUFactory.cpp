/* GPUFactory.cpp:  Implementation of the GPUFactory class. 
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
#include "GPUafw.h"
//#include "StandardOperators/GPUOperatorsFactoryHelper.h"
//#include "StandardOperators/StandardOperators.h"
#include <iostream>
using namespace GAFW::GeneralImplimentation;
using namespace GAFW::GPU;
GPUFactory::GPUFactory(GAFW::FactoryStatisticsOutput * statisticOutput):GAFW::GeneralImplimentation::Factory("GPUFactory","GPUFactory",statisticOutput) {
    this->engine=NULL;
    //Auto-0register the standard operators FactoryHelper
//    FactoryHelper* fh=new GAFW::StandardOperators::GPUOperatorsFactoryHelper();
//    this->registerHelper(fh);
    this->engine=new GPUEngine(this,statisticOutput); 
//    delete fh;  //there is a BUG when deleting so for now we omit
            
}



GPUFactory::~GPUFactory() 
{
    delete this->engine;
}
GAFW::GPU::GPUEngine *GPUFactory::requestEngine()
{
   
    return this->engine;
        
}
DataStore *GPUFactory::requestDataStore(DataTypeBase &type,ArrayDimensions &dim,bool allocate)
{
   
    return new GPUDataStore(this,type,dim,allocate); // This will be changed in the future
    //once we enable a kind of GC
}
