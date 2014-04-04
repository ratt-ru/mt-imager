/* NotSameSupportManipulate.cpp:  Pure C++ implementation of the NotSameSupportManipulate operator 
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
#include "ImagerOperators.h"

#include <string>
using namespace mtimager;
using namespace GAFW;
using namespace GAFW::GPU; 
using namespace GAFW::GeneralImplimentation;
using namespace GAFW::Tools::CppProperties;
using namespace std;



NotSameSupportManipulate::NotSameSupportManipulate(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::gridsf")
{
    
}

NotSameSupportManipulate::~NotSameSupportManipulate() {
}
void NotSameSupportManipulate::validate()
{
   //No of inputs
    if (this->inputs.size()!=2) throw ValidationException("This operator expects two inputs");
    if (this->outputs.size()!=1) throw ValidationException("1 output expected");
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=1)
            throw ValidationException("Input 0 is expected to be of 1 dimension");
   
    int records=this->inputs[0]->getDimensions().getNoOfColumns(); //Input 0 is not same support Indicator
    if (this->inputs[1]->getDimensions().getNoOfDimensions()!=2)
            throw ValidationException("Input 1 is expected to be of 1 dimension");
    if (this->inputs[1]->getDimensions().getNoOfRows()!=records+1)
            throw ValidationException("Input 1 is expected to have the 1 entry more then input 0");
    
    ArrayDimensions dim(1,records);
    this->outputs[0]->setDimensions(dim);
    this->outputs[0]->setType(real_int);


    
    
}


