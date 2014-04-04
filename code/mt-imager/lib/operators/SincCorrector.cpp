/* SincCorrector.cpp:  Pure C++ implementation of the SincCorrector operator 
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
using namespace GAFW::GPU; using namespace GAFW::GeneralImplimentation;
using namespace GAFW::Tools::CppProperties;
using namespace std;



SincCorrector::SincCorrector(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::gridsf")
{
    
}

SincCorrector::~SincCorrector() {
}
void SincCorrector::validate()
{
   //No of inputs
    if (this->inputs.size()!=1) throw ValidationException("This operator expects one input");
    if (this->outputs.size()!=1) throw ValidationException("1 output expected");
    if ((this->inputs[0]->getDimensions().getNoOfDimensions()>3)||(this->inputs[0]->getDimensions().getNoOfDimensions()==1))
            throw ValidationException("Input 0 is expected to be of 2 or 3  dimensions");
    if (this->inputs[0]->getType()!=real_float)
        throw ValidationException ("This operator works only on real_floats");
    
    this->checkParameter("sampling",Properties::Int);
    this->outputs[0]->setDimensions(this->inputs[0]->getDimensions());
    this->outputs[0]->setType(real_float);
}


