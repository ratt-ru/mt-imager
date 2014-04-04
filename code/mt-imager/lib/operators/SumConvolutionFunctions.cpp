/* SumConvolutionFunctions.cpp:  Pure C++ implementation of the SumConvolutionFunctions operator 
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



SumConvolutionFunctions::SumConvolutionFunctions(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::gridsf")
{
    
}

SumConvolutionFunctions::~SumConvolutionFunctions() {
}
void SumConvolutionFunctions::validate()
{
   //No of inputs
    if (this->inputs.size()!=2) throw ValidationException("This operator expects two inputs");
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=2) throw ValidationException("The first input is not 2D");
    if (this->inputs[0]->getType()!=real_int) throw ValidationException("The first input is not of type real_uint");
    if (this->inputs[1]->getDimensions().getNoOfDimensions()!=1) throw ValidationException("The second input is not 1D");
    if (this->inputs[1]->getType()!=complex_float) throw ValidationException("The second input is not complex");
    
    if (this->outputs.size()!=1) throw ValidationException("One outputs supported");
    
    
    this->checkParameter("sampling",Properties::Int);
    int sampling = this->getIntParameter("sampling");
    int planes=this->inputs[0]->getDimensions().getY();
    int convFunctionsPerPlane;
    if (sampling%2)
    {
        convFunctionsPerPlane=sampling*sampling;
    }
    else
    {
        convFunctionsPerPlane=(sampling+1)*(sampling+1);
    }
    
    this->outputs[0]->setDimensions(ArrayDimensions(2,planes,convFunctionsPerPlane));
    this->outputs[0]->setType(real_float);
    
}


