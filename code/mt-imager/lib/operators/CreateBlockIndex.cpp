/* CreateBlockIndex.cpp:  Pure C++ implementation of the CreateBlockIndex operator 
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



CreateBlockIndex::CreateBlockIndex(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::gridsf")
{
    
}

CreateBlockIndex::~CreateBlockIndex() {
}
void CreateBlockIndex::validate()
{
   //No of inputs
    if (this->inputs.size()!=3) throw ValidationException("This operator expects three inputs");
    
    if (this->outputs.size()!=1) throw ValidationException("Four outputs expected");
    int records=this->inputs[0]->getDimensions().getNoOfColumns();
    
    for (int x=0;x<2;x++)
    {
        if (this->inputs[x]->getDimensions().getNoOfDimensions()!=1)
            throw ValidationException("Inputs expected to be all of 1 dimension");
        if (this->inputs[x]->getDimensions().getNoOfColumns()!=records)
            throw ValidationException("All inputs are expected to contain the same no of elements");
    }
    
    
    this->checkParameter("with_support",Properties::Bool);
      
    if (this->getBoolParameter("with_support"))
    {    
        ArrayDimensions dim(2,records+1,2);
        this->outputs[0]->setDimensions(dim);
        this->outputs[0]->setType(real_int);
    }
    else
    {
        ArrayDimensions dim(1,records+1,1);
        this->outputs[0]->setDimensions(dim);
        this->outputs[0]->setType(real_int);
    }
    
    
    
}


