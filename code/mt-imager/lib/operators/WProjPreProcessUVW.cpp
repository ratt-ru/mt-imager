/* WProjPreProcessUVW.cpp:  Pure C++ implementation of the WProjPreProcessUVW operator 
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



WProjPreProcessUVW::WProjPreProcessUVW(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::PreProcessUVW")
{
    
}

WProjPreProcessUVW::~WProjPreProcessUVW() {
}
void WProjPreProcessUVW::validate()
{
    /********************************************************************
     * VAlidations:                                                     *
     * 1. No input Matrixes *
     *  Output matrix is set to real_float   
    
     * Output matrix will be set 1x400 
     * ******************************************************************/
  
   //No of inputs
    if (this->inputs.size()!=6) throw ValidationException("This operator expects six inputs which are Convolution Function support data and UVW entries");
    if (this->outputs.size()!=7) throw ValidationException("Five outputs expected");
    
    logDebug(validation,"No of inputs and outputs are correct");
    this->checkParameter("ConvolutionFunction.sampling",Properties::Int);
    //this->checkParameter("ConvolutionFunction.support",Properties::Int);
    //this->checkParameter("ConvolutionFunction.wplanes",Properties::Int);
    this->checkParameter("ConvolutionFunction.wsquareincrement",Properties::Float);
    this->checkParameter("uvImage.u_increment",Properties::Float);
    this->checkParameter("uvImage.v_increment",Properties::Float);
    this->checkParameter("uvImage.rows",Properties::Int);
    this->checkParameter("uvImage.columns",Properties::Int);
    
    int no_of_polarisations=this->inputs[5]->getDimensions().getNoOfColumns();
    //Continue more checks here
    stringstream s;
    s<< "Polarisation found to be " <<no_of_polarisations;
    logWarn(other,s.str());
    logWarn(other,"Parameters still need to be validated and inputs.. To Program");

    int no_of_points=this->inputs[2]->getDimensions().getNoOfColumns();
    this->outputs[0]->setDimensions(ArrayDimensions(2,no_of_points,4));
    this->outputs[0]->setType(real_int);
    this->outputs[1]->setDimensions(ArrayDimensions(1,no_of_points));
    this->outputs[1]->setType(real_int);
    this->outputs[2]->setDimensions(ArrayDimensions(1,no_of_points));
    this->outputs[2]->setType(real_int);
    this->outputs[3]->setDimensions(ArrayDimensions(1,no_of_points));
    this->outputs[3]->setType(real_int);
    this->outputs[4]->setDimensions(ArrayDimensions(1,no_of_points));
    this->outputs[4]->setType(real_int);
    this->outputs[5]->setDimensions(ArrayDimensions(1,no_of_points));
    this->outputs[5]->setType(real_float);
    this->outputs[6]->setDimensions(ArrayDimensions(2,no_of_polarisations,no_of_points));
    this->outputs[6]->setType(real_float);

}


