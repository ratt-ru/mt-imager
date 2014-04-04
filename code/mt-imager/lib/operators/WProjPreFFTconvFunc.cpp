/* WProjPreFFTconvFunc.cpp:  Pure C++ implementation of the WProjPreFFTconvFunc operator 
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
#include "Properties.h"
#include <string>
#include <iostream>
#include <iomanip>
using namespace mtimager;
using namespace GAFW;
using namespace GAFW::GPU; using namespace GAFW::GeneralImplimentation;
using namespace std;
using namespace GAFW::Tools::CppProperties;


WProjPreFFTConvFunc::WProjPreFFTConvFunc(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::WProjPreFFTConvFunc")
{
    
}

WProjPreFFTConvFunc::~WProjPreFFTConvFunc() {
}
void WProjPreFFTConvFunc::validate(){
    /********************************************************************
     * VAlidations:                                                     *
     * 1. No input Matrixes *
     *  Output matrix is set to real_float   
    
     * Output matrix will be set 1x400 
     * ******************************************************************/
    int support,sampling;
    double total_l;
    double total_m;
    float w;
   //No of inputs
    if (this->inputs.size()!=1) throw ValidationException("This operator expects one input. The taper function correction");
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");

    this->checkParameter("w",Properties::Float);
    this->checkParameter("support",Properties::Int);
    this->checkParameter("sampling",Properties::Int);
    this->checkParameter("total_l",Properties::Double);
    this->checkParameter("total_m",Properties::Double);
    
    if (!this->isParameterSet("lwimager-removeedge"))
        this->setParameter("lwimager-removeedge",false);
    this->checkParameter("lwimager-removeedge",Properties::Bool);
    
    
    
    w=this->params.getFloatProperty("w");
    support=this->params.getIntProperty("support");
    sampling=this->params.getIntProperty("sampling");
    total_l=this->params.getDoubleProperty("total_l");
    total_m=this->params.getDoubleProperty("total_m");
     
    if (w<0.0f) throw ValidationException("w value of plane is expected to be positive or zero")
    if (total_l<=0.0) throw ValidationException("total l is 0 or negative");
    if (total_m<=0.0) throw ValidationException("total m is 0 or negative");
    if (support<1) throw ValidationException("Support must be at least 1");
    if (support%2!=1) throw ValidationException("Support is expected to be odd");
    if (sampling<1) throw ValidationException("Sampling must be at least 1");
    
    switch (this->inputs[0]->getType())
    {
        case real_float:
            this->outputs[0]->setType(complex_float);
            break;
        case real_double:
            this->outputs[0]->setType(complex_double);
            break;
        default:
            throw ValidationException("Input is expected to be of real_float or real_double type");
    }
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=2) throw ValidationException("Input is expected to be a 2D Array");
    if (this->inputs[0]->getDimensions().getNoOfColumns()!=support) throw ValidationException("Columns and rows of input is expected to be equal to the support");
    if (this->inputs[0]->getDimensions().getNoOfRows()!=support) throw ValidationException("Columns and rows of input is expected to be equal to the support");
    this->outputs[0]->setDimensions(ArrayDimensions(2,support*sampling,support*sampling));  

}
