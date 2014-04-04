/* ImagerFactoryHelper.cpp:  C++ Implementation the  ImagerFactoryHelper  
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

#include "ImagerFactoryHelper.h"
#include "ImagerOperators.h"
using namespace std;
using namespace GAFW;
using namespace mtimager;


 FactoryHelper* ImagerFactoryHelper::reCreateForFactory(Factory *f)
 {
    ImagerFactoryHelper * ret=new ImagerFactoryHelper();
    ret->factory=dynamic_cast<GAFW::GPU::GPUFactory *>(f);
    if (f->getName()=="GPUFactory")
        ret->isGPUFactory=true;
    return ret;
 }
ImagerFactoryHelper::ImagerFactoryHelper():Identity("mt-imager FactoryHelper","mt-imager FactoryHelper") 
{
    this->factory=NULL;
    this->isGPUFactory=false;
}

ImagerFactoryHelper::~ImagerFactoryHelper()
{
    
}
ArrayOperator *ImagerFactoryHelper::createOperator(std::string nickname, std::string name)
{
    if (this->isGPUFactory==false)
        return NULL;
    if (name=="casagridsf")
        return new CasaGridSF(this->factory,nickname);
    
    if (name=="WProjPreFFTConvFunc")
        return new WProjPreFFTConvFunc(this->factory,nickname);
    
    
    if (name=="PreIFFTUVCorrection")
        return new PreIFFTUVCorrection(this->factory,nickname);
    if (name=="ConvFunctionReOrganize")
        return new ConvFunctionReOrganize(this->factory,nickname);
    if (name=="ConvFunctionSupportFind")
        return new ConvFunctionSupportFind(this->factory,nickname);
   if (name=="PreProcessUVW")
        return new WProjPreProcessUVW(this->factory,nickname);
    if (name=="CreateIndexAndReorder")
        return new CreateIndexAndReorder(this->factory,nickname);
    if (name=="NotSameSupportManipulate")
        return new NotSameSupportManipulate(this->factory,nickname);
    if (name=="CreateBlockDataIndex")
        return new CreateBlockIndex(this->factory,nickname);
    if (name=="CreateCompressionPlan")
        return new CreateCompressionPlan(this->factory,nickname);
    if (name=="CompressVisibilities")
        return new CompressVisibilities(this->factory,nickname);
    if (name=="ConvolutionGridder")
        return new ConvolutionGridder(this->factory,nickname);
    if (name=="SumConvolutionFunctions")
        return new SumConvolutionFunctions(this->factory,nickname);
    if (name=="SincCorrector")
        return new SincCorrector(this->factory,nickname);
    if (name=="StokesConverter")
        return new PostIFFTStokesConverter(this->factory,nickname);
    

    
    
    return NULL;
}

