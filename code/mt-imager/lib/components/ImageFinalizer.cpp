/* ImageFinalizer.cpp: Implementation  of the ImageFinalizer component, class and GAFW module. 
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

#include "ImageFinalizer.h"
#include "MTImagerException.h"
using namespace GAFW::Tools::CppProperties;
using namespace mtimager;
using namespace GAFW;
ImageFinalizer::ImageFinalizer(GAFW::Factory *factory,std::string nickname, Conf conf):Module(factory,nickname,"UV-LM Finalizer"),conf(conf) 
{
    
    ParameterWrapper<int> int_parameter;
    ParameterWrapper<bool> bool_parameter;

    this->requestMyArray("UVImage");
    this->requestMyArray("Normalizer");
    this->requestMyOperator("FFTPrepare","PreIFFTUVCorrection","UVImage")->a("PreparedUVImage");
    this->requestMyOperator("UVIfft","IFFT2D","PreparedUVImage")->a("OriginalPolImage1"); 
    this->requestMyOperator("Shift","ZeroToCentre2DShift","OriginalPolImage1")->a("OriginalPolImage");
    this->requestMyOperator("StokesConversion","StokesConverter","OriginalPolImage","Normalizer")->a("StokesReadyImage");
    this->getMyOperator("StokesConversion")->setParameter(int_parameter.setNameAndValue("PolarizationType",(int)conf.polType));
    this->requestMyOperator("TaperCorrection",this->conf.taper_operator,"StokesReadyImage")->a("TaperCorrectedImage");
    this->getMyOperator("TaperCorrection")->setParameter(int_parameter.setNameAndValue("support",this->conf.taper_support));
    this->getMyOperator("TaperCorrection")->setParameter(int_parameter.setNameAndValue("sampling",1));
    this->getMyOperator("TaperCorrection")->setParameter(bool_parameter.setNameAndValue("lmPlanePostCorrection",true));
    this->getMyOperator("TaperCorrection")->setParameter(bool_parameter.setNameAndValue("generate",false));
    this->requestMyOperator("SincCorrection","SincCorrector","TaperCorrectedImage")->a("FinalImage");
    this->getMyOperator("SincCorrection")->setParameter(int_parameter.setNameAndValue("sampling",this->conf.conv_function_sampling));
   
}



ImageFinalizer::~ImageFinalizer() 
{


}

void ImageFinalizer::reset()
{
    //Nothing to do
}
void ImageFinalizer::calculate()
{
    //this->getMyOperator("StokesConversion")->setParameter("multiplier.value",1.0f/(this->normalizer)); //float(this->image_nx*this->image_ny)));
    this->getMyResult("FinalImage")->calculate();
    //this->getMyResult("FinalImage")->waitUntilDataValid();
}
void ImageFinalizer::setInput(int inputNo, GAFW::Result *res)
{
    switch (inputNo)
    {
        case 0:
          this->getMyArray("UVImage")->bind_to(res);
          break;
        case 1:
          this->getMyArray("Normalizer")->bind_to(res);
          break;
        default:
                throw ImagerException("This module only one input");
    }
}

Result * ImageFinalizer::getOutput(int outputNo) 
{
    if (outputNo==0) return this->getMyResult("FinalImage");
    else
        throw ImagerException("Only one output available");
}
void ImageFinalizer::resultRead(GAFW::ProxyResult *,int snapshot_no)
{
   // Nothing to do 
}
