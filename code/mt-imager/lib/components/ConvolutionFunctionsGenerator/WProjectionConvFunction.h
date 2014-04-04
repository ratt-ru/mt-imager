/* WProjectionConvFunction.h:  Definition of the WProjectionConvFunction GAFW Module. 
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

#ifndef __WPROJECTIONCONVFUNCTION_H__
#define	__WPROJECTIONCONVFUNCTION_H__
#include "gafw.h"
#include <string>
#include "Properties.h"
#include "PropertiesManager.h"
//#include "WImager_backup.h"
//#include "ImageManagerold.h"
#include "ConvolutionFunctionGenerator.h"
//#include "WPlaneConvFunctionCalculator.h"

namespace mtimager
{
    class WPlaneConvFunctionCalculator;
    class WProjectionConvFunction: public GAFW::Module,public ConvolutionFunctionGenerator {
    public:
         struct Conf
        {
             bool standard_method;
            std::string taper_operator;
            int taper_support;
            
            //int taper_sampling;
            //int conv_function_support;
            int conv_function_sampling;
            int wplanes;
            
            double image_total_l;
            double image_total_m;
            double image_l_increment;
            int img_min_dim;
        };

    protected:
        //WImagerParams params; //this needs change
        float wIncrement;
        const struct Conf conf;
         WPlaneConvFunctionCalculator *convCalculator;
        //ImageManagerOld &imagedef;
        std::vector<GAFW::Array *> convFunctions;
    public:

        //WProjectionConvFunction(GAFW::Factory *factory,std::string nickname, CppProperties::Properties& params, ageManagerOld &imageMan);
        WProjectionConvFunction(GAFW::Factory *factory,std::string nickname, WProjectionConvFunction::Conf conf);
        ~WProjectionConvFunction();
       virtual void reset();
       virtual void calculateConvFunction();
       virtual void calculate();
       virtual void setInput(int inputNo, GAFW::Result *res);
       virtual GAFW::Result * getOutput(int outputNo); 
       virtual void resultRead(GAFW::ProxyResult *,int snapshot_no);
       //void paramLoader(CppProperties::Properties& params);
       
       virtual GAFW::Result * getConvFunction();
       virtual GAFW::Result * getConvFunctionPositionData();
       virtual GAFW::Result * getConvFunctionSumData();
       virtual int getMaxSupport();
       virtual int getSampling();
       virtual float getWSquareIncrement();
       
      

    };
}

#endif	/* WPROJECTIONCONVFUNCTION_H */

