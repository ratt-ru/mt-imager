/* OnlyTaperConvFunction.h:  Definition of the OnlyTaperConvFunction GAFW module. 
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


#ifndef __ONLYTAPERCONVFUNCTION_H__
#define	__ONLYTAPERCONVFUNCTION_H__

#include "gafw.h"
#include <string>
//#include "Properties.h"
//#include "PropertiesManager.h"
//#include "WImager_backup.h"
//#include "ImageManagerold.h"
#include "ConvolutionFunctionGenerator.h"
#include "WProjectionConvFunction.h"
namespace mtimager
{
     /*struct WImagerParams
    {
        string taper_operator;
        int taper_support;
        int taper_sampling;
        int conv_function_support;
        int conv_function_sampling;
        //int conv_function_padding_factor;
        int wplanes;
        //int nx;
        //int ny;
        //float intervalsize_l;
        //float intervalsize_m;
        int minimum_support_init;
        
    };*/
    class OnlyTaperConvFunction: public GAFW::Module,public ConvolutionFunctionGenerator {
    public:
        struct Conf
        {
                string taper_operator;
                int taper_support;
                int taper_sampling;
        };
    protected:
        
        
        //WImagerParams params; //this needs change
        float wIncrement;
        //ImageManagerOld &imagedef;
        std::vector<GAFW::Array *> convFunctions;
    public:
        const struct Conf conf;
        OnlyTaperConvFunction(GAFW::Factory *factory,std::string nickname, Conf conf);
        ~OnlyTaperConvFunction();
       virtual void reset();
       virtual void calculate();
       virtual void calculateConvFunction();
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
       
       //static  void registerParameters(CppProperties::PropertiesManager & manager);


    };
}


#endif	/* ONLYTAPERCONVFUNCTION_H */

