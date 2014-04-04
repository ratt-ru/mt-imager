/* WPlaneConvFunctionCaclulator.h:  Definition of the WPlaneConvFunctionCaclulator  GAFW Module. 
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

#ifndef WPLANECONVFUNCTIONCALCULATOR_H
#define	WPLANECONVFUNCTIONCALCULATOR_H
#include "gafw.h"

#include "WProjectionConvFunction.h"
#include <complex>
namespace mtimager
{
    
    class WPlaneConvFunctionCalculator: public GAFW::Module {
    public:
         
    protected:
        const struct WProjectionConvFunction::Conf conf;
        
    public:

        WPlaneConvFunctionCalculator(GAFW::Factory *factory,std::string nickname, WProjectionConvFunction::Conf conf);
        ~WPlaneConvFunctionCalculator();
       virtual void reset();
       virtual void calculate();
       virtual void setInput(int inputNo, GAFW::Result *res);
       virtual GAFW::Result * getOutput(int outputNo); 
       virtual void resultRead(GAFW::ProxyResult *,int snapshot_no);
       void setTrialSupport(int support);
       void setW(double w);
       virtual GAFW::Result * getConvFunction();
       int getCalculatedSupport();
       std::complex<double> getW0OriginValue();
       void setNormalizer(float normalizer);
       };
}

#endif	/* WPLANECONVFUNCTIONCALCULATOR_H */

