/* GPUOperatorsFactoryHelper.h:  Definition of the GPUOperatorsFactoryHelper class 
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
#ifndef __GPUOPERATORSFACTORYHELPER_H__
#define	__GPUOPERATORSFACTORYHELPER_H__
#include <string>

namespace GAFW { namespace GPU { namespace StandardOperators
{
    class GPUOperatorsFactoryHelper: public GAFW::FactoryHelper {
    private:
        GAFW::GPU::GPUFactory *factory;
        bool isGPUFactory;
        GPUOperatorsFactoryHelper(const GPUOperatorsFactoryHelper& orig):GAFW::Identity("",""),GAFW::FactoryHelper(){};
    protected:
        virtual GAFW::FactoryHelper* reCreateForFactory(GAFW::Factory *f);
    public:
        GPUOperatorsFactoryHelper();
        ~GPUOperatorsFactoryHelper();
        virtual GAFW::ArrayOperator *createOperator(std::string nickname, std::string name); 
    };
}}} // end of 2 namespaces

#endif	/* GPUOPERATORSFACTORYHELPER_H */

