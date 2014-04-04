/* ImageFinalizer.h: Definition  of the ImageFinalizer component, class and GAFW module. 
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

#ifndef __IMAGEFINALIZER_H__
#define	__IMAGEFINALIZER_H__
#include "gafw.h"
#include "mtimager.h"
namespace mtimager
{
    class ImageFinalizer: public GAFW::Module 
    {
    public :
        struct Conf
        {
                int image_nx;
                int image_ny;
                string taper_operator;
                int taper_support;
                int conv_function_sampling;
                enum PolarizationType::Type polType;
        };
    protected:
        const struct Conf conf;
        float normalizer;
    public:
        ImageFinalizer(GAFW::Factory *factory,std::string nickname,  Conf conf);
        virtual ~ImageFinalizer();
        virtual void reset();
        virtual void calculate();
        virtual void setInput(int inputNo, GAFW::Result *res);
        
        virtual GAFW::Result * getOutput(int outputNo); 
        virtual void resultRead(GAFW::ProxyResult *,int snapshot_no);
        //static  void registerParameters(CppProperties::PropertiesManager & manager);
    };
}

#endif	/* IMAGEFINALIZER_H */

