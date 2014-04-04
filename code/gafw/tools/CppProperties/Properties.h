/* Properties.h: Header file defining the Properties class, part of the GAFW CPPProperties Tool     
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */
#ifndef __PROPERTIES_H__
#define	__PROPERTIES_H__
#include <map>
#include <string>
#include <complex>
namespace GAFW {  namespace Tools { namespace CppProperties
{
    class Properties {
    public:
    enum PropertyType
    {
        Int, Float, Double, Complex, String,Pointer,Bool
    };

    protected:
        std::map <std::string,int> intProperties;
        std::map <std::string,float> floatProperties;
        std::map <std::string,double> doubleProperties;
        std::map <std::string,std::string> stringProperties;
        std::map <std::string,std::complex<float> > complexProperties;
        std::map <std::string, void *> pointerProperties;
        std::map <std::string, bool> boolProperties;
        std::map <std::string, enum PropertyType> propertyMap;  
    public:
        Properties();
        Properties(const Properties& orig);
        ~Properties();
        void setProperty(std::string name,std::string value);
        void setProperty(std::string name, int value);
        void setProperty(std::string name,float value);
        void setProperty(std::string name,double value);
        
        void setProperty(std::string name,void * value);
        
        void setProperty(std::string name,std::complex<float> value);
        void setProperty(std::string name,bool value);
        std::string getStringProperty(std::string name);
        int getIntProperty(std::string name);
        float getFloatProperty(std::string name);
        double getDoubleProperty(std::string name);
        void * getPointerProperty(std::string name);
        inline std::string getProperty(std::string name); 
        std::complex<float> getComplexProperty(std::string name);
        bool getBoolProperty(std::string name);
        
        enum PropertyType getPropertyType(std::string name);
        inline bool isPropertySet(std::string name);
        void deleteProperty(std::string name);
        Properties& operator=(Properties &orig);
    };


    inline bool Properties::isPropertySet(std::string name)
    {
        return (bool)this->propertyMap.count(name);
    }
    inline std::string Properties::getProperty(std::string name)
    {
        return this->getStringProperty(name);
    }
    
    

} } }
#endif	/* PROPERTIES_H */

