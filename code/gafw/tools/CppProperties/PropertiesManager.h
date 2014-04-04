/* PropertyManager.h: Header file for the PropertyManager class, part of the GAFW CPPProperties Tool     
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

#ifndef __PROPERTIESMANAGER_H__
#define	__PROPERTIESMANAGER_H__
#include <map>
#include <string>
#include "Properties.h"
namespace GAFW {  namespace Tools { namespace CppProperties
{
    class PropertiesManager 
    {
    protected:
        std::map<std::string,std::pair<std::string,enum Properties::PropertyType> > PropertyDefenitions;

    public:
        PropertiesManager();
        PropertiesManager(const PropertiesManager& orig);
        virtual ~PropertiesManager();
        
        void addPropertyDefenition(std::string propertyName, std::string propertyDescription,enum Properties::PropertyType type);
        void changePropertyDescription(std::string propertyName, std::string propertyDescription);
        void changePropertyType(std::string propertyName,enum Properties::PropertyType type);
        void deletePropertyDefenition(std::string propertyName);
        std::string getPropertyDescription(std::string propertyName);
        enum Properties::PropertyType getPropertyType(std::string propertName);
        bool isPropertyDefined(std::string propertyName);
        void loadProperty(Properties & prop, std::string propertyName, std::string value);
        void loadPropertyFile(Properties & prop, std::string filename);
        void loadPropertyArgs(Properties &prop,int argc, char** argv);
    };
} } }


#endif	/* PROPERTIESMANAGER_H */

