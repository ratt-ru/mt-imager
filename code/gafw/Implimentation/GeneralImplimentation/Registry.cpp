/* Registry.cpp:  Implementation of the Registry.
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
#include "gafw-impl.h"
#include <string>
using namespace std;
using namespace GAFW::GeneralImplimentation;
Registry::Registry():Identity("Registry","Registry")
{
    LogFacility::init();
}
Registry::~Registry()
{
    this->objectnameRegistry.clear();
}
bool Registry::isObjectnameRegistered(string objectname)
{
    return (this->objectnameRegistry.count(objectname)==0)?false:true;
}
 std::string Registry::getParentObjectname(std::string objectname)
{
     int objectname_size=objectname.size();
     int pos;
     for (pos=objectname_size;pos>0;pos--)
     {
         if (objectname[pos]=='.') break;
     }
     return objectname.substr(0,pos);
}
bool Registry::isParentRegistered(std::string objectname)
{
    std::string parentObjectname=Registry::getParentObjectname(objectname);
    return (parentObjectname=="")?true:(this->objectnameRegistry.count(parentObjectname));
}
void Registry::registerIdentity(Identity *ob)
{
    string objectname=ob->objectName;
    this->logDebug(builder,this,string("Registering ")+ob->objectName);
    if (isObjectnameRegistered(objectname))
    {
        this->logError(builder,string("Objectname ")+objectname + string(" is already in use. Throwing exception"));
        throw GeneralException(string("Objectname ")+objectname + string(" already registered"));
    }
    Identity *parent=NULL;
    string parentObjectname=Registry::getParentObjectname(objectname);
    if (parentObjectname!="")
    {
        if (!isObjectnameRegistered(parentObjectname))
        {
                this->logError(builder,string("Parent Objectname ")+parentObjectname + string(" is not registered. Throwing exception"));
                throw GeneralException(string("Parent Objectname ")+parentObjectname + string(" is not registered"));
        }
        parent=const_cast<Identity *>(this->objectnameRegistry[parentObjectname].pointer);
    
    }
    this->objectnameRegistry[objectname]=ObjectData(ob,parent);
    this->objectnameRegistry[parentObjectname].children.push_back(ob);

    this->logDebug(builder,string("Successfully registered ")+objectname);
}
Identity* Registry::getIdentity(std::string objectname)
{
    this->logDebug(builder,this,string("getIdentity(): Getting Object with objectname: ")+objectname);
    if (!isObjectnameRegistered(objectname))
    {
        this->logError(builder,string("getIdentity(): Object with objectname ")+objectname+"does not exist. Throwing exception");
        throw GeneralException(string("Object with objectname ")+objectname+"does not exist.");
    }
    ObjectData &data=this->objectnameRegistry[objectname];
    
    this->logDebug(builder,this,"getIdentity(): Found and returning Object");
    return const_cast<Identity *>(data.pointer);
}
Identity *Registry::getParent(std::string objectname)
{
    this->logDebug(builder,this,string("getParent(): Getting Parent of Object with objectname: ")+objectname);
    if (!isObjectnameRegistered(objectname))
    {
        this->logError(builder,string("getParent(): Object with objectname ")+objectname+"does not exist. Throwing exception");
        throw GeneralException(string("Object with objectname ")+objectname+"does not exist.");
    }
    ObjectData &data=this->objectnameRegistry[objectname];
    
    this->logDebug(builder,this,"getParent(): Found and returning parent");
    return const_cast<Identity *>(data.parent);
}


    