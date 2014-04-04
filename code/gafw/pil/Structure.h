/* Structure.h:  Definition of the Structure     
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */

#ifndef __PIL_STRUCTURE_H__
#define	__PIL_STRUCTURE_H__
#include "gafw.h"
#include <string>
namespace GAFW
{
class Structure: public FactoryAccess,public Identity, public LogFacility {
private:
    
public:
   Structure(Factory *factory,std::string nickname,std::string name);
   virtual ~Structure();
   Array * getOutput();
   virtual void setInput(int inputNo, Array *inp)=0;
   virtual Array * getOutput(int outputNo)=0; 
   virtual void setInput(Array *inp)=0;
   inline std::string getMemberFullNickname(std::string memeberNickname);
};
inline Structure::Structure(Factory *factory,std::string nickname,std::string name="Structure"):FactoryAccess(factory),Identity(nickname,name+"::Structure")
{
    LogFacility::init();
    FactoryAccess::init();
    factory->registerIdentity(this);
}
inline Structure::~Structure()
{
    
}
inline void Structure::setInput( Array *inp)
{
    this->setInput(0,inp);
}
inline  Array * Structure::getOutput()
{
    return this->getOutput(0);
}
inline std::string Structure::getMemberFullNickname(std::string memberObjectname)
{
    return this->getObjectName()+"."+memberObjectname;
}

}
#endif	/* STRUCTURE_H */

