/* Module.h:  Definition and skeletal code for the Module 
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

#ifndef __PIL_MODULE_H__
#define	__PIL_MODULE_H__
#include "gafw.h"
#include <string>

namespace GAFW 
{
    

class Module: public FactoryAccess,public Identity, public LogFacility {
private:
   
public:
   Module(Factory *factory,std::string nickname,std::string name);
   virtual ~Module();
   virtual void reset()=0;
   Result * getOutput();
   virtual void calculate()=0;
   virtual void setInput(int inputNo, Result *res)=0;
   virtual Result * getOutput(int outputNo)=0; 
   void setInput( Result *res);
   
   virtual void resultRead(ProxyResult *,int snapshot_no)=0;
   std::string getMemberFullNickname(std::string memeberNickname);
   
   
   
};

inline Module::Module(Factory *factory,std::string nickname,std::string name="Module"):FactoryAccess(factory),Identity(nickname,name+"::Module")
{
    LogFacility::init();
    FactoryAccess::init();
       
    factory->registerIdentity(this);
}
inline Module::~Module()
{
    
}
inline void Module::setInput( Result *res)
{
    this->setInput(0,res);
}
inline Result * Module::getOutput()
{
    return this->getOutput(0);
}
inline std::string Module::getMemberFullNickname(std::string memberNickname)
{
    return this->getObjectName()+string(".")+memberNickname;
} 
}


#endif	/* MODULE_H */

