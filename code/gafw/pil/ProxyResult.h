/* ProxyResult.h:  Definition of the ProxyResult class
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

#ifndef __PIL_PROXYRESULT_H__
#define	__PIL_PROXYRESULT_H__
#include "gafw.h"
#include <map>
namespace GAFW 
{
    

class ProxyResult:  public Result  {
private:
    ProxyResult(const ProxyResult& orig){};
protected:
    inline ProxyResult();
    inline ProxyResult(Factory *factory,std::string objectName, std::string name);
public:
    virtual void setBind(Result *)=0;
    virtual Result*  getBind()=0;
};

inline ProxyResult::ProxyResult()
{
    throw Bug("This function is available for programming convenience only and should never be called");
}
inline ProxyResult::ProxyResult(Factory *factory,std::string objectName, std::string name):Result(factory,objectName,name)
{
    
}

} //end of namespace
#endif	/* COPYRESULTMATRIX_H */

