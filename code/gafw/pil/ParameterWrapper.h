/* Parameter.h:  Defines ParameterWrapper and ParameterWrapperBase classes.     
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

#ifndef __PIL_PARAMETERWRAPPER_H__
#define	__PIL_PARAMETERWRAPPER_H__
namespace GAFW {
class ParameterWrapperBase
{
public:
    const int parameterType;
    std::string parameterName;
    inline ParameterWrapperBase& setName(std::string parameterName);
    virtual ~ParameterWrapperBase();
protected:
    inline ParameterWrapperBase(const int pointerType);
    
};
template <class T>
class ParameterWrapper: public ParameterWrapperBase
{
public:
    T value;
    ParameterWrapper(std::string name, T value);
    ParameterWrapper();
    ParameterWrapper(std::string name);
    ParameterWrapper<T> & setNameAndValue(std::string name,T value);
    ParameterWrapper<T> & setValue(T value);
    //template <class C> 
    //void convertTo(ParameterWrapper<C> &to);
};

inline ParameterWrapperBase::ParameterWrapperBase(const int parameterType):parameterType(parameterType)
{
    
}

inline ParameterWrapperBase& ParameterWrapperBase::setName(std::string parameterName)
{
    this->parameterName=parameterName;
    return *this;
}
}
#endif	/* POINTERWRAPPER_H */

