/* Properties.cpp: Code for the Properties class, part of the GAFW CPPProperties Tool     
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
#include "CppProperties.h"

#include <iostream>

using namespace GAFW::Tools::CppProperties;
using namespace std;
Properties::Properties()
{
    
}
Properties::Properties(const Properties& orig)
{
    this->complexProperties=orig.complexProperties;
    this->floatProperties=orig.floatProperties;
    this->doubleProperties=orig.doubleProperties;
    this->intProperties=orig.intProperties;
    this->pointerProperties=orig.pointerProperties;
    this->stringProperties=orig.stringProperties;
    this->boolProperties=orig.boolProperties;
    this->propertyMap=orig.propertyMap;
}
Properties::~Properties()
{
    
}
void Properties::setProperty(string name,string value)
{
    //First check if exists and if exists is of type string
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=String) throw PropertyException("setProperty()",name,"Property is already set with a different type then string");
        else this->stringProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=String;
        this->stringProperties[name]=value;
    }
    
}
void Properties::setProperty(string name, int value)
{
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Int) throw PropertyException("setProperty()",name,"Property is already set with a different type then integer");
        else this->intProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Int;
        this->intProperties[name]=value;
    }
    
}
void Properties::setProperty(string name,float value)
{
    
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Float) throw PropertyException("setProperty()",name,"Property is already set with a different type then float");
        else this->floatProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Float;
        this->floatProperties[name]=value;
    }
}

void Properties::setProperty(string name,double value)
{
    
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Double) throw PropertyException("setProperty()",name,"Property is already set with a different type then double");
        else this->doubleProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Double;
        this->doubleProperties[name]=value;
    }
}

void Properties::setProperty(string name,bool value)
{
    
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Bool) throw PropertyException("setProperty()",name,"Property is already set with a different type then bool");
        else this->boolProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Bool;
        this->boolProperties[name]=value;
    }
}

void Properties::setProperty(string name,void * value)
{
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Pointer) throw PropertyException("setProperty()",name,"Property is already set with a different type then pointer");
        else this->pointerProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Pointer;
        this->pointerProperties[name]=value;
    }
    
}
void Properties::setProperty(string name,complex<float> value)
{
    if (this->isPropertySet(name))
    {
        if (this->getPropertyType(name)!=Complex) throw PropertyException("setProperty()",name,"Property is already set with a different type then complex");
        else this->complexProperties[name]=value;
    }
    else 
    {
        this->propertyMap[name]=Complex;
        this->complexProperties[name]=value;
    }
    
}
string Properties::getStringProperty(string name)
{
    if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
    else if (this->getPropertyType(name)!=String) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type string");
    
    return this->stringProperties[name];
}
int Properties::getIntProperty(string name)
{
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
    else if (this->getPropertyType(name)!=Int) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type int");
    
    return this->intProperties[name];
}
double Properties::getDoubleProperty(string name)
{
    
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
     else if (this->getPropertyType(name)!=Double) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type double");
     return this->doubleProperties[name];
}
float Properties::getFloatProperty(string name)
{
    
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
    else if (this->getPropertyType(name)!=Float) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type float");
    
    return this->floatProperties[name];
}
bool Properties::getBoolProperty(string name)
{
    
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
    else if (this->getPropertyType(name)!=Bool) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type bool");
    
    return this->boolProperties[name];
}

void *  Properties::getPointerProperty(string name)
{
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
     else if (this->getPropertyType(name)!=Pointer) 
        throw PropertyException("getProperty()",name,"Property exists but is not of type pointer");
    
    return this->pointerProperties[name];
}
complex<float> Properties::getComplexProperty(string name)
{
     if (!this->isPropertySet(name))
        throw PropertyException("getProperty()",name,"Property is not set");
     else if (this->getPropertyType(name)!=Complex) 
        throw  PropertyException("getProperty()",name,"Property exists but is not of type complex");
    
    return this->complexProperties[name];
}
enum Properties::PropertyType Properties::getPropertyType(string name)
{
    if (!this->isPropertySet(name))
        throw PropertyException("getPropertyType()",name,"Property is not set");
    return this->propertyMap[name];
}
void Properties::deleteProperty(std::string name)
{
    if (!this->isPropertySet(name))
        throw PropertyException("deleteProperty()",name,"Property is not set");
    switch (this->propertyMap[name])
    {
        case String:
            this->stringProperties.erase(name);
            break;
        case Complex:
            this->complexProperties.erase(name);
            break;
        case Float:
            this->floatProperties.erase(name);
            break;
        case Int:
            this->intProperties.erase(name);
            break;
        case Pointer:
            this->pointerProperties.erase(name);
            break;
        case Bool:
            this->boolProperties.erase(name);
            break;
            
        default:
            throw PropertyException("deleteProperty()",name,"BUG: Could not identify property type");
            
    }
    this->propertyMap.erase(name);
  
}
Properties & Properties::operator=(Properties& orig)
{
    this->complexProperties=orig.complexProperties;
    this->floatProperties=orig.floatProperties;
    this->doubleProperties=orig.doubleProperties;
    this->intProperties=orig.intProperties;
    this->pointerProperties=orig.pointerProperties;
    this->stringProperties=orig.stringProperties;
    this->propertyMap=orig.propertyMap;
    this->boolProperties=orig.boolProperties;

    // CHECK if we only delete what there was before
}
