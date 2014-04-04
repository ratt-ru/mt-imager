/* PropertiesManager.cpp: Code for the PropertyManager class, part of the GAFW CPPProperties Tool     
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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/regex.hpp>
using namespace GAFW::Tools::CppProperties;
using namespace std;
using namespace boost;

PropertiesManager::PropertiesManager() 
{

}

PropertiesManager::PropertiesManager(const PropertiesManager& orig) 
{

}

PropertiesManager::~PropertiesManager() 
{

}

void PropertiesManager::addPropertyDefenition(std::string propertyName, std::string propertyDescription,enum Properties::PropertyType type)
{
    if (this->PropertyDefenitions.count(propertyName)!=0)
    {
        if (this->PropertyDefenitions[propertyName].first!=propertyDescription ||
             this->PropertyDefenitions[propertyName].second!=type)
                throw PropertyException("PropertiesManager::addPropertyDefenition()",propertyName,"Property already defined differently");  
        else return;
 
    }
           
    this->PropertyDefenitions[propertyName]=pair<string, enum Properties::PropertyType>(propertyDescription,type);
}

void PropertiesManager::changePropertyDescription(std::string propertyName, std::string propertyDescription)
{
    if (this->PropertyDefenitions.count(propertyName)==0)
        throw PropertyException("PropertiesManager:changePropertyDescription()",propertyName,"Property is not defined");
    this->PropertyDefenitions[propertyName].first=propertyDescription;
}

void PropertiesManager::changePropertyType(std::string propertyName,enum Properties::PropertyType type)
{
    if (this->PropertyDefenitions.count(propertyName)==0)
        throw PropertyException("PropertiesManager:changePropertyType()",propertyName,"Property is not defined");
    this->PropertyDefenitions[propertyName].second=type;
}

void PropertiesManager::deletePropertyDefenition(std::string propertyName)
{
    this->PropertyDefenitions.erase(propertyName);
}
string PropertiesManager::getPropertyDescription(std::string propertyName)
{
    if (this->PropertyDefenitions.count(propertyName)==0)
        throw PropertyException("PropertiesManager:getPropertyDescription()",propertyName,"Property is not defined");
    return this->PropertyDefenitions[propertyName].first;
}
enum Properties::PropertyType PropertiesManager::getPropertyType(std::string propertyName)
{
    if (this->PropertyDefenitions.count(propertyName)==0)
        throw PropertyException("PropertiesManager:getPropertyType()",propertyName,"Property is not defined");
    return this->PropertyDefenitions[propertyName].second;

}
bool PropertiesManager::isPropertyDefined(std::string propertyName)
{
    return  this->PropertyDefenitions.count(propertyName);
}
void PropertiesManager::loadProperty(Properties & prop, std::string propertyName, std::string value)
{
    enum Properties::PropertyType type;
    if (this->isPropertyDefined(propertyName))
        type=this->getPropertyType(propertyName);
    else
        type=Properties::String;
    //We now need to convert values and register to prop
    switch (type)
    {
        case Properties::String:
            //Simplest case as no conversion required
            prop.setProperty(propertyName,value);
            break;
        case Properties::Int:
        {    //This code will ned to be made more robust .. some day
            int intValue=(int)strtol(value.c_str(),NULL,0);
            prop.setProperty(propertyName,intValue);
        }
            break;
        case Properties::Pointer:
        {   
            // A bit useless but we impliment:)
            void * pointerValue=(void *)strtol(value.c_str(),NULL,0);
            prop.setProperty(propertyName,pointerValue);
        }   
            break;
        case Properties::Float:
        {   
            float floatValue=(float)atof(value.c_str());
            prop.setProperty(propertyName,floatValue);
        }
            break;
        case Properties::Bool:
        {
            bool boolValue;
            if ((value==string("0"))||(value==string("false")))
                    boolValue=false;
            else if ((value==string("1"))||(value==string("true")))
                    boolValue=true;
            else
                throw PropertyException("PropertiesManager::loadProperties()",propertyName,"Unable to decode to a bool value");
            prop.setProperty(propertyName,boolValue);
        }
            break;
        case Properties::Complex:
            throw PropertyException("PropertiesManager::loadProperties()",propertyName,"Loading of complex properties from string is not yet implemented");
            break;
        default:
            throw PropertyException("PropertiesManager::loadProperties()",propertyName,"BUG:: Unable to identify property Type");
            break;
    }   
    
}

void PropertiesManager::loadPropertyFile(Properties & prop, std::string filename)
{
    regex onlyspace("^[[:space:]]*$");
    regex reg_comment("^[[:space:]]*[#!].*$");
    regex prop_genformat("^[[:space:]]*([^[:space:]\\=]+)[[:space:]]*=[[:space:]]*([^[:space:]]*)$");
    ifstream in;
    in.open(filename.c_str());
    if (!in)
        throw PropertyException("PropertiesManager::loadPropertyFile()","",string("Unable to open properties file: ")+filename);
    string line;
    do
    {
        getline(in,line);
       
        //is this a comment?? or just white space
        if (regex_match(line,reg_comment)||regex_match(line,onlyspace))
        {
            continue;
        }
        //Check general format
        match_results<std::string::const_iterator> what;
        if (!regex_search(line,what,prop_genformat))
        {
            throw PropertyException("PropertiesManager::loadPropertyFile()","Unknown", string("Could not parse line: ")+line);
        }
        loadProperty(prop,what[1],what[2]);
        
    } while(!in.eof());
    
   
    
    
}
void PropertiesManager::loadPropertyArgs(Properties &prop, int argc, char** argv)
{
    regex prop_genformat("^[[:space:]]*([^[:space:]\\=]+)[[:space:]]*=(.+)$");
    regex boolprop_genformat("^[[:space:]]*\\-([^[:space:]\\=]+)[[:space:]]*$");
    for (int i=1; i<argc; i++)
    {
        string arg(argv[i]);
        match_results<std::string::const_iterator> what;
        if (regex_search(arg,what,boolprop_genformat))
        {
            //Ok this is a bool property that must be set to true
            //but is it registered to bool
            if (this->isPropertyDefined(what[1]))
                if (this->getPropertyType(what[1])==Properties::Bool)
                {
                    this->loadProperty(prop,what[1],"true");
                }
                else 
                   throw PropertyException("PropertiesManager::loadPropertyArgs()",what[1],"Property is not of bool type");
            else
                throw PropertyException("PropertiesManager::loadPropertyArgs()",what[1],"Property is not registered");
            continue;  
        }
            
        if (!regex_search(arg,what,prop_genformat))
        {
            throw PropertyException("PropertiesManager::loadPropertyArgs()","Unknown", string("Could not parse argument: ")+arg);
        }
        loadProperty(prop,what[1],what[2]);
    }
}