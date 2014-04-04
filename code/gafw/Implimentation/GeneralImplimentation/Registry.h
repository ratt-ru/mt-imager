/* Registry.h:  Definition of the Registry Class
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
#ifndef __GEN_REGISTRY_H__
#define	__GEN_REGISTRY_H__
#include <map>
namespace GAFW { namespace GeneralImplimentation 
{

    

    class Registry : public Identity, public  LogFacility{
    private:
        Registry(const Registry& orig);
    protected:
        class ObjectData
        {
        public:

            const Identity * pointer;
            const Identity * parent;
            std::vector <Identity *> children;
            ObjectData(Identity *pointer,Identity * parent):pointer(pointer),parent(parent) {children.clear();};
            ObjectData():pointer(NULL),parent(NULL) {}; //required by the map
        };
        std::map <std::string,ObjectData> objectnameRegistry;

    public:
        Registry();
        virtual ~Registry();
        bool isObjectnameRegistered(std::string objectname);
        bool isParentRegistered(std::string objectname);
        void registerIdentity(Identity *);
        Identity* getIdentity(std::string objectname);
        static std::string getParentObjectname(std::string objectname);
        Identity *getParent(std::string objectname);

        /*
        void registerArray(Array * matrix);
        void registerOperator(ArrayOperator *matrixOperator);
        void registerProxyResult(ProxyResult *proxy);
        Array * getArray(std::string objectname);
        ArrayOperator * getOperator(std::string objectname);
        ProxyResult *getProxyResult(std::string nickanme);
        */

    };
}}

#endif	/* REGISTRY_H */

