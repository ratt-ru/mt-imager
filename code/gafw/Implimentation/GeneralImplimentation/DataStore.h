/* DataStore.h:  Header file for the DataStore class.
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

#ifndef __GEN_DATASTORE_H__
#define	__GEN_DATASTORE_H__
#include <stdarg.h>
#include <complex>

#include "gafw-impl.h"

namespace GAFW { namespace GeneralImplimentation
{
 class DataStore: public GAFW::FactoryAccess, public GAFW::Identity, public GAFW::LogFacility {
protected:
        
        ArrayDimensions dim;
        const DataTypeBase storetype;
private:
    DataStore(const DataStore & f):storetype(DataType<void>()){};
public:
        DataStore(Factory * factory, std::string nickname,std::string name,DataTypeBase type, ArrayDimensions dim);
        
        virtual DataTypeBase getType();
        
        inline ArrayDimensions getDimensions();
        
        virtual void getValues(PointerWrapperBase &pointer,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)=0;
        virtual void getValue(ValueWrapperBase &value, unsigned int noOfIndexes,va_list &vars)=0;
        virtual void getValue(std::vector<unsigned int> &position, ValueWrapperBase &value)=0;
        virtual void getSimpleArray(PointerWrapperBase &pointer)=0;
        virtual void setValue(ValueWrapperBase &value,unsigned int noOfIndexes,va_list &vars)=0;
        virtual void setValues(PointerWrapperBase &value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)=0;
        virtual void setValue(std::vector<unsigned int> &position, ValueWrapperBase &value)=0;
        virtual void setAllValues(PointerWrapperBase &values)=0;
        virtual size_t getSize()=0;
        virtual DataStoreSnapshotDescriptor createSnapshot(int id,int otherId)=0;
        virtual void deleteSnapshot(int id)=0;
        virtual DataStoreSnapshotDescriptor describeMySelf()=0;
        virtual bool isDataValid()=0;
        virtual void waitUntilDataValid()=0;
        virtual void setUnlinkedFromArray()=0;
        virtual void * allocMemeory()=0;

};
  
 inline ArrayDimensions DataStore::getDimensions()
{
    return this->dim;
}
 inline DataTypeBase DataStore::getType()
 {
     return this->storetype;
 }
} }//end of namespace

#endif	/* MATRIXDATASTORE_H */

