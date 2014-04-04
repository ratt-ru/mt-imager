/* GPUDataStore.cpp:  Implementation of the GPUDataStore class. 
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
#include "GPUafw.h"
#include "StoreManager.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

using namespace GAFW::GPU;
using namespace GAFW::GeneralImplimentation;
using namespace std;




GPUDataStore::GPUDataStore(GPUFactory *factory,GAFW::GeneralImplimentation::DataTypeBase &type, ArrayDimensions &dim,bool allocate): DataStore(factory,"Data store","DataStore",type, dim) 
{
 
    /* Lets create store manager according to type*/
    switch (type._type) 
    {
        case real_float:
            this->storeManager=new StoreManagerProper<float>(this->store);
            break;
        case real_double:
            this->storeManager=new StoreManagerProper<double>(this->store);
            break;
        case complex_double:
            this->storeManager=new StoreManagerProper<cuDoubleComplex>(this->store);
            break;
        case complex_float:
            this->storeManager=new StoreManagerProper<cuFloatComplex>(this->store);
            break;
        case real_int:
            this->storeManager=new StoreManagerProper<int>(this->store);
            break;
        case real_uint:
            this->storeManager=new StoreManagerProper<unsigned int>(this->store);
            break;
        case real_longint:
            this->storeManager=new StoreManagerProper<long int>(this->store);
            break;
        case real_ulongint:
            this->storeManager=new StoreManagerProper<unsigned long int>(this->store);
            break;
        case real_shortint:
            this->storeManager=new StoreManagerProper<short int>(this->store);
            break;
        case real_ushortint:
            this->storeManager=new StoreManagerProper<unsigned short int>(this->store);
            break;
        default:
            throw GeneralException("Store type is unsupported");
    }
    this->dataValid=true; //we always assume that we have an input
    this->unLinked=false;
    this->sizeofelement=this->storeManager->getSizeOfElement();
    //Now create the store
    if (allocate)
    {
        cudaError_t err;
        err=cudaHostAlloc(&this->store,this->dim.getTotalNoOfElements()*this->sizeofelement,cudaHostAllocPortable);
        if (err!=cudaSuccess) throw CudaException("Unable to allocate memory",err);
    }
    else this->store=NULL;
    
    this->snapshotsActive=0;
    
    this->totalElementsIndx=new unsigned int[dim.getNoOfDimensions()];
    unsigned int tot=1;
    for (int i=0;i<dim.getNoOfDimensions();i++)
    {
        this->totalElementsIndx[i]=tot;
        tot*=dim.getDimension(dim.getNoOfDimensions()-i-1);
    }
    
    
    //The beolw will go soon as it is supr inefficient
    
     tot=1;
   // this->store=malloc(this->dim.getTotalNoOfElements()*this->sizeofelement);
    for (int i=dim.getNoOfDimensions()-1;i>-1;i--)
    {
        this->totalElementsIndex.insert(this->totalElementsIndex.begin(),tot);
        tot*=dim.getDimension(i);
    }
    this->sizeOfArray=tot;
}

bool GPUDataStore::isDataValid()
{
    boost::mutex::scoped_lock lock (this->myMutex);
    return this->dataValid;
}
 void GPUDataStore::waitUntilDataValid()
 {
     boost::mutex::scoped_lock lock(this->myMutex);
     for (;!this->dataValid;)
         this->myCondition.wait(lock);
 }
 
 void GPUDataStore::setDataNotValid()
 {
    boost::mutex::scoped_lock lock (this->myMutex);
    this->dataValid=false;
    this->myCondition.notify_all();
 }
 void GPUDataStore::setDataAsValid()
 {
    boost::mutex::scoped_lock lock (this->myMutex);
    this->dataValid=true;
    this->myCondition.notify_all();
 }
 void * GPUDataStore::allocMemeory()
 {
     if (this->store==NULL)
     {
        cudaError_t err;
        err=cudaHostAlloc(&this->store,this->dim.getTotalNoOfElements()*this->sizeofelement,cudaHostAllocDefault);
        if (err!=cudaSuccess) throw CudaException("Unable to allocate memory",err);
     }
     else
         throw GeneralException("Allocation for already allocated store");

     return this->store;
     
 
 }


GPUDataStore::~GPUDataStore() 
{
    cudaError_t err;
   //Need to delete snapshots here
    
    
    if (this->store!=NULL)
    {
        err=cudaFreeHost(this->store);
        if (err!=cudaSuccess) throw CudaException("Unable to free memory",err);
    }
    
}

template<class U> U GPUDataStore::internal_getValue(unsigned int &NoOfIndexes,va_list &vars)
{
    U value;
    unsigned int pos=this->getPositionInSimpleArray(NoOfIndexes,vars);
    this->storeManager->getElement(pos,value);
    return value;
    
}
template<class U> void GPUDataStore::internal_getValues(U*values, unsigned& no_of_elements, bool& isColumnWiseStored,unsigned int &NoOfIndexes,va_list &vars)
{
    
    if (!isColumnWiseStored) throw GeneralException("Currently only column wise is accepted");
    unsigned int beginpos=this->getPositionInSimpleArray(NoOfIndexes,vars);
    if ((beginpos+no_of_elements)>this->dim.getTotalNoOfElements()) throw GeneralException("Request of out of bounds data");
    if ((this->storetype._type==(StoreType)(DataType<U>()._type))&&(typeid(U)!=typeid(std::complex<double>))&&(typeid(U)!=typeid(std::complex<float>)))
    {
        //We can make a large copy
        memcpy((void*)values,this->storeManager->getPositionPointer(beginpos),no_of_elements*this->sizeofelement);
    }
    else
    {
        for (unsigned int i=beginpos;i<beginpos+no_of_elements;i++)
        {
                this->storeManager->getElement(i,*(values+i));
        }
    }
}
    
template<class U> void GPUDataStore::internal_setValue(U &value,unsigned int &noOfIndexes,va_list &vars)
{
    if (this->snapshotsActive!=0) this->copy_on_write_snapshot();
    unsigned int pos=this->getPositionInSimpleArray(noOfIndexes,vars);
    this->storeManager->setElement(pos,value);
    
}
template<class U>  void GPUDataStore::internal_setValues(U *values,unsigned& no_of_elements, bool& isColumnWiseStored, unsigned int noOfIndexes,va_list &vars)
{
     if (this->snapshotsActive!=0) this->copy_on_write_snapshot();
    if (!isColumnWiseStored) throw GeneralException("Currently only column wise is accepted");
    unsigned int beginpos=this->getPositionInSimpleArray(noOfIndexes,vars);
    if ((beginpos+no_of_elements)>this->dim.getTotalNoOfElements()) throw GeneralException("Request of out of bounds data");
    if ((this->storetype._type==(StoreType)(DataType<U>()._type))&&(typeid(U)!=typeid(std::complex<double>))&&(typeid(U)!=typeid(std::complex<float>)))
    {
        //We can make a large copy
        memcpy(this->storeManager->getPositionPointer(beginpos),(void*)values,no_of_elements*this->sizeofelement);
    }
    else
    {
        for (unsigned int i=beginpos;i<beginpos+no_of_elements;i++)
        {
                this->storeManager->setElement(i,*(values+i));
        }
    }
}
    


template <class U> void GPUDataStore::internal_getSimpleArray(U * array)
{
    unsigned int elements=this->dim.getTotalNoOfElements();
    //We need to find  away how to think about memcopies
    // for now easy way out
    for (unsigned int i=0;i<elements;i++)
    {
        this->storeManager->getElement(i,*(array+i));
    }
    
}



template<class U> void GPUDataStore::internal_getValue(std::vector<unsigned int> &position, U &value)
{
    unsigned int pos=this->getPositionInSimpleArray(position);
    this->storeManager->getElement(pos,value);
    
}
template<class U> void GPUDataStore::internal_setValue(std::vector<unsigned int> &position, U &value)
{
     if (this->snapshotsActive!=0) this->copy_on_write_snapshot();
    unsigned int pos=this->getPositionInSimpleArray(position);
    this->storeManager->setElement(pos,value);
    
}
template<class U> void GPUDataStore::internal_setAllValues(U *values)
{
     if (this->snapshotsActive!=0) this->copy_on_write_snapshot();
    unsigned int elements=this->dim.getTotalNoOfElements();
    //We need to find  away how to think about memcopies
    // for now easy way out
    for (unsigned int i=0;i<elements;i++)
    {
        this->storeManager->setElement(i,*(values+i));
    }
    
}

DataStoreSnapshotDescriptor GPUDataStore::createSnapshot(int id,int otherId)
{
    stringstream s;
     s<<"Request to create snapshot no "<< id;
     this->logInfo(other,s.str());
    
    
    //Check if there is already such a snapshot id
    // we have to think harder here
     
    if (this->snapshots.count(id)!=0) return this->snapshots[id] ;
    //int fff=this->snapshots.size();
   // printf ("COMPPPP createSnap %p %f ",((float*)this->store),*((float*)this->store));
    s.str("");
     s<<"Creating snapshot no "<< id;
     this->logInfo(other,s.str());
    
    DataStoreSnapshotDescriptor d;
    d.dim=this->dim;
    d.parent=this;
    d.pointer=this->store;
    d.size=this->dim.getTotalNoOfElements()*this->sizeofelement;
    d.snapshot_id=id;
    d.type=(StoreType)this->storetype._type;
    d.otherId=otherId;
    this->snapshotsActive++;
    
    
    this->snapshots[id]=d;
    //The below should insert if does not exist or increase if exists
    if (this->snapshots_count_by_pointer.count(this->store)!=0)
        this->snapshots_count_by_pointer[this->store]++;
    else
        this->snapshots_count_by_pointer[this->store]=1;
    return d;
    
    
}
DataStoreSnapshotDescriptor GPUDataStore::describeMySelf()
{
    DataStoreSnapshotDescriptor d;
    d.dim=this->dim;
    d.parent=this;
    d.pointer=this->store;
    d.size=this->dim.getTotalNoOfElements()*this->sizeofelement;
    d.snapshot_id=-1;
    d.type=(StoreType)this->storetype._type;
    return d;
}

 void GPUDataStore::deleteSnapshot(int id)
{
    //Check that ID exists
     stringstream s;
     s<<"Deleting snapshot no "<< id;
     this->logInfo(other,s.str());
    if (this->snapshots.count(id)==0) {
        throw GeneralException("Snapshot was not found");
    } 
    DataStoreSnapshotDescriptor &d=this->snapshots[id];
    if (d.pointer!=this->store) //ie a copy has been made
    {   if (this->snapshots_count_by_pointer[d.pointer]--==0)
        {  
            cudaError_t err;
            err=cudaFreeHost(d.pointer);
            if (err!=cudaSuccess) throw CudaException("Unable to free memory",err);

           this->snapshots_count_by_pointer.erase(d.pointer);
        }
    }
    else
    {
        if (this->snapshots_count_by_pointer[d.pointer]--==0)
        {  
           this->snapshots_count_by_pointer.erase(d.pointer);
        }
         this->snapshotsActive--;
   
    }
    this->snapshots.erase(id);
    
}

void GPUDataStore::copy_on_write_snapshot()
{
    //this function is to be invoked only when we need to write value in store that is also a snapshot
    void *p;
    cudaError_t err;
    int size=this->dim.getTotalNoOfElements()*this->sizeofelement;
    err=cudaHostAlloc(&p,size,cudaHostAllocPortable);
    if (err!=cudaSuccess) throw CudaException("Unable to allocate memory",err);
    
    std::memcpy(p,this->store,size);
    this->store=p;
    this->snapshotsActive=0;
    
}
int GPUDataStore::getPositionInSimpleArray(vector <unsigned int> &position)
{
    
    int pos=0;
    for (int i=0;i<this->dim.getNoOfDimensions();i++)
    {
        pos+=position[i]*this->totalElementsIndex[i];
    }
    return pos;
}
unsigned int GPUDataStore::getPositionInSimpleArray(unsigned int &NoOfIndexes,va_list &var)
{
    
    int pos=0;
    if (NoOfIndexes>dim.getNoOfDimensions()) throw GeneralException("The no of indexes is larger then the no of dimensions");
    
    for (int i=0;i<NoOfIndexes;i++)
    {
        unsigned int position=va_arg(var,unsigned int );
        pos+=position*this->totalElementsIndx[i];
    }
    return pos;
}
void GPUDataStore::setUnlinkedFromArray()
{
    this->unLinked=true;
}
 bool GPUDataStore::canDelete()
{
    if ((this->unLinked)&&(this->snapshots.size()==0))
        return true;
    else
        return false;
               
}
 
 void GPUDataStore::getValues(PointerWrapperBase &pointer,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
 {
     switch (pointer.pointerType)
     {
         case std_complex_float:
             this->getValues(dynamic_cast<PointerWrapper<std::complex<float> > & >(pointer).pointer, no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_complex_double:
             this->getValues(dynamic_cast<PointerWrapper<std::complex<double> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case simple_complex_float:
             this->getValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case simple_complex_double:
             this->getValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_float:
             this->getValues(dynamic_cast<PointerWrapper<float > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_double:
             this->getValues(dynamic_cast<PointerWrapper<double> & >(pointer).pointer, no_of_elements,isColumnWiseStored,noOfIndexes,vars);
             break;
         case std_int:
             this->getValues(dynamic_cast<PointerWrapper<int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_int:
             this->getValues(dynamic_cast<PointerWrapper<unsigned int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_long_int:
             this->getValues(dynamic_cast<PointerWrapper<long int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_long_int:
             this->getValues(dynamic_cast<PointerWrapper<unsigned long int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_short_int:
             this->getValues(dynamic_cast<PointerWrapper<short int > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_short_int:
             this->getValues(dynamic_cast<PointerWrapper<unsigned short int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
     }
 }
 void GPUDataStore::getValue(ValueWrapperBase &value, unsigned int noOfIndexes,va_list &vars)
 {
     switch (value.valueType)
     {
         case std_complex_float:
             dynamic_cast<ValueWrapper<std::complex<float> > & >(value).value=this->getComplexFloatValue(noOfIndexes,vars);
             break;
         case std_complex_double:
             dynamic_cast<ValueWrapper<std::complex<double> > & >(value).value=this->getComplexDoubleValue(noOfIndexes,vars);
             break;
         case simple_complex_float:
             dynamic_cast<ValueWrapper<SimpleComplex<float> > & >(value).value=this->getSimpleComplexFloatValue(noOfIndexes,vars);
             break;
         case simple_complex_double:
             dynamic_cast<ValueWrapper<SimpleComplex<double> > & >(value).value=this->getSimpleComplexDoubleValue(noOfIndexes,vars);
             break;
         case std_float:
             dynamic_cast<ValueWrapper<float > & >(value).value=this->getFloatValue(noOfIndexes,vars);
             break;
         case std_double:
             dynamic_cast<ValueWrapper<double > & >(value).value=this->getDoubleValue(noOfIndexes,vars);
             break;
         case std_int:
             dynamic_cast<ValueWrapper<int> & >(value).value=this->getIntValue(noOfIndexes,vars);
             break;
         case std_unsigned_int:
             dynamic_cast<ValueWrapper<unsigned int > & >(value).value=this->getUIntValue(noOfIndexes,vars);
             break;
         case std_long_int:
             dynamic_cast<ValueWrapper<long int> & >(value).value=this->getLongValue(noOfIndexes,vars);
             break;
         case std_unsigned_long_int:
             dynamic_cast<ValueWrapper<unsigned long int > & >(value).value=this->getULongValue(noOfIndexes,vars);
             break;
         case std_short_int:
             dynamic_cast<ValueWrapper<short int> & >(value).value=this->getShortIntValue(noOfIndexes,vars);
             break;
         case std_unsigned_short_int:
             dynamic_cast<ValueWrapper<unsigned short int > & >(value).value=this->getUShorInttValue(noOfIndexes,vars);
             break;
         
     }
 }
 void GPUDataStore::getValue(std::vector<unsigned int> &position, ValueWrapperBase &value)
 {
     switch (value.valueType)
     {
         case std_complex_float:
             this->getValue(position, dynamic_cast<ValueWrapper<std::complex<float> > & >(value).value);
             break;
         case std_complex_double:
             this->getValue(position,dynamic_cast<ValueWrapper<std::complex<double> > & >(value).value);
             break;
         case simple_complex_float:
             //this->getValue(position,dynamic_cast<ValueWrapper<SimpleComplex<float> > & >(value).value);
             throw GeneralException("Unsupported");
             break;
         case simple_complex_double:
             throw GeneralException("Unsupported");
             //this->getValue(position,dynamic_cast<ValueWrapper<SimpleComplex<double> > & >(value).value);
             break;
         case std_float:
             this->getValue(position,dynamic_cast<ValueWrapper<float > & >(value).value);
             break;
         case std_double:
             this->getValue(position,dynamic_cast<ValueWrapper<double > & >(value).value);
             break;
         case std_int:
             this->getValue(position,dynamic_cast<ValueWrapper<int> & >(value).value);
             break;
         case std_unsigned_int:
             this->getValue(position,dynamic_cast<ValueWrapper<unsigned int > & >(value).value);
             break;
         case std_long_int:
             this->getValue(position,dynamic_cast<ValueWrapper<long int> & >(value).value);
             break;
         case std_unsigned_long_int:
             this->getValue(position,dynamic_cast<ValueWrapper<unsigned long int > & >(value).value);
             break;
         case std_short_int:
             this->getValue(position,dynamic_cast<ValueWrapper<short int> & >(value).value);
             break;
         case std_unsigned_short_int:
             this->getValue(position,dynamic_cast<ValueWrapper<unsigned short int > & >(value).value);
             break;
     }

 }
 void GPUDataStore::getSimpleArray(PointerWrapperBase &pointer)
 {
     switch(pointer.pointerType)
     {
         case std_complex_float:
             this->getSimpleArray(dynamic_cast<PointerWrapper<std::complex<float> > & >(pointer).pointer);
             break;
         case std_complex_double:
             this->getSimpleArray(dynamic_cast<PointerWrapper<std::complex<double> > & >(pointer).pointer);
             break;
         case simple_complex_float:
             throw GeneralException("Unsupported");
             
             //this->getSimpleArray(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer);
             break;
         case simple_complex_double:
             throw GeneralException("Unsupported");
             
             //this->getSimpleArray(dynamic_cast<PointerWrapper<SimpleComplex<double> > & >(pointer).pointer);
             break;
         case std_float:
             this->getSimpleArray(dynamic_cast<PointerWrapper<float > & >(pointer).pointer);
             break;
         case std_double:
             this->getSimpleArray(dynamic_cast<PointerWrapper<double> & >(pointer).pointer);
             break;
         case std_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<int> & >(pointer).pointer);
             break;
         case std_unsigned_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<unsigned int> & >(pointer).pointer);
             break;
         case std_long_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<long int> & >(pointer).pointer);
             break;
         case std_unsigned_long_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<unsigned long int> & >(pointer).pointer);
             break;
         case std_short_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<short int > & >(pointer).pointer);
             break;
         case std_unsigned_short_int:
             this->getSimpleArray(dynamic_cast<PointerWrapper<unsigned short int> & >(pointer).pointer);
             break;
     }

 }
 void GPUDataStore::setValue(ValueWrapperBase &value,unsigned int noOfIndexes,va_list &vars)
 {
     switch (value.valueType)
     {
         case std_complex_float:
             this->setValue( dynamic_cast<ValueWrapper<std::complex<float> > & >(value).value,noOfIndexes,vars);
             break;
         case std_complex_double:
             this->setValue(dynamic_cast<ValueWrapper<std::complex<double> > & >(value).value,noOfIndexes,vars);
             break;
         case simple_complex_float:
             this->setValue(dynamic_cast<ValueWrapper<SimpleComplex<float> > & >(value).value,noOfIndexes,vars);
             //throw GeneralException("Unsupported");
             break;
         case simple_complex_double:
             //throw GeneralException("Unsupported");
             this->setValue(dynamic_cast<ValueWrapper<SimpleComplex<double> > & >(value).value,noOfIndexes,vars);
             break;
         case std_float:
             this->setValue(dynamic_cast<ValueWrapper<float > & >(value).value,noOfIndexes,vars);
             break;
         case std_double:
             this->setValue(dynamic_cast<ValueWrapper<double > & >(value).value,noOfIndexes,vars);
             break;
         case std_int:
             this->setValue(dynamic_cast<ValueWrapper<int> & >(value).value,noOfIndexes,vars);
             break;
         case std_unsigned_int:
             this->setValue(dynamic_cast<ValueWrapper<unsigned int > & >(value).value,noOfIndexes,vars);
             break;
         case std_long_int:
             this->setValue(dynamic_cast<ValueWrapper<long int> & >(value).value,noOfIndexes,vars);
             break;
         case std_unsigned_long_int:
             this->setValue(dynamic_cast<ValueWrapper<unsigned long int > & >(value).value,noOfIndexes,vars);
             break;
         case std_short_int:
             this->setValue(dynamic_cast<ValueWrapper<short int> & >(value).value,noOfIndexes,vars);
             break;
         case std_unsigned_short_int:
             this->setValue(dynamic_cast<ValueWrapper<unsigned short int > & >(value).value,noOfIndexes,vars);
             break;
     }
 }
 void GPUDataStore::setValues(PointerWrapperBase &pointer,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
 {
     switch(pointer.pointerType)
     {
         case std_complex_float:
             this->setValues(dynamic_cast<PointerWrapper<std::complex<float> > & >(pointer).pointer, no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_complex_double:
             this->setValues(dynamic_cast<PointerWrapper<std::complex<double> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case simple_complex_float:
             this->setValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case simple_complex_double:
             this->setValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_float:
             this->setValues(dynamic_cast<PointerWrapper<float > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_double:
             this->setValues(dynamic_cast<PointerWrapper<double> & >(pointer).pointer, no_of_elements,isColumnWiseStored,noOfIndexes,vars);
             break;
         case std_int:
             this->setValues(dynamic_cast<PointerWrapper<int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_int:
             this->setValues(dynamic_cast<PointerWrapper<unsigned int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_long_int:
             this->setValues(dynamic_cast<PointerWrapper<long int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_long_int:
             this->setValues(dynamic_cast<PointerWrapper<unsigned long int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_short_int:
             this->setValues(dynamic_cast<PointerWrapper<short int > & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
         case std_unsigned_short_int:
             this->setValues(dynamic_cast<PointerWrapper<unsigned short int> & >(pointer).pointer,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
             break;
     }

 }
 void GPUDataStore::setValue(std::vector<unsigned int> &position, ValueWrapperBase &value)
 {
     switch (value.valueType)
     {
         case std_complex_float:
             this->setValue(position, dynamic_cast<ValueWrapper<std::complex<float> > & >(value).value);
             break;
         case std_complex_double:
             this->setValue(position,dynamic_cast<ValueWrapper<std::complex<double> > & >(value).value);
             break;
         case simple_complex_float:
             //this->setValue(position,dynamic_cast<ValueWrapper<SimpleComplex<float> > & >(value).value);
             throw GeneralException("Unsupported");
             break;
         case simple_complex_double:
             throw GeneralException("Unsupported");
             //this->setValue(position,dynamic_cast<ValueWrapper<SimpleComplex<double> > & >(value).value);
             break;
         case std_float:
             this->setValue(position,dynamic_cast<ValueWrapper<float > & >(value).value);
             break;
         case std_double:
             this->setValue(position,dynamic_cast<ValueWrapper<double > & >(value).value);
             break;
         case std_int:
             this->setValue(position,dynamic_cast<ValueWrapper<int> & >(value).value);
             break;
         case std_unsigned_int:
             this->setValue(position,dynamic_cast<ValueWrapper<unsigned int > & >(value).value);
             break;
         case std_long_int:
             this->setValue(position,dynamic_cast<ValueWrapper<long int> & >(value).value);
             break;
         case std_unsigned_long_int:
             this->setValue(position,dynamic_cast<ValueWrapper<unsigned long int > & >(value).value);
             break;
         case std_short_int:
             this->setValue(position,dynamic_cast<ValueWrapper<short int> & >(value).value);
             break;
         case std_unsigned_short_int:
             this->setValue(position,dynamic_cast<ValueWrapper<unsigned short int > & >(value).value);
             break;
     }
 }
 void GPUDataStore::setAllValues(PointerWrapperBase &pointer)
 {
     switch(pointer.pointerType)
     {
         case std_complex_float:
             this->setAllValues(dynamic_cast<PointerWrapper<std::complex<float> > & >(pointer).pointer);
             break;
         case std_complex_double:
             this->setAllValues(dynamic_cast<PointerWrapper<std::complex<double> > & >(pointer).pointer);
             break;
         case simple_complex_float:
             throw GeneralException("Unsupported");
             
             //this->setAllValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer);
             break;
         case simple_complex_double:
             throw GeneralException("Unsupported");
             
             //this->setAllValues(dynamic_cast<PointerWrapper<SimpleComplex<float> > & >(pointer).pointer);
             break;
         case std_float:
             this->setAllValues(dynamic_cast<PointerWrapper<float > & >(pointer).pointer);
             break;
         case std_double:
             this->setAllValues(dynamic_cast<PointerWrapper<double> & >(pointer).pointer);
             break;
         case std_int:
             this->setAllValues(dynamic_cast<PointerWrapper<int> & >(pointer).pointer);
             break;
         case std_unsigned_int:
             this->setAllValues(dynamic_cast<PointerWrapper<unsigned int> & >(pointer).pointer);
             break;
         case std_long_int:
             this->setAllValues(dynamic_cast<PointerWrapper<long int> & >(pointer).pointer);
             break;
         case std_unsigned_long_int:
             this->setAllValues(dynamic_cast<PointerWrapper<unsigned long int> & >(pointer).pointer);
             break;
         case std_short_int:
             this->setAllValues(dynamic_cast<PointerWrapper<short int > & >(pointer).pointer);
             break;
         case std_unsigned_short_int:
             this->setAllValues(dynamic_cast<PointerWrapper<unsigned short int> & >(pointer).pointer);
             break;
        }
 }
    