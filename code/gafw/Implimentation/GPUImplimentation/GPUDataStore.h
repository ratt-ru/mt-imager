/* GPUDataStore.h:  Definition of the GPUDataStore class. 
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

#ifndef __GPUMATRIXDATASTORE_H__
#define	__GPUMATRIXDATASTORE_H__
#include "StoreManager.h"
#include <complex>
#include <map>
#include <typeinfo>
#include "GPUafw.h"
#include <boost/thread.hpp> 


namespace GAFW { namespace GPU
{
    

 class GPUDataStore: public GAFW::GeneralImplimentation::DataStore
{
     
 private:
//     GPUMatrixDataStore(const GPUMatrixDataStore& orig){}; //copying disallowed
 protected:
     //
     bool unLinked;
     bool dataValid;
     boost::mutex myMutex;
     boost::condition_variable myCondition;
     
    std::vector <unsigned int> totalElementsIndex;
    unsigned int *totalElementsIndx;
    std::map<int,GAFW::GeneralImplimentation::DataStoreSnapshotDescriptor> snapshots;
    std::map<void*,int> snapshots_count_by_pointer;
    int snapshotsActive; // this bool will be true if current data is also
                                //part of a sniopshot.. If a setValue is onvoked
                                //then a copy is required. Thus COW (Copy-on-Write)
                                //is implemented
    int sizeOfArray;
    GPUDataStore(GPUFactory * factory, DataTypeBase &type,GAFW::ArrayDimensions &dim,bool allocate=true);
    
    ~GPUDataStore();
    void *store;
    GAFW::GPU::StoreManager * storeManager;                   
    
    int sizeofelement;
    void copy_on_write_snapshot();
    //friend class GPUMatrixDataStoreTest;
    int getPositionInSimpleArray(std::vector<unsigned int> &position);
    unsigned int getPositionInSimpleArray(unsigned int &NoOfIndexes,va_list &vars);
    //template<class T,GAFW::StoreType Type> T* cast_store();  
    template<class U> void internal_getSimpleArray(U *array);
    template<class U> void internal_getValue(std::vector<unsigned int> &position, U &value);
    template<class U> void internal_setValue(std::vector<unsigned int> &position, U &value);
    template<class U> void internal_setAllValues(U *values);
    
    template<class U> U internal_getValue(unsigned int &NoOfIndexes,va_list &vars);
    template<class U> void internal_getValues(U*values, unsigned& no_of_elements, bool& isColumnWiseStored,unsigned int &NoOfIndexes,va_list &vars);
    
    template<class U> void internal_setValue(U &value,unsigned int &noOfIndexes,va_list &vars);
    template<class U>  void internal_setValues(U *value,unsigned& no_of_elements, bool& isColumnWiseStored, unsigned int noOfIndexes,va_list &vars);
    
    
public:
    bool isDataValid();
    void waitUntilDataValid();
    void setDataNotValid();
    void setDataAsValid();
     virtual void * allocMemeory();
    
    
    
    friend class GPUFactory;
    friend class GAFW::GPU::DataDescriptor; //required as to delete
    virtual inline size_t getSize();
    virtual GAFW::GeneralImplimentation::DataStoreSnapshotDescriptor createSnapshot(int id,int otherId);
    virtual void deleteSnapshot(int id);
    virtual GAFW::GeneralImplimentation::DataStoreSnapshotDescriptor describeMySelf();
    virtual inline void getValue(std::vector<unsigned int> &position, float &value);
    virtual inline void getValue(std::vector<unsigned int> &position, double &value);
    virtual inline void getValue(std::vector<unsigned int> &position, std::complex<float> &value);
    virtual inline void getValue(std::vector<unsigned int> &position, std::complex<double> &value);
    virtual inline void getValue(std::vector<unsigned int> &position, int &value);
    virtual inline void getValue(std::vector<unsigned int> &position, unsigned int &value);
    virtual inline void getValue(std::vector<unsigned int> &position, short int &value);
    virtual inline void getValue(std::vector<unsigned int> &position, unsigned short int &value);
    virtual inline void getValue(std::vector<unsigned int> &position, long int &value);
    virtual inline void getValue(std::vector<unsigned int> &position, unsigned long int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, float &value);
    virtual inline void setValue(std::vector<unsigned int> &position, double &value);
    virtual inline void setValue(std::vector<unsigned int> &position, std::complex<float> &value);
    virtual inline void setValue(std::vector<unsigned int> &position, std::complex<double> &value);
    virtual inline void setValue(std::vector<unsigned int> &position, int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, unsigned int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, short int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, unsigned short int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, long int &value);
    virtual inline void setValue(std::vector<unsigned int> &position, unsigned long int &value);
    virtual inline void getSimpleArray(float * values);
    virtual inline void getSimpleArray(double * values);
    virtual inline void getSimpleArray(std::complex<float> *values);
    virtual inline void getSimpleArray(std::complex<double> *values);
    virtual inline void getSimpleArray( int *values);
    virtual inline void getSimpleArray(unsigned int *values);
    virtual inline void getSimpleArray(short int *values);
    virtual inline void getSimpleArray(unsigned short int *values);
    virtual inline void getSimpleArray(long int *values);
    virtual inline void getSimpleArray(unsigned long int  *values);
    virtual inline void setAllValues(float * values);
    virtual inline void setAllValues(double * values);
    virtual inline void setAllValues(std::complex<float> *values);
    virtual inline void setAllValues(std::complex<double> *values);
    virtual inline void setAllValues( int *values);
    virtual inline void setAllValues(unsigned int *values);
    virtual inline void setAllValues(short int *values);
    virtual inline void setAllValues(unsigned short int *values);
    virtual inline void setAllValues(long int *values);
    virtual inline void setAllValues(unsigned long int  *values);
    
    virtual float getFloatValue(unsigned int noOfIndexes,va_list &vars);
    virtual double getDoubleValue(unsigned int noOfIndexes,va_list &vars);
    virtual int getIntValue(unsigned int noOfIndexes,va_list &vars);
    virtual unsigned getUIntValue(unsigned int noOfIndexes,va_list &vars);
    virtual short getShortIntValue(unsigned int noOfIndexes,va_list &vars);
    virtual unsigned short getUShorInttValue(unsigned int noOfIndexes,va_list &vars);
    virtual long getLongValue(unsigned int noOfIndexes,va_list &vars);
    virtual unsigned long getULongValue(unsigned int noOfIndexes,va_list &vars);
    virtual std::complex<double> getComplexDoubleValue(unsigned int noOfIndexes,va_list &vars);
    virtual std::complex<float> getComplexFloatValue(unsigned int noOfIndexes,va_list &vars);
    virtual GAFW::SimpleComplex<double> getSimpleComplexDoubleValue(unsigned int noOfIndexes,va_list &vars);
    virtual GAFW::SimpleComplex<float> getSimpleComplexFloatValue(unsigned int noOfIndexes,va_list &vars);

    virtual inline void  getValues(float * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(double * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(int * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(unsigned int * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(short * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(unsigned short * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(long * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(unsigned long * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(std::complex<double> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(std::complex<float> *value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(GAFW::SimpleComplex<float> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual inline void getValues(GAFW::SimpleComplex<double> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    
    virtual inline void setValue(float &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(double &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(std::complex<float> &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(std::complex<double> &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(unsigned int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(short int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(unsigned short int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(long int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(unsigned long int &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(GAFW::SimpleComplex<float> &value,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValue(GAFW::SimpleComplex<double> &value,unsigned int noOfIndexes,va_list &vars);

    virtual inline void setValues(float *value,unsigned no_of_elements, bool isColumnWiseStored, unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(double *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(std::complex<float> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(std::complex<double> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(unsigned int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(short int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(unsigned short int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(long int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(unsigned long int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(GAFW::SimpleComplex<float> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual inline void setValues(GAFW::SimpleComplex<double> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    
    virtual void setUnlinkedFromArray();
    virtual bool canDelete();
    
    virtual void getValues(PointerWrapperBase &pointer,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual void getValue(ValueWrapperBase &value, unsigned int noOfIndexes,va_list &vars);
    virtual void getValue(std::vector<unsigned int> &position, ValueWrapperBase &value);
    virtual void getSimpleArray(PointerWrapperBase &pointer);
    virtual void setValue(ValueWrapperBase &value,unsigned int noOfIndexes,va_list &vars);
    virtual void setValues(PointerWrapperBase &value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars);
    virtual void setValue(std::vector<unsigned int> &position, ValueWrapperBase &value);
    virtual void setAllValues(PointerWrapperBase &values);
        
     
};

inline void GPUDataStore::getSimpleArray(float * values)
{
    this->internal_getSimpleArray<float>(values);
}

inline void GPUDataStore::getSimpleArray(double * values)
{
    this->internal_getSimpleArray<double>(values);
}

inline void GPUDataStore::getSimpleArray(std::complex<float> *values)
{
    this->internal_getSimpleArray<std::complex<float> >(values);
}


inline void GPUDataStore::getSimpleArray(std::complex<double> *values)
{
    this->internal_getSimpleArray<std::complex<double> >(values);
}
inline void GPUDataStore::getSimpleArray( int *values)
{
    this->internal_getSimpleArray<int>(values);
}
inline void GPUDataStore::getSimpleArray(unsigned int *values)
{
    this->internal_getSimpleArray<unsigned int>(values);
}
inline void GPUDataStore::getSimpleArray(short int *values)
{
    this->internal_getSimpleArray<short int>(values);
}
inline void GPUDataStore::getSimpleArray(unsigned short int *values)
{
    this->internal_getSimpleArray<unsigned short int> (values);
}
inline void GPUDataStore::getSimpleArray(long int *values)
{
    this->internal_getSimpleArray<long int>(values);
}
inline void GPUDataStore::getSimpleArray(unsigned long int  *values)
{
    this->internal_getSimpleArray<unsigned long int>(values);
}
inline void GPUDataStore::setAllValues(float * values)
{
    this->internal_setAllValues<float>(values);
}

inline void GPUDataStore::setAllValues(double * values)
{
    this->internal_setAllValues<double>(values);
}

inline void GPUDataStore::setAllValues(std::complex<float> *values)
{
    this->internal_setAllValues<std::complex<float> >(values);
}


inline void GPUDataStore::setAllValues(std::complex<double> *values)
{
    this->internal_setAllValues<std::complex<double> >(values);
}
inline void GPUDataStore::setAllValues( int *values)
{
    this->internal_setAllValues<int>(values);
}
inline void GPUDataStore::setAllValues(unsigned int *values)
{
    this->internal_setAllValues<unsigned int>(values);
}
inline void GPUDataStore::setAllValues(short int *values)
{
    this->internal_setAllValues<short int>(values);
}
inline void GPUDataStore::setAllValues(unsigned short int *values)
{
    this->internal_setAllValues<unsigned short int >(values);
}
inline void GPUDataStore::setAllValues(long int *values)
{
    this->internal_setAllValues<long int>(values);
}
inline void GPUDataStore::setAllValues(unsigned long int  *values)
{
    this->internal_setAllValues<unsigned long int>(values);
}


inline void GPUDataStore::getValue(std::vector<unsigned int> &position,std::complex<float> &value)
{
    this->internal_getValue<std::complex<float> >(position,value);
}

inline void GPUDataStore::getValue(std::vector<unsigned int> &position,std::complex<double> &value)
{
    this->internal_getValue<std::complex<double> >(position,value);
}

inline void GPUDataStore::getValue(std::vector<unsigned int> &position, float &value)
{
    this->internal_getValue<float>(position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position, double &value)
{
    this->internal_getValue<double>(position,value);
}


inline void GPUDataStore::getValue(std::vector<unsigned int> &position, int &value)
{
    this->internal_getValue<int>(position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position,unsigned int &value)
{
    this->internal_getValue<unsigned int>(position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position,short int &value)
{
    this->internal_getValue<short int>(position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position,unsigned short int &value)
{
    this->internal_getValue<unsigned short int> (position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position,long int &value)
{
    this->internal_getValue<long int>(position,value);
}
inline void GPUDataStore::getValue(std::vector<unsigned int> &position,unsigned long int  &value)
{
    this->internal_getValue<unsigned long int>(position,value);
}


inline void GPUDataStore::setValue(std::vector<unsigned int> &position,std::complex<float> &value)
{
    this->internal_setValue<std::complex<float> >(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position, double &value)
{
    this->internal_setValue<double>(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,float &value)
{
    this->internal_setValue<float>(position,value);
}

inline void GPUDataStore::setValue(std::vector<unsigned int> &position,std::complex<double> &value)
{
    this->internal_setValue<std::complex<double> >(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position, int &value)
{
    this->internal_setValue<int>(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,unsigned int &value)
{
    this->internal_setValue<unsigned int>(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,short int &value)
{
    this->internal_setValue<short int>(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,unsigned short int &value)
{
    this->internal_setValue<unsigned short int> (position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,long int &value)
{
    this->internal_setValue<long int>(position,value);
}
inline void GPUDataStore::setValue(std::vector<unsigned int> &position,unsigned long int  &value)
{
    this->internal_setValue<unsigned long int>(position,value);
}
inline size_t GPUDataStore::getSize()
{
    return this->sizeOfArray;
}


inline float GPUDataStore::getFloatValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<float>(noOfIndexes,vars);
}
inline double GPUDataStore::getDoubleValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<double>(noOfIndexes,vars);
}

inline int GPUDataStore::getIntValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<int>(noOfIndexes,vars);
}

inline unsigned GPUDataStore::getUIntValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<unsigned int>(noOfIndexes,vars);
}

inline short GPUDataStore::getShortIntValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<short>(noOfIndexes,vars);
}

inline unsigned short GPUDataStore::getUShorInttValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<unsigned short>(noOfIndexes,vars);
}

inline long GPUDataStore::getLongValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<long>(noOfIndexes,vars);
}

inline unsigned long GPUDataStore::getULongValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<unsigned long>(noOfIndexes,vars);
}

inline std::complex<double> GPUDataStore::getComplexDoubleValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<std::complex<double> >(noOfIndexes,vars);
}

inline std::complex<float> GPUDataStore::getComplexFloatValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<std::complex<float> >(noOfIndexes,vars);
}

inline GAFW::SimpleComplex<double> GPUDataStore::getSimpleComplexDoubleValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<GAFW::SimpleComplex<double> >(noOfIndexes,vars);
}

inline GAFW::SimpleComplex<float> GPUDataStore::getSimpleComplexFloatValue(unsigned int noOfIndexes,va_list &vars)
{
    return this->internal_getValue<GAFW::SimpleComplex<float> >(noOfIndexes,vars);
}


inline void  GPUDataStore::getValues(float * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<float>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}

inline void GPUDataStore::getValues(double * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<double>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(int * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<int>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(unsigned int * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<unsigned int>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(short * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<short>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(unsigned short * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<unsigned short>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(long * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<long>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}

inline void GPUDataStore::getValues(unsigned long * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<unsigned long>(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}

inline void GPUDataStore::getValues(std::complex<double> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<std::complex<double> >(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::getValues(std::complex<float> *value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<std::complex<float> >(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}

inline void GPUDataStore::getValues(GAFW::SimpleComplex<float> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<GAFW::SimpleComplex<float> >(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}

inline void GPUDataStore::getValues(GAFW::SimpleComplex<double> * value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_getValues<GAFW::SimpleComplex<double> >(value, no_of_elements, isColumnWise,noOfIndexes,vars);
}
inline void GPUDataStore::setValue(float &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<float>(value,noOfIndexes,vars);
}
inline void GPUDataStore::setValue(double &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<double>(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(std::complex<float> &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<std::complex<float> >(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(std::complex<double> &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<std::complex<double> >(value,noOfIndexes,vars);
    
}
inline void GPUDataStore::setValue(int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<int>(value,noOfIndexes,vars);
}
inline void GPUDataStore::setValue(unsigned int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<unsigned int>(value,noOfIndexes,vars);
}
inline void GPUDataStore::setValue(short int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<short>(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(unsigned short int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<unsigned short>(value,noOfIndexes,vars);
}
inline void GPUDataStore::setValue(long int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<long>(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(unsigned long int &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<unsigned long>(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(GAFW::SimpleComplex<float> &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<GAFW::SimpleComplex<float> >(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValue(GAFW::SimpleComplex<double> &value,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValue<GAFW::SimpleComplex<double> >(value,noOfIndexes,vars);
}

inline void GPUDataStore::setValues(float *value,unsigned no_of_elements, bool isColumnWiseStored, unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<float>(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(double *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<double>(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}


inline void GPUDataStore::setValues(std::complex<float> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<std::complex<float> >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(std::complex<double> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<std::complex<double> >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<int> (value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(unsigned int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<unsigned int >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(short int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<short int >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(unsigned short int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<unsigned short >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}
inline void GPUDataStore::setValues(long int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<long >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(unsigned long int *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<unsigned long >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(GAFW::SimpleComplex<float> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<GAFW::SimpleComplex<float> >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}

inline void GPUDataStore::setValues(GAFW::SimpleComplex<double> *value,unsigned no_of_elements, bool isColumnWiseStored,unsigned int noOfIndexes,va_list &vars)
{
    this->internal_setValues<GAFW::SimpleComplex<double> >(value,no_of_elements,isColumnWiseStored, noOfIndexes,vars);
}
    

} }
#endif	/* GPUMATRIXDATASTORE_H */

