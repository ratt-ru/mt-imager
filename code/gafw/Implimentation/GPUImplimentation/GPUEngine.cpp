/* GPUEngine.cpp:  Implementation of the GPUEngine class. 
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
namespace GAFW { namespace GeneralImplimentation {}};
using namespace GAFW::GeneralImplimentation;

#include "GPUafw.h"
using namespace GAFW::GPU;


using namespace std;

GPUEngine::GPUEngine(GPUFactory * factory,GAFW::FactoryStatisticsOutput * statisticOutput):Engine(factory,"GPUEngine","GPUEngine")
{
    this->nextDeviceSubmitTo=0;
    this->statisticsOutput=statisticOutput;
    this->snapshot_no=0;
    this->factory=factory;
    this->calculationRequestQueue=new SynchronousQueue<std::vector<OperationDescriptor *> *>();
    this->operationReadyQueue=new SynchronousQueue<OperationDescriptor * >;
    this->broadcastRequestQueue=new SynchronousQueue<void * >;
    this->dataDescriptorQueue=new SynchronousQueue<DataDescriptor *>;
    //Have to do the algorithm to detect how much devices
    
    //we assume only 1 cuda device
    this->logDebug(other,"Initialising devices");
    checkCudaError(cudaGetDeviceCount(&this->noOfDevices),"Unable to get Device count");
    stringstream ss;
    ss<< "No Of CUDA GPUs is" <<this->noOfDevices;
    this->logDebug(other,ss.str());
    this->devices=new CUDADeviceManager*[this->noOfDevices];
    for (int i=0;i<this->noOfDevices;i++)
    {
        this->devices[i]=new CUDADeviceManager(i,this); //*this->operationReadyQueue,*this->broadcastRequestQueue);
    }
    this->logDebug(other,"Initialising Engine Threads");
    this->threadCalcScheduling=new boost::thread(ThreadEntry<GPUEngine>(this,&GPUEngine::calculationSchedulingThread));
    this->threadOperationReady=new boost::thread(ThreadEntry<GPUEngine>(this,&GPUEngine::operatorReadyThread));
    this->threadHandleDataDescriptors=new boost::thread(ThreadEntry<GPUEngine>(this,&GPUEngine::handleDataDescriptorsThread));
    
    
}
GPUEngine::~GPUEngine()
{
    for (int i=0;i<this->noOfDevices;i++)
        delete this->devices[i];
    vector<OperationDescriptor *> *null=NULL;
    OperationDescriptor *shutdown=new OperationDescriptor();
    shutdown->specialActions=OperationDescriptor::ThreadShutdown;
    DataDescriptor *shutdownd=new DataDescriptor(0);
    shutdownd->specialActions=DataDescriptor::ThreadShutdown;
    this->calculationRequestQueue->push(null);
    this->dataDescriptorQueue->push(shutdownd);
    this->operationReadyQueue->push(shutdown);
    this->threadCalcScheduling->join();
    this->threadHandleDataDescriptors->join();
    this->threadOperationReady->join();
    
    
}

size_t GPUEngine::calculateArraySize(ArrayDimensions &dim, GAFW::GeneralImplimentation::StoreType &type)
{

    size_t element_size;
    
    switch (type)
    {
        case real_float:
            element_size=sizeof(float);
            break;
        case complex_float:
            element_size=sizeof(cuFloatComplex);
            break;
        case real_double:
            element_size=sizeof(double);
            break;
        case complex_double:
            element_size=sizeof(cuDoubleComplex);
            break;
        case real_int:
            element_size=sizeof(int);
            break;
        case real_uint:
            element_size=sizeof(unsigned int);
            break;
        case real_shortint:
            element_size=sizeof(short int);
            break;
        case real_ushortint:
            element_size=sizeof(unsigned short int);
            break;
        case real_longint:
            element_size=sizeof(long int);
            break;
        case real_ulongint:
            element_size=sizeof(unsigned long int);
            break;
         default:
             element_size=0;
             break;
    }
    return dim.getTotalNoOfElements()*element_size;
}

CalculationId GPUEngine::calculate(GAFW::GeneralImplimentation::Result *result)
{
    
    this->logInfo(other,result->getParent(),"Calculation Request..Validating");
    this->snapshot_no++;
    this->arrayDataDescriptorMap.clear();
    this->toPublishReusablesMap.clear();
    this->toDeleteAsReusable.clear();
    this->calcVector=new std::vector<OperationDescriptor *>;
    GAFW::GeneralImplimentation::Result *r=dynamic_cast<GAFW::GeneralImplimentation::Result *>(result);
    GAFW::GeneralImplimentation::Array *firstArray=dynamic_cast<GAFW::GeneralImplimentation::Array*>(this->findProperResult(r)->getParent());
    validate_Array(firstArray); //All validation is done here
    //Now post validation
    //We need to update the activeReusablesMap
    //First deletion
    for (vector<GAFW::GeneralImplimentation::Array*>::iterator i=this->toDeleteAsReusable.begin();i<this->toDeleteAsReusable.end();i++)
    {
        DataDescriptor * desc=this->activeReusablesMap[*i];
        if (desc->reusable!=true) throw GeneralException("BUG");
        desc->DataMutex.lock();
        desc->reusable=false;  
        desc->DataMutex.unlock();
        this->activeReusablesMap.erase(*i);
    }
    for (map<GAFW::GeneralImplimentation::Array*,DataDescriptor *>::iterator i=this->toPublishReusablesMap.begin();i!=this->toPublishReusablesMap.end();i++)
    {
        this->activeReusablesMap[i->first]=i->second;
    }
    
    
    
    //I might need to insert something in result here... but need to think a bit
        
    
    this->logInfo(other,r->getParent(),"Validation Ready.. Submitting for calculation");
    //And now ... submit to queue
    this->calculationRequestQueue->push(this->calcVector);
    
    return this->snapshot_no;
    
}
DataDescriptor * GPUEngine::validate_Array(GAFW::GeneralImplimentation::Array * array)
{
    //First think check for loops and if we already validated
    if (this->arrayDataDescriptorMap.count(array)!=0)
    {
        if (this->arrayDataDescriptorMap[array]->validationReady)
        {
            //great we just return as validation has been completed already
            return this->arrayDataDescriptorMap[array];
        }
        else
        {
            //This implies there is a loop... that will go to infinite.. so throw a validation exception
            ValidationException2("Short circuit detected",array);
        }
    }
    
    
    DataDescriptor * myDesc=new DataDescriptor(noOfDevices);
    //we immediately register this DataDescriptor
    this->arrayDataDescriptorMap[array]=myDesc;  // Note in case this is binded or input the descriptor will change
    
    ArrayInternalData arrayData;
    this->getArrayInternalData(array,arrayData);
    //PreValidate as per requirment
    for (vector<GAFW::GeneralImplimentation::Array *>::iterator i=arrayData.preValidatorDependents.begin();i<arrayData.preValidatorDependents.end();i++)
    {
        if (*i==NULL) throw ValidationException("A NULL was found in the preValidatorDependents vector");
        this->validate_Array(*i);
    }  //PreVaildation has to move to Operator not array
    
    //Ok let's check if input or binded
     
     if (arrayData.result_Outputof!=NULL)
     {
         //Ok this is a binded Array...give control over to the below function
         //it might change the data descriptor object so we return what we find in map
         this->validate_bindedArray(array);
         return this->arrayDataDescriptorMap[array];
     }
     if (arrayData.output_of==NULL)
     {
         //This is an input Array
         this->validate_inputArray(array);
         return this->arrayDataDescriptorMap[array];
     }
     //Ok so this an Output Array 
     //It will be validated by validate_OutputArray() called through ValidateOperator
    validate_Operator((GPUArrayOperator*) arrayData.output_of);
    //Let's just check that I have been validated
    //Non validation might happen if the operator removes this array from its output
    
    myDesc=this->arrayDataDescriptorMap[array];
    if (!myDesc->validationReady)
    {
        throw ValidationException2("Operator output was not validated!!!",array);
    }
    else return myDesc;
    
}
DataDescriptor * GPUEngine::validate_outputArray(GAFW::GeneralImplimentation::Array *array)
{
    DataDescriptor * myDesc;
    //Description might already be stored..but should not have been validated
    if (this->arrayDataDescriptorMap.count(array)!=0)
    {
        myDesc=this->arrayDataDescriptorMap[array];
    }
    else
    {
        myDesc=new DataDescriptor(this->noOfDevices);
        this->arrayDataDescriptorMap[array]=myDesc;  //register
    }
    if (myDesc->validationReady)
        throw GeneralException2("Output array has been found already validated, which is not expected",array);
    //we immediately register this DataDescriptor
     
    
    ArrayInternalData arrayData;
    this->getArrayInternalData(array,arrayData);
    if (!arrayData.dim.isWellDefined())
        throw ValidationException2("Array dimensions not well defined",array);
    myDesc->dim=arrayData.dim;
    if ((arrayData.type<0)||arrayData.type>=StoreTypeUnknown)
        throw ValidationException2("Array Store Type is not set",array);
    myDesc->type=arrayData.type;
    //Let's calculate size
    myDesc->size=this->calculateArraySize(myDesc->dim,myDesc->type);
    myDesc->snapshot_no=this->snapshot_no;
    if (myDesc->size==0) throw ValidationException2("Array size is 0!!!",array);
    //Do we require a copy of the result?
   // myDesc->resultDataStore=arrayData.result;
    if (arrayData.requireResult)
    {
        //The results have to be saved
        this->createResultStore(arrayData.result,this->snapshot_no,arrayData.type,arrayData.dim);
        GPUDataStore * myStore=(GPUDataStore *)this->getResultStore(arrayData.result,this->snapshot_no);
        myDesc->copyTo=myStore->describeMySelf().pointer;
        myStore->setDataNotValid();
        myDesc->resultDataStore=myStore;
        myDesc->copyToReady=false;
    }
    else
    {
        myDesc->copyToReady=true; //that means that there is no need to copy
    }
    if (arrayData.toOverwrite)
    { 
        //We are supposed to find a reusable result 
        if (this->activeReusablesMap.count(array)==0)
            throw ValidationException("Result was not reusable to use it now for overwrite");
        DataDescriptor *otherDesc=this->activeReusablesMap[array];
        myDesc->overwrite=otherDesc;
        otherDesc->DataMutex.lock();
        otherDesc->forOverwrite=true;
        otherDesc->DataMutex.unlock();
    }
    if (arrayData.toReuse)
    {
        myDesc->reusable=true;
        this->toPublishReusablesMap[array]=myDesc;
    }
    if (this->activeReusablesMap.count(array)!=0)
    {
        this->toDeleteAsReusable.push_back(array);
    }
    myDesc->array=array;
    //validaion ready...
    myDesc->validationReady=true;
    return myDesc;
}
void GPUEngine::validate_Operator(GPUArrayOperator *arrayOperator)
{
    OperationDescriptor * myDesc=new OperationDescriptor();
    
    
    
    
    
    //Operator input validation
    vector<GAFW::GeneralImplimentation::Array *> vec=arrayOperator->_getInputs();;
    int noOfInputs=vec.size();
    myDesc->noOfInputs=noOfInputs;
    myDesc->submissionData.noOfInputs=noOfInputs;
    if (myDesc->noOfInputs!=0)
    {
        myDesc->inputs=new DataDescriptor*[noOfInputs];
        myDesc->submissionData.inputs=new GPUArrayDescriptor[noOfInputs]; 
        
    }
    for (int i=0;i<noOfInputs;i++)
    {
            if (vec[i]==NULL) throw ValidationException("A NULL was found in the inputs vector");;
            myDesc->inputs[i]=this->validate_Array(vec[i]);
            myDesc->submissionData.inputs[i].dim=myDesc->inputs[i]->dim;  //input might be active but these two variables never change
            myDesc->submissionData.inputs[i].type=myDesc->inputs[i]->type;
            
            //Some inputs (in case of reusability) might be active .. so to update I have to acquire lock
            myDesc->inputs[i]->DataMutex.lock();
            myDesc->inputs[i]->relatedOperationsCounter++;
            myDesc->inputs[i]->engineKnownRelatedOperations++;
            myDesc->inputs[i]->DataMutex.unlock();
            
    }
    
    //Time to validate operator and then output...
    //This validation is expected change the output... but never the input
    ValidationData valData;
    arrayOperator->validate(valData);
    //We can set data on buffer
    myDesc->buffer.mode=valData.bufferMode;
    myDesc->buffer.size=valData.bufferSize;
    
    
    myDesc->snapshot_id=this->snapshot_no;
    myDesc->arrayOperator=arrayOperator;
    myDesc->submissionData.params=this->getOperatorParamsObject(arrayOperator);
    //Let's now validate the outputs
    vec=arrayOperator->_getOutputs();
    int noOfOutputs=vec.size();
    myDesc->noOfOutputs=noOfOutputs;
    myDesc->submissionData.noOfOutputs=noOfOutputs;
    
    if (myDesc->noOfOutputs!=0)
    {
        myDesc->outputs=new DataDescriptor*[noOfOutputs];
        myDesc->submissionData.outputs=new GPUArrayDescriptor[noOfOutputs]; 
    }
    for (int i=0;i<noOfOutputs;i++)
    {
            if (vec[i]==NULL) throw ValidationException("A NULL was found in the inputs vector");;
            myDesc->outputs[i]=this->validate_outputArray(vec[i]);
            myDesc->submissionData.outputs[i].dim=myDesc->outputs[i]->dim; //If active these two should not make any changes 
            myDesc->submissionData.outputs[i].type=myDesc->outputs[i]->type;
            //Different form inputs.. outputs cannot be active 
            myDesc->outputs[i]->relatedOperationsCounter++;
            myDesc->outputs[i]->engineKnownRelatedOperations++;
            
            
    }
    //Register calculation
    
    this->calcVector->push_back(myDesc);
    

}
void GPUEngine::validate_inputArray(GAFW::GeneralImplimentation::Array *array)
{
    //Assumption is that it is already registered
    //The easiest procedure
    ArrayInternalData arrayData;
    DataDescriptor *myDesc;
    myDesc=this->arrayDataDescriptorMap[array];
    
    this->getArrayInternalData(array,arrayData);
  
    //Run pre-validation... What's this?---TODO
    if (arrayData.preValidator!=NULL) 
        arrayData.preValidator->preValidate(array); 
                
    this->getArrayInternalData(array,arrayData);
   //UUM WHAT if two 
    
    if (arrayData.store==NULL)
    {
            throw ValidationException2("The array is expected to be input but store was not found.",array);
    }
    DataStoreSnapshotDescriptor snap_desc=arrayData.store->createSnapshot(this->snapshot_no,0); //TO CHANGE
    myDesc->cache=snap_desc.pointer;
    myDesc->cacheDataStore=(GPUDataStore*)arrayData.store;
    myDesc->dim=snap_desc.dim;
    myDesc->type=snap_desc.type;
    myDesc->size=snap_desc.size;
    myDesc->snapshot_no=this->snapshot_no;
    myDesc->array=array;
    myDesc->validationReady=true;
    
    
}

void GPUEngine::validate_bindedArray(GAFW::GeneralImplimentation::Array *array)
{
   ArrayInternalData arrayData;
   DataDescriptor *myDesc;
   this->getArrayInternalData(array,arrayData);
   GAFW::GeneralImplimentation::Result *result=dynamic_cast<GAFW::GeneralImplimentation::Result *>(this->findProperResult(arrayData.result_Outputof)); //We might have ProxyResult
   GAFW::GeneralImplimentation::Array* binded_to=dynamic_cast<GAFW::GeneralImplimentation::Array *>(result->getParent());
   //WE can have two situtaion.. either an Input Array or a reusable result
   ArrayInternalData bindedArrayData;
   this->getArrayInternalData(binded_to,bindedArrayData);
   if (bindedArrayData.store!=NULL)
   {
       myDesc=this->validate_Array(binded_to);
   }
   else
   {
       //This is reusable.. check in active reusable map
       if (this->activeReusablesMap.count(binded_to)==0)
           throw ValidationException2(string("Array is bounded to an array with nickname ")+binded_to->objectName+string (" that was not reusable"),array);
       myDesc=this->activeReusablesMap[binded_to];
       if(bindedArrayData.removeReusability)
           this->toDeleteAsReusable.push_back(binded_to);
       //I shudl also remve flag
       //TODO
   }
   delete this->arrayDataDescriptorMap[array];
   //Final thing...define array
   array->setDimensions(myDesc->dim);
   DataTypeManual d(myDesc->type);
   array->setType(d);
   
   this->arrayDataDescriptorMap[array]=myDesc;
 }  

void GPUEngine::calculationSchedulingThread()
{
    this->logDebug(other,"Scheduling Thread initialised");
    //Let's keep it simple
    for(;;)
    {
        //nextDeviceSubmitTo=1;
        vector<OperationDescriptor *> * snapshot; 
        this->calculationRequestQueue->pop_wait(snapshot);
        if (snapshot==NULL) 
        {
            this->logDebug(other,"Shutting down Scheduling thread");
            break;
        }
        for (vector<OperationDescriptor *>::iterator i=snapshot->begin();i<snapshot->end();i++)
        {
            this->logInfo(other,string("Submitting ")+(*i)->arrayOperator->objectName);
            this->devices[this->nextDeviceSubmitTo]->submitOperation(*i);
        }
        
        this->logDebug(other,"All operations submitted");
        nextDeviceSubmitTo++;
        nextDeviceSubmitTo%=this->noOfDevices;
        //nextDeviceSubmitTo=0;
        
    }
    
}
void GPUEngine::operatorReadyThread()
{
    this->logDebug(other,"Operation Ready Thread initialised");
    for (;;)
    {
        OperationDescriptor *desc;
        this->operationReadyQueue->pop_wait(desc);
        this->logInfo(other,desc->arrayOperator,"Popped in Operation Ready Engine Thread");
        if (desc->specialActions==OperationDescriptor::ThreadShutdown)
        {
            this->logDebug(other,"Shutting down Operations Ready Thread");
            break;
        }
        
        GAFW::GPU::GPUEngineOperatorStatistic *stat=new GPUEngineOperatorStatistic;
        stat->kernelExcecutionTime=desc->kernelExecutionDuration;
        stat->operatorName=desc->arrayOperator->name;
        stat->operatorNickname=desc->arrayOperator->objectName;
        stat->snapshotNo=desc->snapshot_id;
        
        this->statisticsOutput->push_statistic(stat);
     
        //All data Descriptors are to be handled by the handleDataDescriptor thread
        for (int i=0;i<desc->noOfInputs;i++)
        {
            boost::mutex::scoped_lock lock(desc->inputs[i]->DataMutex);
            desc->inputs[i]->engineKnownRelatedOperations--;
            //desc->inputs[i]->sentToEngineDataQueueCounter++;
            //this->dataDescriptorQueue->push(desc->inputs[i]);
            this->submitDataDescriptorReview(desc->inputs[i]);
        }
        for (int i=0;i<desc->noOfOutputs;i++)
        {
            boost::mutex::scoped_lock lock(desc->outputs[i]->DataMutex);
            //desc->outputs[i]->sentToEngineDataQueueCounter++;
            desc->outputs[i]->engineKnownRelatedOperations--;
            //this->dataDescriptorQueue->push(desc->outputs[i]);
            this->submitDataDescriptorReview(desc->outputs[i]);
            if (desc->outputs[i]->overwrite!=NULL)
            {
                DataDescriptor * overwriteDesc=desc->outputs[i]->overwrite;
                boost::mutex::scoped_lock lock(overwriteDesc->DataMutex);
                //It is not any more for overwrite
                if (!overwriteDesc->forOverwrite)
                    throw GeneralException("BUG:Logic not as expected");
                overwriteDesc->forOverwrite=false;
                this->submitDataDescriptorReview(overwriteDesc);
                //overwriteDesc->sentToEngineDataQueueCounter++;
                //this->dataDescriptorQueue->push(overwriteDesc);
            }
            
        }
        delete desc;
        
        //handle overwrite here
    }
    
    return;
}
void GPUEngine::handleDataDescriptorsThread()
{
    this->logDebug(other,"Engine DataDescriptor thread initialised");
    for (;;)
    {
        DataDescriptor *data;
        this->dataDescriptorQueue->pop_wait(data);
        if (data->specialActions==DataDescriptor::ThreadShutdown)
        {
            this->logDebug(other,"Engine DataDescriptor Thread Shutting down");
            break;
        }
       data->DataMutex.lock();
       data->sentToEngineDataQueueCounter--;
       this->logWarn(other,data->array,"Received");
       
           
        //handle any overwrite
        // Important ...an overwrite can be overwritten by only the next calculation
        
        
        
        //In here we have to analyse data.. to see if to delete or not or maybe send it to a CUDA device for immediate caching
        //First decide of this packet is useless
        
        
        bool useless=true;
        if (data->sentToEngineDataQueueCounter!=0) useless=false;
        else if (data->reusable||data->forOverwrite) useless=false;
        else if (data->engineKnownRelatedOperations!=0) useless=false;
        
        if ((useless)&&(data->relatedOperationsCounter!=0))
            useless=false;
        
        if (useless)
        {       //Ok we now decided it is useless but is the data in some list or queue
            for (int i=0;i<this->noOfDevices;i++)
            {
                if ((data->inNonLinkedList[i])||(data->SubmitForGPUReviewCounter[i]!=0))
                { 
                    useless=false; 
                    this->devices[i]->submitDataDescriptorForReview(data); //This will trigger the particular devices as to remove from list
                                               //We will again be informed as  the system re-sends the DataDescriptor Object
                }
            }
        }
        else
        {
            //We might have received this DataDescriptor because of an immediate require for cache 
            //(data is to be moved from one GPU to another)
            if ((data->cache==NULL)&&(data->requireImmediateCache))
            {
                //Ok we need to search for that device that has the memory if currently there is
                //If not then it means either a bug and the system will halt or the calculation is not yet ready
                for (int i=0;i<this->noOfDevices;i++)
                {
                    if (data->GPUPointer[i]!=NULL)
                    {
                        if (!data->underCalculation)
                                this->devices[i]->submitDataDescriptorForReview(data);
                        //Otherwise do not bother since it will be handled when the calculation is ready
                    }
                }
            }
        }
        
        
        data->DataMutex.unlock();
       
        if (useless)
            delete data;
            
            
    }
    return;
}
void GPUEngine::submitOperationReady(OperationDescriptor * desc)
{
    this->operationReadyQueue->push(desc);
}
void GPUEngine::submitDataDescriptorReview(DataDescriptor *desc)
{
    //Important... This function must always be called when 
    //desc->DataMutex is locked by the calling thread
    desc->sentToEngineDataQueueCounter++;
    this->dataDescriptorQueue->push(desc);
}
        
        
