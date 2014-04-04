/* OpeartionDescriptor.h:  Definition of the OperationDescriptor class. 
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

#ifndef __OPERATIONDESCRIPTOR_H__
#define	__OPERATIONDESCRIPTOR_H__

#include <vector>
namespace GAFW { namespace GPU
{
    class DataDescriptor;
    class OperationDescriptor {
    private:
        OperationDescriptor(const OperationDescriptor& orig):noOfInputs(0),noOfOutputs(0) {};
    public:
        enum {
           NoAction,
           ThreadShutdown,
           JustData
        } specialActions;
        GAFW::GPU::GPUArrayOperator *arrayOperator;
        cudaEvent_t *eventStart;
        cudaEvent_t *eventEnd; //The event that is fired once the submitted operation ends execution
        void * data; //A pointer to some data that will be passed from submitToGPU() to postRunExecute()
        //CppProperties::Properties params; // A snapshot of the parameters set to the operator
        GPUSubmissionData submissionData;
        //Inputs// outputs and buffers
        DataDescriptor** inputs;
        size_t noOfInputs;
        DataDescriptor** outputs;
        size_t noOfOutputs;
        float kernelExecutionDuration;
        int snapshot_id;
        struct Buffer
        { 
            void *GPUPointer;
            size_t size; //0 always implies no buffers required
            /*enum {
                Buffer_Normal, //allocted and transferred to submission 
                Buffer_DeallocBeforeSubmit, //deallocated before submition but allocationm continue after submission
                Buffer_DeallocBeforeSubmit_FreezeAlloc, //deallocated before submition and no new memory allocation until operation finished
                Buffer_UnkownSize, //buffer size is not known and hadled during submission.. defragmantation and etc will happen before
                Buffer_UnkownSize_FreezeAlloc
            }*/ 
            enum GAFW::GPU::Buffer::BufferType mode;
        } buffer;
        OperationDescriptor();
        virtual ~OperationDescriptor();
    private:

    };
}};


#endif	/* OPERATIONDESCRIPTOR_H */

