/* GPUSubmissionData.h:  Definition of the GPUSubmissionData class. 
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

#ifndef __GPUSUBMISSIONDATA_H__
#define	__GPUSUBMISSIONDATA_H__
#include "Properties.h"
namespace GAFW { namespace GPU
{
    

class GPUSubmissionData {
public:
    GPUSubmissionData();
    virtual ~GPUSubmissionData();
    int noOfInputs;
    GPUArrayDescriptor *inputs;
    int noOfOutputs;
    GPUArrayDescriptor *outputs;
    cudaStream_t stream;
    void ** postExecutePointer;
    cudaEvent_t *startEvent;
    cudaEvent_t *endEvent;
    bool endEventRecorded;
    GAFW::Tools::CppProperties::Properties params;
    void *bufferGPUPointer;
    int buffersize;
    GPUDeviceDescriptor *deviceDescriptor;
    
private:

};
} }
#endif	/* GPUSUBMISSIONDATA_H */

