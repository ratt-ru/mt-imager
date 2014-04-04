/* GPUArrayOperator.cpp:  Definition of the GPUArrayOperator class which implements the GAFW Array.
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

#ifndef __GPUARRAYOPERATOR_H__
#define	__GPUARRAYOPERATOR_H__

#include "ValidationData.h"

namespace GAFW { namespace GPU {

class GPUArrayOperator :public  GAFW::GeneralImplimentation::ArrayOperator {
protected:
   
public:
    GPUArrayOperator(GPUFactory *f,std::string nickname, std::string name="GPUMatrixOperator");
    
    virtual ~GPUArrayOperator();
    virtual void validate();  ///we must obsolete
    virtual void validate(GAFW::GPU::ValidationData &data);  ///we must obsolete
    virtual void submitToGPU(GPUSubmissionData &data)=0; 
    virtual void postRunExecute(void *data);
    std::vector<GAFW::GeneralImplimentation::Array *> getInputsArrays();
    
    
    

};
} }


#endif	/* GPUMATRIXOPERATOR_H */

