/* CompressVisibilities.h:  Definition of the CompressVisibilities operator class 
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
#ifndef __COMPRESSVISIBILTIES_H__
#define	__COMPRESSVISIBILTIES_H__


#include "GPUafw.h"

namespace mtimager {
    class CompressVisibilities  : public GAFW::GPU::GPUArrayOperator 
    {
        
    public:
        CompressVisibilities(GAFW::GPU::GPUFactory * factory,std::string nickname);
        ~CompressVisibilities();
        virtual void submitToGPU(GAFW::GPU::GPUSubmissionData &data);
        virtual void validate();

        template <class PolVisType,class PolWeightType,class PolFlagsType>
          void kernel_launch_wrapper(dim3 &blocks,dim3 &threadsPerBlock,GAFW::GPU::GPUSubmissionData &data,int &records);

    };
}
#endif	/* COMPRESSVISIBILTIES_H */

