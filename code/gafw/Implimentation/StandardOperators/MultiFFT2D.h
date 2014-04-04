/* MultiFFT2D.h:  Definition of the MultiFFT2D operator class 
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

#ifndef __MULTIFFT2D_H__
#define	__MULTIFFT2D_H__

namespace GAFW { namespace GPU 
{
    namespace StandardOperators
    {
        class MultiFFT2D: public GAFW::GPU::GPUArrayOperator  
        {
        private:
   
        public:
            MultiFFT2D(GAFW::GPU::GPUFactory * factory, std::string nickname );
            ~MultiFFT2D();
            virtual void submitToGPU(GAFW::GPU::GPUSubmissionData &data);
            virtual void validate();
            virtual void validate(GAFW::GPU::ValidationData &valData);

            virtual void postRunExecute(void *data);
        };
    }
}}
#endif	/* MULTIFFT2D_H */

