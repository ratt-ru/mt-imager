/* ZeroToCorner2DShift.h:  Definition of the ZeroToCorner2DShift operator class 
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

#ifndef __ZEROTOCORNER2DSHIFT_H__
#define	__ZEROTOCORNER2DSHIFT_H__
//Operator shifts 0 component to centre of spectrum
namespace GAFW { namespace GPU 
{
    namespace StandardOperators
    {
        class ZeroToCorner2DShift: public GAFW::GPU::GPUArrayOperator {
        private:
//            FFT2DShift(const FFT2DShift& orig):GPUArrayOperator(NULL) {};  
            unsigned int calculateArraySize(GAFW::ArrayDimensions& d, GAFW::GeneralImplimentation::StoreType  &type);
        public:
            ZeroToCorner2DShift(GAFW::GPU::GPUFactory * factory,std::string nickname);
            ~ZeroToCorner2DShift();
            virtual void submitToGPU(GAFW::GPU::GPUSubmissionData &data);
            virtual void validate();
        };
    }
}}



#endif	/* ZEROTOCORNER2DSHIFT_H */

