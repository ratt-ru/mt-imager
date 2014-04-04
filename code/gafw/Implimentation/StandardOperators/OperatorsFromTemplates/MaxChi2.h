/* MaxChi2.h:  GeneralReduction template specialisation for MaxChi2 operator  
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

#ifndef __MAXCHI2_H__
#define	__MAXCHI2_H__


#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"

namespace GAFW { namespace GPU { namespace StandardOperators
{
    class MaxChi2ReduceFullDefenition
    {
    public:
        const static int NoOfInputs=2;
        static const char *OperatorName;
        typedef class GAFW::GPU::OperatorTemplates::InnerChi2 InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationMax OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationDoNothing FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];
    };
    const char * MaxChi2ReduceFullDefenition::OperatorName="MaxChi2";
    const GAFW::GPU::OperatorTemplates::GeneralReductionOptions MaxChi2ReduceFullDefenition::options[]={ GENERAL_REDUCTION_OPTION_SET_1(MaxChi2ReduceFullDefenition),GENERAL_REDUCTION_OPTION_END };
    typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<MaxChi2ReduceFullDefenition> MaxChi2; 
}}}
 




#endif	/* MAXCHI2_H */

