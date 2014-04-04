/* SumDifference.h:  GeneralReduction template specialisation for SumDifference operator  
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
#ifndef __SUMDIFFERENCE_H__
#define	__SUMDIFFERENCE_H__


#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"
namespace GAFW { namespace GPU { namespace StandardOperators
{
    class SumDifferenceReduceFullDefenition
    {
    public:
        const static int NoOfInputs=2;
        static const char *OperatorName;
        typedef class GAFW::GPU::OperatorTemplates::InnerOperatorTakeDifference InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationAdd OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationDoNothing FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];
    };
    const char * SumDifferenceReduceFullDefenition::OperatorName="SumDifference";
    const GAFW::GPU::OperatorTemplates::GeneralReductionOptions SumDifferenceReduceFullDefenition::options[]=
    { GENERAL_REDUCTION_OPTION_SET_1(SumDifferenceReduceFullDefenition),
      GENERAL_REDUCTION_OPTION_END };

    typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<SumDifferenceReduceFullDefenition> SumDifference; 
}}}

#endif	

