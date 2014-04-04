/* RootSumSquare.h:  GeneralReduction template specialisation for RootSumSquare operator  
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

#ifndef ROOTSUMSQUARE2_H
#define	ROOTSUMSQUARE2_H


#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"

namespace GAFW { namespace GPU { namespace StandardOperators
{

    class RootSumSquareReduceFullDefenition
    {
    public:
        const static char * OperatorName;
        const static int NoOfInputs=1;
        typedef class GAFW::GPU::OperatorTemplates::SumSquares<1> InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationAdd OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationSquareRoot FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];

    };

const  GAFW::GPU::OperatorTemplates::GeneralReductionOptions RootSumSquareReduceFullDefenition::options[]={ GENERAL_REDUCTION_OPTION_SET_1(RootSumSquareReduceFullDefenition),GENERAL_REDUCTION_OPTION_END};
const char * RootSumSquareReduceFullDefenition::OperatorName="RootSumSquare";
typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<RootSumSquareReduceFullDefenition> RootSumSquare; 

}}}

#endif	/* ROOTSUMSQUARE2_H */

