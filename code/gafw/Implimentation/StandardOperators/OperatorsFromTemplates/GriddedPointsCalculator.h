/* GriddedPointsCalculator.h:  GeneralReduction template specialisation 
 * for GriddedPointsCalculator operator  
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

#ifndef __GRIDDEDPOINTSCALCULATOR_H__
#define	__GRIDDEDPOINTSCALCULATOR_H__

#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"

namespace GAFW { namespace GPU { namespace StandardOperators 
{
    class InnerGridCalc 
    {
    public:
        __device__ __inline__ static void  InnerOperation(unsigned long int &ans,struct GAFW::GPU::OperatorTemplates::InputElements<int,2> &el)
        {
            //Input 0 must be the indicator, input 1 the support
            ans= (unsigned long int)(el.inputs[0]*el.inputs[1]*el.inputs[1]);
           
        }
    };
    class GriddedPointsCalculatorDefenition
    {
    public:
        const static int NoOfInputs=2;
        static const char *OperatorName;
        typedef class InnerGridCalc InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationAdd OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationDoNothing FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];
    };
    const char * GriddedPointsCalculatorDefenition::OperatorName="GriddedPointsCalculator";
    const GAFW::GPU::OperatorTemplates::GeneralReductionOptions GriddedPointsCalculatorDefenition::options[]={ GENERAL_REDUCTION_OPTION(GriddedPointsCalculatorDefenition,real_int,real_ulongint,1),GENERAL_REDUCTION_OPTION_END };

    typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<GriddedPointsCalculatorDefenition> GriddedPointsCalculator; 
 
}}}

#endif	/* GRIDDEDPOINTSCALCULATOR_H */

