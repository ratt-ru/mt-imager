/* SumMultiply.h:  GeneralReduction template specialisation for SumMultiply operator  
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
#ifndef __SUMMULTIPLY_H__
#define	__SUMMULTIPLY_H__
#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"
namespace GAFW { namespace GPU { namespace StandardOperators
{
    class SumMultiplyReduceFullDefenition
    {
    public:
        const static int NoOfInputs=2;
        static const char *OperatorName;
        typedef class GAFW::GPU::OperatorTemplates::InnerMultiply InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationAdd OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationDoNothing FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];
     
    };
    const char * SumMultiplyReduceFullDefenition::OperatorName="SumMultiply";
    const GAFW::GPU::OperatorTemplates::GeneralReductionOptions SumMultiplyReduceFullDefenition::options[]={ GENERAL_REDUCTION_OPTION_SET_1(SumMultiplyReduceFullDefenition),GENERAL_REDUCTION_OPTION_END };

    typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<SumMultiplyReduceFullDefenition> SumMultiply; 
}}};

#endif	/* SUMMULTIPLY_H */

