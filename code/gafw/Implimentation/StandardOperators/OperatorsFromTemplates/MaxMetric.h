/* MaxMetric.h:  GeneralReduction template specialisation for MaxMetric operator  
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

#ifndef __MAXMETRIC_H__
#define	__MAXMETRIC_H__
#include "../Templates/HandyOperators.h"
#include "../Templates/GeneralReduction.tcu"
//#include "Templates/GeneralReductionTestGroup.tcu"

namespace GPUAFW { namespace StandardOperators {

class InnerMaxMetric
    {//designed for double and floats and only 2 inputs
    public:
        template<class InputType,class OutputType>
        __device__ __inline__ static void  InnerOperation(OutputType &ans,struct GAFW::GPU::OperatorTemplates::InputElements<InputType,2> &el)
        {
            OutputType max;
            OutputType input0=(OutputType) el.inputs[0];
            OutputType input1=(OutputType) el.inputs[1];
            
            if (el.inputs[0]<el.inputs[1]) max=input1;
            else max=input0;
            OutputType chi=(input0-input1)*(input0-input1); 
            ans=max*max*chi;
                    }
      /*  static GAFW::GPU::OperatorTemplates::GeneralTestData **getTestData()
        {
            std::vector<GAFW::GPU::OperatorTemplates::GeneralTestData *> data;
            
            data.push_back(new GAFW::GPU::OperatorTemplates::InnerOperationTestData<float,float,2>(1.0f,0.0f,5.0f,5.0f));
            data.push_back(new GAFW::GPU::OperatorTemplates::InnerOperationTestData<double,double,2>(1.0,1e-6,1.0,0.0));
            
            //Write more simple tests here
            
            
            data.push_back(NULL);
            //Now we see how much and create teh space
            GAFW::GPU::OperatorTemplates::GeneralTestData ** ret =new GAFW::GPU::OperatorTemplates::GeneralTestData*[data.size()];
            for (int i=0;i<data.size();i++)
            {
                ret[i]=data[i];
            }
            return ret;   
        }
       * */
};
    class MaxMetricFullDefenition
    {
    public:
        const static int NoOfInputs=2;
        static const char *OperatorName;
        typedef class InnerMaxMetric InnerOperationDefenition ;
        typedef class GAFW::GPU::OperatorTemplates::OuterOperationMax OuterOperationDefenition;
        typedef class GAFW::GPU::OperatorTemplates::FinalOperationDoNothing FinalOperationDefenition;
        static const GAFW::GPU::OperatorTemplates::GeneralReductionOptions options[];

    };
    const char * MaxMetricFullDefenition::OperatorName="MaxMetric";
    
    const GAFW::GPU::OperatorTemplates::GeneralReductionOptions MaxMetricFullDefenition::options[]={ GENERAL_REDUCTION_OPTION_SET_1(MaxMetricFullDefenition),GENERAL_REDUCTION_OPTION_END };

    typedef class GAFW::GPU::OperatorTemplates::GeneralReduction<MaxMetricFullDefenition> MaxMetric; 
//    typedef class GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup<MaxMetricFullDefenition> MaxMetricTestGroup;
//    template <> class GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup<MaxMetricFullDefenition>;
   
   
    
    
    
    

}}
#endif	/* MAXMETRIC_H */

