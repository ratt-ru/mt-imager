/* GeneralReduction.h:  Definition of the GeneralReduction template class 
 * and other related stuff 
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
#ifndef __GENERALREDUCTION_H__
#define	__GENERALREDUCTION_H__
#include "GPUafw.h"
#include "SharedStructures.h"
namespace GAFW { namespace GPU { namespace OperatorTemplates
{
    
    class GeneralReductionOptions
    {
    public:
        GeneralReductionOptions(const GAFW::GeneralImplimentation::StoreType  inputType,const GAFW::GeneralImplimentation::StoreType  outputType,
                const int inputMultiply,  void  (*submitFunc)(GAFW::GPU::GPUSubmissionData&)) //,
            :inputType(inputType),outputType(outputType),inputMultiply(inputMultiply),submitFunc(submitFunc)
            {}
        const GAFW::GeneralImplimentation::StoreType  inputType;
        const GAFW::GeneralImplimentation::StoreType  outputType;
        const int inputMultiply;  //1 or 2 ..2 if complex are treated as two floats
        void (*submitFunc)(GAFW::GPU::GPUSubmissionData&);
    };

        #define TYPE_complex_float_2 float
        #define TYPE_complex_double_2 double
        #define TYPE_complex_float_1 cuComplex
        #define TYPE_complex_double_1 cuDoubleComplex
        #define TYPE_real_float_1 float
        #define TYPE_real_double_1 double
        #define TYPE_real_int_1 int
        #define TYPE_real_uint_1 unsigned int
        #define TYPE_real_shortint_1 short
        #define TYPE_real_ushortint_1 unsigned 
        #define TYPE_real_longint_1 long
        #define TYPE_real_ulongint_1 unsigned long
        #define TYPE_complex_float GAFW::SimpleComplex<float>
        #define TYPE_complex_double GAFW::SimpleComplex<double>
        #define TYPE_real_float float
        #define TYPE_real_double double
        #define TYPE_real_int int
        #define TYPE_real_uint unsigned int
        #define TYPE_real_shortint short
        #define TYPE_real_ushortint unsigned 
        #define TYPE_real_longint long
        #define TYPE_real_ulongint unsigned long

      
        #define GENERAL_REDUCTION_OPTION(classname,input,output,treatment)  \
                GAFW::GPU::OperatorTemplates::GeneralReductionOptions( \
                GAFW::GeneralImplimentation::input,GAFW::GeneralImplimentation::output, treatment, \
                GAFW::GPU::OperatorTemplates::GeneralReduction_submitToGPU< classname, TYPE_ ## input ## _ ## treatment, TYPE_ ## output ## _1 , treatment > ) //,  \
    
    
        #define GENERAL_REDUCTION_OPTION_END GAFW::GPU::OperatorTemplates::GeneralReductionOptions(GAFW::GeneralImplimentation::StoreTypeUnknown,GAFW::GeneralImplimentation::StoreTypeUnknown,1,NULL/*,NULL/*,NULL,NULL*/)

    
        #define GENERAL_REDUCTION_OPTION_SET_1(classname)   \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,complex_float,real_float,2),\
                GENERAL_REDUCTION_OPTION(classname,complex_float,real_double,2),\
                GENERAL_REDUCTION_OPTION(classname,complex_double,real_double,2),\
                GENERAL_REDUCTION_OPTION(classname,complex_double,real_float,2)
                
        #define GENERAL_REDUCTION_OPTION_SET_ALL_NOCOMPLEX(classname) \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_float,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_double,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_int,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_uint,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_shortint,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ushortint,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_longint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_longint,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_ulongint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_float,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_double,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_int,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_uint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_shortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_ushortint,1), \
                GENERAL_REDUCTION_OPTION(classname,real_ulongint,real_longint,1) 

        template<class FullDefenition>  
        class GeneralReduction: public GAFW::GPU::GPUArrayOperator 
        {
        private:
            //GeneralReduction_Complex2Real(const GeneralReduction_Complex2Real& orig):GPUArrayOperator(NULL),Factory {};  
            size_t getSizeOfElement(GAFW::GeneralImplimentation::StoreType  type);
        public:
            GeneralReduction(GPUFactory * factory,std::string nickname);
           ~GeneralReduction();

            virtual void submitToGPU(GAFW::GPU::GPUSubmissionData &data);
            virtual void validate(GAFW::GPU::ValidationData &data);
        };
        //submitToGpuWill call the below function
        template<class FullDefenition,class InputType, class OutputType,int InputMultiplier>
            void GeneralReduction_submitToGPU(GAFW::GPU::GPUSubmissionData &data);

    
        }}}
#endif	/* GENERALREDUCTION_H */

