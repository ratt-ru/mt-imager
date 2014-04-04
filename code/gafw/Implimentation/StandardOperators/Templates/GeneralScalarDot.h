/* 
 * File:   GeneralScalarDot.h
 * Author: daniel
 *
 * Created on 20 February 2013, 22:16
 */

#ifndef GENERALSCALARDOT_H
#define	GENERALSCALARDOT_H

#include "../GPUafw.h"
#include "SharedStructures.h"
#include "Tester/gafwtester.h"
namespace GPUAFW { namespace OperatorTemplates
{
    typedef struct {} NoScalar;
    class GeneralScalarDotOptions
    {
    public:
        GeneralScalarDotOptions(const GAFW::GeneralImplimentation::StoreType  scalarInputType,const GAFW::GeneralImplimentation::StoreType  inputType,const GAFW::GeneralImplimentation::StoreType  outputType,
                void  (*submitFunc)(GAFW::GPU::GPUSubmissionData&) , GAFW::Tester::PrepareFunctionPointer testPreparefunc,GAFW::Tester::TestFunctionPointer testfunc
        )
            :scalarInputType(scalarInputType),inputType(inputType),outputType(outputType),submitFunc(submitFunc),
            innerOperatorTestFunction(testfunc),innerOperatorTestPrepareFunction(testPreparefunc)
            {}
        const GAFW::GeneralImplimentation::StoreType  scalarInputType;
        const GAFW::GeneralImplimentation::StoreType  inputType;
        const GAFW::GeneralImplimentation::StoreType  outputType;
        void (*submitFunc)(GAFW::GPU::GPUSubmissionData&);
        const GAFW::Tester::TestFunctionPointer innerOperatorTestFunction;
        const GAFW::Tester::PrepareFunctionPointer innerOperatorTestPrepareFunction;
        
        
    };
        #define TYPE_StoreTypeUnknown NoScalar;
        #define TYPE_complex_float cuComplex
        #define TYPE_complex_double cuDoubleComplex
        #define TYPE_real_float float
        #define TYPE_real_double double
        #define TYPE_real_int int
        #define TYPE_real_uint unsigned int
        #define TYPE_real_shortint short
        #define TYPE_real_ushortint unsigned 
        #define TYPE_real_longint long
        #define TYPE_real_ulongint unsigned long
      
        #define GENERAL_SCALARDOT_OPTION(classname,inputScalar,input,output)  \
                GAFW::GPU::OperatorTemplates::GeneralScalarDotOptions( \
                GAFW::input,GAFW::output, \
                GAFW::GPU::OperatorTemplates::GeneralScalarDotOptions_submitToGPU< classname, TYPE_ ## inputScalar,TYPE_ ## input , TYPE_ ## output  >, \
                NULL,NULL) 
              //  GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup< classname >::prepareInnerOperatorTests< TYPE_ ## input ## _ ## treatment , TYPE_ ## output ## _1 >, \
              //  GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup< classname >::testInnerOperatorTests< TYPE_ ## input ## _ ## treatment, TYPE_ ## output ## _1 > )
   
    
    /*NULL,
    
    NULL )
    
    /*\
                GAFW::GPU::OperatorTemplates::GeneralReduction_submitToGPU< classname, TYPE_ ## input ## _ ## treatment, TYPE_ ## output ## _1 , treatment > )
    /*, \
                GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup< classname >::prepareInnerOperatorTests< TYPE_ ## input ## _ ## treatment , TYPE_ ## output ## _1 > , \
       NULL )
     /*         
     * GAFW::GPU::OperatorTemplates::GeneralReductionTestGroup< classname >::testInnerOperatorTests< TYPE_ ## input ## _ ## treatment, TYPE_ ## output ## _1 > \
                )
      */  
        #define GENERAL_SCALARDOT_OPTION_END GAFW::GPU::OperatorTemplates::GeneralScalarDotOptions(GAFW::GeneralImplimentation::StoreType Unknown,GAFW::GeneralImplimentation::StoreType Unknown,GAFW::GeneralImplimentation::StoreType Unknown,NULL,NULL,NULL)

    
        #define GENERAL_SCALARDOT_SET_1(classname)   \
                GENERAL_SCALARDOT_OPTION(classname,real_float,real_float,real_float), \
                GENERAL_SCALARDOT_OPTION(classname,real_double,real_double,real_double), \
                GENERAL_SCALARDOT_OPTION(classname,real_float,complex_float,real_float),\
                GENERAL_SCALARDOT_OPTION(classname,real_double,complex_double,real_double)

    //kpmli NULL billi tibni l-function
    template<class FullDefenition>  
    class GeneralScalarDot: public GPUArrayOperator 
    {
    private:
        //GeneralReduction_Complex2Real(const GeneralReduction_Complex2Real& orig):GPUArrayOperator(NULL),Factory {};  
    public:
        GeneralScalarDot(GAFW::GPU::GPUFactory * factory,std::string nickname);
       ~GeneralScalarDot();
        virtual void submitToGPU(GAFW::GPU::GPUSubmissionData &data);
        virtual void validate();
    };
    //submitToGpuWill call the below function
    template<class FullDefenition,class InputType, class OutputType>
        void GeneralScalarDot_submitToGPU(GAFW::GPU::GPUSubmissionData &data);

    
}}


#endif	/* GENERALSCALARDOT_H */

