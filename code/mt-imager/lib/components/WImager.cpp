/* WImager.cpp: Implementation  of the WImager component, GAFW module, and class. 
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
#include "WImager.h"

#include "MTImagerException.h"
#include "ConvolutionFunctionGenerator.h"
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <gafw.h>
using namespace GAFW;
using namespace std;
using namespace mtimager;
//using namespace GAFW::Tools::CppProperties;

void print1Di(GAFW::Result *r)
{
    //return;
    //int y=0;
    std::vector<unsigned int> pos;
    pos.push_back(0);
    //pos.push_back(3);
    //pos.push_back(0);
    int b=r->getLastSnapshotId();
    ValueWrapper<int> i;
    for (int x=0; x<r->getDimensions(b).getNoOfColumns();x++)
        {       
                pos[0]=x;
                r->getValue(b,pos,i);
                if (i.value!=0)
                        cout << x << '\t' << i.value <<endl;
        }
   /*pos[0]=r->getDimensions(b).getNoOfColumns()-1;
     float  i;
                
                r->getValue(b,pos,i);
        
              cout <<endl<< i <<endl;
    */
     
}
void print2Di(GAFW::Result *r)
{
    //return;
    //int y=0;
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    int b=r->getLastSnapshotId();
   /* for (int x=1245; x<1255;x++)
    {    for (int y=1245; y<1255;y++)
        
        {
        complex<float> i;
                
                pos[0]=x;
                pos[1]=y;
                r->getValue(b,pos,i);
        
              cout << i <<'\t';
        }
    cout << endl;
    */
    cout << "HERE" <<endl;
    for (int x=0;x<r->getDimensions(b).getNoOfRows();x++)
    {   
        bool toprint=false;
        ValueWrapper<int> i;
        for (int y=0;y<r->getDimensions(b).getNoOfColumns();y++)
        {
                
                pos[0]=x;
                pos[1]=y;
                r->getValue(b,pos,i);
                if ((i.value!=0)&&(y==0))
                {
                    cout <<endl<<x;
                    toprint=true;
                }
                if (toprint)
                {
                    cout  <<'\t'<< i.value;   
                }
         }
        
    }
    cout <<endl<< "END";
  
     
}
void printi(GAFW::Result *r)
{
    int b=r->getLastSnapshotId();
    switch (r->getDimensions(b).getNoOfDimensions())
    {
        case 1:
            print1Di(r);
            break;
        case 2:
            print2Di(r);
            break;
    }
}
void printImage(GAFW::Result *r)
{
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    pos.push_back(0);
    ValueWrapper<complex<float> > i;
    int b=r->getLastSnapshotId();
    for (unsigned int r2=0; r2<4;r2++)
    {pos[2]=r2; cout <<r2<<endl;
    for (int x=0; x<2501;x++)
    {    for (int y=0; y<2501;y++)
        
        {
                
        pos[0]=x;
                pos[1]=y;
                r->getValue(b,pos,i);
                 if ((i.value.real()+i.value.imag())!=0.0f) 
                     cout << pos[0]<< " " <<pos[1] << " " << i.value << endl;
     }
    
    
    
    
    }
    }
}


WImager::WImager(GAFW::Factory *factory,std::string nickname, ConvolutionFunctionGenerator * generator,Conf conf):Module(factory,nickname,"W-Projection Imager"),/*imagedef(imageMan),*/convGenerator(generator)
{
    logDebug(other,"Initialising...");
    this->conf=conf;
    checkConf();
    

    this->requestMyArray("UVWData");
    this->requestMyArray("Visibilities");
    this->requestMyArray("MyLSRKFrequencies");
    this->requestMyArray("MyWeights");
    this->requestMyArray("MyFlags");
    
    this->requestMyArray("ConvolutionFunctionDataCopy")->bind_to(this->convGenerator->getConvFunction());
    this->requestMyArray("ConvolutionFunctionPositionCopy")->bind_to(this->convGenerator->getConvFunctionPositionData());
    this->requestMyArray("ConvolutionFunctionSumsCopy")->bind_to(this->convGenerator->getConvFunctionSumData());
    
    this->requestMyOperator("PreProcessUVW","PreProcessUVW",
            "ConvolutionFunctionPositionCopy","ConvolutionFunctionSumsCopy",
            "UVWData","MyLSRKFrequencies","MyWeights","MyFlags")
            ->a(7,"GriddingData","GriddingDataConvolutionIndex",
            "ToGridIndicator","ToCompressIndicator","GriddingDataSupport",
            "TakeConjugateIndicator","ConvolutionFunctionSum");
    this->requestMyOperator("GriddedRecords","Sum","ToGridIndicator")->a("Statistic:TotalGriddedRecords");
    this->requestMyOperator("CompressedRecords","Sum","ToCompressIndicator")->a("Statistic:TotalCompressedRecords");
    this->requestMyOperator("GriddedPoints","GriddedPointsCalculator","ToGridIndicator","GriddingDataSupport")->a("Statistic:TotalGriddedPoints");
    this->requestMyOperator("CompressedPoints","GriddedPointsCalculator","ToCompressIndicator","GriddingDataSupport")->a("Statistic:TotalCompressedPoints");
    
    this->getMyResult("Statistic:TotalGriddedPoints")->requireResults();
    this->getMyResult("Statistic:TotalCompressedPoints")->requireResults();
    this->getMyResult("Statistic:TotalGriddedRecords")->requireResults();
    this->getMyResult("Statistic:TotalCompressedRecords")->requireResults();
   // this->getMyResult("GriddingData")->requireResults();
    ParameterWrapper<int> int_parameter;
    ParameterWrapper<float> float_parameter;
    ParameterWrapper<bool> bool_parameter;
    
    this->getMyOperator("PreProcessUVW")->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.sampling",this->convGenerator->getSampling()));
    this->getMyOperator("PreProcessUVW")->setParameter(float_parameter.setNameAndValue("ConvolutionFunction.wsquareincrement",this->convGenerator->getWSquareIncrement()));
    this->getMyOperator("PreProcessUVW")->setParameter(float_parameter.setNameAndValue("uvImage.u_increment",(float)this->conf.u_increment));
    this->getMyOperator("PreProcessUVW")->setParameter(float_parameter.setNameAndValue("uvImage.v_increment",(float)this->conf.v_increment));
    this->getMyOperator("PreProcessUVW")->setParameter(int_parameter.setNameAndValue("uvImage.rows",this->conf.v_pixels));
    this->getMyOperator("PreProcessUVW")->setParameter(int_parameter.setNameAndValue("uvImage.columns",this->conf.u_pixels));
    
    
    //calulation of normalization
    this->requestMyOperator("NormalizerCalculate","Sum","ConvolutionFunctionSum")->a("NormalizerValue")->getResults()->reusable();
    this->getMyOperator("NormalizerCalculate")->setParameter(int_parameter.setNameAndValue("dimension",1));
    
    
    //First Generate sum accumulation of as to generate an index
    this->requestMyOperator("GridIndexAccumulate","AccumulatedSum","ToGridIndicator")->a("GridAccumulated");
    
    this->requestMyOperator("CreateIndexAndReorder", "CreateIndexAndReorder",
          //5 inputs
            "ToGridIndicator","GridAccumulated",
            "GriddingDataSupport",
            "GriddingDataConvolutionIndex")
            ->a(3,"DataIndex","DataSupportReOrdered","GriddingDataConvolutionIndexReordered");
    
    this->requestMyOperator("NotSameSupportDetect","ValueChangeDetect","DataSupportReOrdered")->a("NotSameSupportIndicatorReordered");
    
    this->requestMyOperator("NotSameSupportAccumulate","AccumulatedSum","NotSameSupportIndicatorReordered")->a("NotSameSupportAccumulated1");
    this->requestMyOperator("CreateBlockDataIndex","CreateBlockDataIndex","NotSameSupportIndicatorReordered","NotSameSupportAccumulated1","DataSupportReOrdered")->a("BlockDataIndx_without_support");  //remove support
    this->getMyOperator("CreateBlockDataIndex")->setParameter(bool_parameter.setNameAndValue("with_support",true));
    
    this->requestMyOperator("NotSameSupportManipulate","NotSameSupportManipulate","NotSameSupportIndicatorReordered","BlockDataIndx_without_support")->a("ManipulatedNotSameSupport");
    
   this->requestMyOperator("NotSameSupportAccumulate2","AccumulatedSum","ManipulatedNotSameSupport")->a("NotSameSupportAccumulated2");
    this->requestMyOperator("CreateBlockDataIndex2","CreateBlockDataIndex","ManipulatedNotSameSupport","NotSameSupportAccumulated2","DataSupportReOrdered")->a("BlockDataIndx_with_support");  //remove support
    
    this->getMyOperator("CreateBlockDataIndex2")->setParameter(bool_parameter.setNameAndValue("with_support",true));
    
    //this->getMyResult("BlockDataIndx_with_support")->requireResults();
    
    //Last thing we take care of compression
    this->requestMyOperator("CreateCompressionPlan","CreateCompressionPlan","DataIndex","GridAccumulated","ToCompressIndicator")->a("CompressPlan");
    this->requestMyOperator("CompressVisibilities","CompressVisibilities","DataIndex","CompressPlan","Visibilities","TakeConjugateIndicator","MyWeights","MyFlags")->a("CompressedVisibilities");
    this->requestMyArray("MyConvolutionFunction")->bind_to(generator->getConvFunction());
    this->requestMyOperator("ConvolutionGridder","ConvolutionGridder",11,"BlockDataIndx_with_support","DataIndex","GriddingData","CompressedVisibilities","MyConvolutionFunction","GriddingDataConvolutionIndexReordered","NormalizerValue","Statistic:TotalGriddedPoints","Statistic:TotalCompressedPoints","Statistic:TotalGriddedRecords","Statistic:TotalCompressedRecords");
    this->getMyOperator("ConvolutionGridder")->a("Grid");
    this->getMyArray("Grid")->setDimensions(ArrayDimensions(3,this->conf.v_pixels,this->conf.u_pixels,this->conf.no_of_polarizations));    
    this->getMyArray("Grid")->setType(DataType<complex <float> >());
    this->getMyArray("Grid")->getResults()->reusable();
    this->getMyOperator("ConvolutionGridder")->setParameter(bool_parameter.setNameAndValue("initialize_output",true));
    logDebug(other,"Initialisation complete");
}
WImager::~WImager()
{
    
}
void WImager::checkConf()
{
    logDebug(other,"Checking Configuration");
   
    stringstream s;
    s << "No of polarizations set to "<< this->conf.no_of_polarizations;
    logDebug(other,s.str());
    
    switch (this->conf.no_of_polarizations)
    {
        case 1:
        case 2:
        case 4:
            break;
        default:
            throw ImagerException("No of polarisations requested is not supported, it must be 1,2 or 4");
    }
    //A bit of logging for info  //Have to change the way as it is too cumbersome
    s.str("");
    s<<"U-V plane dimensions (V x U) = "<< this->conf.v_pixels << "x" <<this->conf.u_pixels;
    logDebug(other,s.str());
    if ((this->conf.v_pixels<1) ||(this->conf.u_pixels<1))
        throw ImagerException("v or u pixels do not make sense");
   
    logDebug(other,"Check complete");
}
void WImager::reset()
{
    
}
void WImager::calculate()
{   
    ParameterWrapper<bool> bool_parameter;
   //std::string s="BlockDataIndx_with_support";
   //this->getMyResult(s)->requireResults();
    int snap_id=this->getMyResult("Grid")->calculate();
    //this->getMyResult("Grid")->waitUntilDataValid(snap_id);
    this->getMyOperator("ConvolutionGridder")->setParameter(bool_parameter.setNameAndValue("initialize_output",false));//printi(this->getMyArray(toPrint)->getResults());
    this->getMyResult("Grid")->overwrite();
    this->getMyResult("NormalizerValue")->overwrite();
    
    int records=this->getMyArray("UVWData")->getDimensions().getNoOfColumns();
    
    this->snapshots.push_back(pair<int,int>(snap_id,records)); //For statistics
    
    //this->getMyResult(s)->waitUntilDataValid(snap_id);
    //print2Di(this->getMyResult(s));
    //exit(0);
    
}
void WImager::setInput(int inputNo, GAFW::Result *res)
{
    if (inputNo>4) throw ImagerException("This module expects three inputs");
    if (inputNo==0) this->getMyArray("UVWData")->bind_to(res);
    if (inputNo==1) this->getMyArray("Visibilities")->bind_to(res);
    if (inputNo==2) this->getMyArray("MyLSRKFrequencies")->bind_to(res);
    if (inputNo==3) this->getMyArray("MyWeights")->bind_to(res);
    if (inputNo==4) this->getMyArray("MyFlags")->bind_to(res);
    
}
Result * WImager::getOutput(int outputNo) 
{
    switch (outputNo)
    {
        case 0:
            return this->getMyResult("Grid");
        case 1:
            return this->getMyResult("NormalizerValue");
        default:
            throw ImagerException("Only one output available");
    }
        
}
void WImager::resultRead(GAFW::ProxyResult *,int snapshot_no)
{
   // Nothing to do 
}
/*
 float WImager::getNormalizer()
 {
     float normalizer=0;
     Result *NormalizerResults=this->getMyArray("NormalizerValue")->getResults();
     for (vector<int>::iterator i=this->snapshots.begin();i<this->snapshots.end();i++)
     {
         float value;
         vector<unsigned int> pos;
         pos.push_back(0);
         NormalizerResults->waitUntilDataValid(*i),
         NormalizerResults->getValue(*i,pos,value);
         normalizer+=value;
     }
     return normalizer;
 }
 int WImager::getNoOfPointsGridded()
 {
     int total=0;
     Result *NormalizerResults=this->getMyArray("Statistic:TotalGriddedPoints")->getResults();
     for (vector<int>::iterator i=this->snapshots.begin();i<this->snapshots.end();i++)
     {
         int value;
         vector<unsigned int> pos;
         pos.push_back(0);
         NormalizerResults->waitUntilDataValid(*i),
         NormalizerResults->getValue(*i,pos,value);
         total+=value;
     }
     return total;
 }
 int WImager::getNoOfPointsCompressed()
 {
     int total=0;
     Result *NormalizerResults=this->getMyArray("Statistic:TotalCompressedPoints")->getResults();
     for (vector<int>::iterator i=this->snapshots.begin();i<this->snapshots.end();i++)
     {
         int value;
         vector<unsigned int> pos;
         pos.push_back(0);
         NormalizerResults->waitUntilDataValid(*i),
         NormalizerResults->getValue(*i,pos,value);
         total+=value;
     }
     return total;
 }*/
void WImager::generateStatistics()
{
     Result *compressedRecords=this->getMyResult("Statistic:TotalCompressedRecords");
     Result *griddedRecords=this->getMyResult("Statistic:TotalGriddedRecords");
     Result *compressedGridPoints=this->getMyResult("Statistic:TotalCompressedPoints");
     Result *griddedGridPoints=this->getMyResult("Statistic:TotalGriddedPoints");
     //this->getMyResult("Grid")->waitUntilDataValid();
     vector<unsigned int> zeropos;
     zeropos.push_back(0);
     for (vector<pair<int,int> >::iterator i=this->snapshots.begin();i<this->snapshots.end();i++)
     {
         WImager::WImagerStatistic * stat=new WImager::WImagerStatistic;
         int snapshotNo=i->first;
         stat->snapshotNo=snapshotNo;
         stat->noOfRecords=i->second;
         stat->noOfPolarizations=this->conf.no_of_polarizations;
         ValueWrapper<int> int_value;
         ValueWrapper<long int> long_value;
         compressedRecords->waitUntilDataValid(snapshotNo);
         compressedRecords->getValue(snapshotNo,zeropos,int_value);
         stat->compressedRecords=int_value.value;
         griddedRecords->waitUntilDataValid(snapshotNo);
         griddedRecords->getValue(snapshotNo,zeropos,int_value);
         stat->griddedRecords=int_value.value;
         compressedGridPoints->waitUntilDataValid(snapshotNo);
         compressedGridPoints->getValue(snapshotNo,zeropos,long_value);
         stat->compressedGridPoints=long_value.value;
         griddedGridPoints->waitUntilDataValid(snapshotNo);
         griddedGridPoints->getValue(snapshotNo,zeropos,long_value);
         stat->griddedGridPoints=long_value.value;
         this->conf.statisticsManager->push_statistic(stat);
     }
     
}
