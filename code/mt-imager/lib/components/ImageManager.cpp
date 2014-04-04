/* ImageManager.cpp: Implementation  of the ImageComponent component and class. 
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
#include "ImageManager.h"
#include "MTImagerException.h"
#include "casa/Containers/Record.h"
#include "fits/FITS/FITSDateUtil.h"
#include "casa/Quanta/MVTime.h"
#include "fits/FITS/FITSKeywordUtil.h"
#include "fits/FITS.h"
#include "ThreadEntry.h"
using namespace casa;
using namespace mtimager;
ImageManager::ImageManager(const Conf conf,std::vector<GAFW::Result*> res) 
{
    this->imageRes=res;
    //We copy configuration and begin setting up our image configuration
    if (!conf.initialized)
    {
        throw mtimager::ImagerException("ImagerManager initialised with an un-initialised configuration");
    }
    this->conf=conf;
    this->readyChannelNo=-1; ///We assume No channel should be ready for now
    //We do not do anything in the main thread everything in a personal Thread
    this->myThread=new boost::thread(mtimager::ThreadEntry<ImageManager>(this,&ImageManager::threadFunc));
}
void ImageManager::threadFunc()
{
   
   this->createFITS();
   this->tempImage=(float*)malloc(this->conf.lpixels*this->conf.mpixels*this->conf.stokes.size()*sizeof(float));
   int saveImageNo=0;
   
   while (saveImageNo<(int)this->imageRes.size())
   {
       {
           boost::mutex::scoped_lock lock(this->myMutex);
           while (saveImageNo>this->readyChannelNo)
                this->myCond.wait(lock);
       }
       //Ok next image ready
       
       this->saveChannel(saveImageNo);
       saveImageNo++;
   }
   int status=0;
   fits_close_file(this->outputFITSImage, &status);
   if (status)
       throw ImagerException("Could not close FITS file!!!")
   
    
}
void ImageManager::createFITS()
{
    //We create the FITS file and then save all channels after they are ready
    //This code is copied from casarest
    MVDirection mvPhaseCenter(this->conf.phaseCenter.getAngle());
   // Normalize correctly
    MVAngle ra=mvPhaseCenter.get()(0);
    ra(0.0);
    MVAngle dec=mvPhaseCenter.get()(1);
    
    Matrix<Double> xform(2,2);
    xform=0.0;xform.diagonal()=1.0;
    
    DirectionCoordinate myRaDec(MDirection::Types(this->conf.phaseCenter.getRefPtr()->getType()),
	    Projection(Projection::SIN),
	    ra.get().getValue(), (double)dec,
	    -this->conf.l_increment,this->conf.m_increment,
	    xform,
	    this->conf.lpixels/2, this->conf.mpixels/2);
    
    //Next axis .. frequency
    vector<double> f;
    for (vector<MFrequency>::iterator i=this->conf.freqs.begin();i<this->conf.freqs.end();i++)
    {
        f.push_back(i->getValue().getValue());
    }
    SpectralCoordinate mySpectral((casa::MFrequency::Types)conf.freqs[0].getRef().getType(),f);
            
   StokesCoordinate myStokes(this->conf.stokes);
   
   ObsInfo myobsinfo;
   myobsinfo.setTelescope(this->conf.telescope);
   myobsinfo.setPointingCenter(this->conf.phaseCenter.getValue());
   myobsinfo.setObsDate(this->conf.obsEpoch);
   myobsinfo.setObserver(this->conf.observer);
  
  //Adding everything to the coordsystem
  this->coord.addCoordinate(myRaDec);
  this->coord.addCoordinate(myStokes);
  this->coord.addCoordinate(mySpectral);
  this->coord.setObsInfo(myobsinfo);
    
  //Output FITS file open  
    int status = 0;

    //FitsOutput *output =new FitsOutput(this->conf.outputFITSFileName.c_str(),FITS::Disk);
    
    fits_create_file(&this->outputFITSImage, this->conf.outputFITSFileName.c_str(), &status);
    if (status != 0) 
    {

        //logError(other, "Unable to create FITS file Reason:");
        // to extract reason
        string error = "Unable to Create FITS file";
        throw ImagerException(error);
    }
    
   
    if (this->coord.nWorldAxes() != this->coord.nPixelAxes()) 
   {
       throw ImagerException("BUG: internal coordinates not set well for FITS");
   }     
   
   
    //Now let's set some headers fro FITS
   casa::Record header;
   //We only support 32 bit floating point
   header.define("bitpix", -32);
   header.setComment("bitpix", "Floating point (32 bit)");
   long int naxes[4];
   casa::Vector<int> naxis(4);
   //We always expect a three dimensional imageRes
   naxis(0)=this->conf.lpixels;
   naxes[0]=this->conf.lpixels;
   naxis(1)=this->conf.mpixels;
   naxes[1]=this->conf.mpixels;
   
   naxis(2)=this->conf.stokes.size(); 
   naxes[2]=this->conf.stokes.size(); 
   naxis(3)=this->conf.freqs.size(); 
   naxes[3]=this->conf.freqs.size(); 
   header.define("NAXIS",naxis);
   
   IPosition shape(4,this->conf.lpixels,this->conf.mpixels,this->conf.stokes.size(),this->conf.freqs.size());
   if (!this->coord.toFITSHeader(header, shape, true, 'c', true, // use WCS 
                                false, false,false))
   {
       throw ImagerException("Unable to create headers from co-ordinates")
   }
   
   header.define("bscale", 1.0);
   header.setComment("bscale", "PHYSICAL = PIXEL*BSCALE + BZERO");
   header.define("bzero", 0.0);
   
   header.define("BUNIT", "JY/BEAM");
   header.setComment("BUNIT", "Brightness (pixel) unit");
   
   
   casa::String date, timesys;
   Time nowtime;
   MVTime now(nowtime);
   FITSDateUtil::toFITS(date, timesys, now);
   header.define("date", date);
   header.setComment("date", "Date FITS file was written");
   header.define("timesys", timesys);
   header.setComment("timesys", "Time system for HDU");
    
   header.define("ORIGIN", "The Malta Imager (mt-imager)");
   
   // Set up the FITS header
   FitsKeywordList kw = FITSKeywordUtil::makeKeywordList();
   if (!FITSKeywordUtil::addKeywords(kw, header))
   {
       throw ImagerException("Error Creating FITS header");
   }
   kw.end();
   kw.first();
   kw.next();
   
   status=0;
   fits_create_img(this->outputFITSImage, FLOAT_IMG, 4, naxes, &status);
   if (status != 0)
       throw ImagerException("There was a problem //TODO");
   
   
   casa::FitsKeyCardTranslator trans(0);
   char *header_char=(char *)malloc(2880*sizeof(char)); //FITS standard
   int cont=1;
   while (cont)
   {
       cont=trans.build(header_char,kw) ;
       status = 0;
        for (int i = 0; (i < 2880/80)&&(header_char[i*80]!=' '); i++) {
             if (fits_write_record(this->outputFITSImage, (header_char) + i * 80, &status) != 0)
                 throw ImagerException("There was a problem");
             
        }
   }
   free(header_char);
   
   //last thing allocate permanent memory for one channel
  
}   

void ImageManager::saveChannel(int channelNo)
{
   imageRes[channelNo]->waitUntilDataValid();
   scoped_detailed_timer t((new DetailedTimerStatistic("Image Saving","N/A",channelNo)),conf.stat);
    
   int snapshotNo=this->imageRes[channelNo]->getLastSnapshotId();
   GAFW::ArrayDimensions dim=imageRes[channelNo]->getDimensions(snapshotNo);
   if (dim.getNoOfDimensions()!=3)
   {
       throw ImagerException("unexpected Image dimensions");
   }
   if (dim.getTotalNoOfElements()!=this->conf.lpixels*this->conf.mpixels*this->conf.stokes.size())
       throw ImagerException("Inconsistent image dimensions");
   int status=0;
   GAFW::PointerWrapper<float> tempImageWrapper(this->tempImage);
    imageRes[channelNo]->getValues(tempImageWrapper,dim.getTotalNoOfElements(),true,0);
    //Now we need swap the m dimension
    int img_index=1+dim.getTotalNoOfElements()*channelNo;
    //for (int pol=0;pol<dim.getZ();pol++)
    //{
        imageRes[channelNo]->logDebug(GAFW::LogFacility::other,"Saving Image");
        //for (int j=dim.getY()-1;j>=0;j--)
        //{
                fits_write_img(this->outputFITSImage, TFLOAT, img_index, /*dim.getNoOfColumns()*/ /*this->tempImage+(pol*dim.getNoOfRows()*dim.getNoOfColumns())+j*dim.getNoOfColumns()*/dim.getTotalNoOfElements(),this->tempImage, &status);
                //img_index+=dim.getNoOfColumns();
        //}
        
    //}
    if (status != 0)
        throw ImagerException("There was a problem //TODO");
   
}

ImageManager::~ImageManager() 
{
    this->myThread->join();
    free(this->tempImage);
}
void ImageManager::nextChannelReady()
{
    boost::mutex::scoped_lock lock(this->myMutex);
    this->readyChannelNo++;
    this->myCond.notify_all();
}
