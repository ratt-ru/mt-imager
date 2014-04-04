#include "HelperFunctions.h"

//#include "Magick++.h"
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
#if 0
/*using namespace Magick;
void saveImage(GAFW::Result *src,int id,string name,int channel)
{
    /* channel 0 - Red
       channel 1 - Green
       channel 2 - Blue
     * channel 3 -- All same value 
     */
    int cols=src->getDimensions(id).getNoOfColumns();
    int rows=src->getDimensions(id).getNoOfRows();
    Magick::Image image(Geometry(cols,rows),"black");
    image.modifyImage();
    Pixels cache(image);
    PixelPacket * pixels=cache.get(0,0,cols,rows);
    float values[rows][cols];
    src->getSimpleArray(id,(float*)values);
    int p=0;
    //Findmax
    float max=0;
    float min=0;
    float toadd;
    float sum=0;
    for (int y=0;y<rows;y++)
        for (int x=0; x<cols;x++)
        {
            sum+=values[y][x];
            if (values[y][x]>max) max=values[y][x];
            if (values[y][x]<min) min=values[y][x];
            
        }
    if (min<0) {
        max+=-min;
        toadd=-min;
    }
    else toadd=0;
        
    cout << "max=" << max << " toadd=" << toadd <<" sum=" << sum << endl;
    for (int y=0;y<rows;y++)
        for (int x=0; x<cols;x++)
        {
            switch (channel)
            {
                
            
                case 0:
                        (pixels+p)->red=values[y][x];
                        (pixels+p)->green=0;
                        (pixels+p)->blue=0;
                        break;
                case 1:

                        (pixels+p)->red=0;
                        (pixels+p)->green=values[y][x];
                        (pixels+p)->blue=0;
                        break;
                case 2:
                        (pixels+p)->red=0;
                        (pixels+p)->green=0;
                        (pixels+p)->blue=values[y][x];
                        break;
                 case 3:
                        (pixels+p)->red=(values[y][x]+toadd)*((float)MaxRGB/(max));
                        (pixels+p)->green=(values[y][x]+toadd)*((float)MaxRGB/(max));
                        (pixels+p)->blue=(values[y][x]+toadd)*((float)MaxRGB/(max));
                        break;
                        
                 default:
                            
                    //To throw exception
                     ;
            }                    
          p++;
        }
    cache.sync();
    image.write(name);
    
}*/
void print3D(GAFW::Result *r,int b)
{
    //return;
    //int y=0;
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    pos.push_back(0);
    
    for (int z=0;z<r->getDimensions(b).getZ();z++)
    {
    for (int y=0;y<r->getDimensions(b).getNoOfRows();y++) 
    {    for (int x=0; x<r->getDimensions(b).getNoOfColumns();x++)
        {    complex<float> c;
                pos[0]=z;
                pos[1]=y;
                pos[2]=x;
                r->getValue(b,pos,c);
        
              cout << c <<'\t';
        }
        cout << "END " << y << '\n';
    
    }          
    
    } 
}
void print2D(GAFW::Result *r,int b)
{
    //return;
    //int y=0;
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    pos.push_back(0);
    
    for (int y=0;y<r->getDimensions(b).getNoOfRows();y++) 
    {    for (int x=0; x<r->getDimensions(b).getNoOfColumns();x++)
        {    complex<float> c;
                pos[0]=y;
                pos[1]=x;
                r->getValue(b,pos,c);
        
              cout << c <<'\t';
        }
        cout << "END " << y << '\n';
    
    }          

     
}
#endif 