#include "object.h"
#include <cmath>  
#include <iostream>
#include <fstream>
#include <stdio.h>  
#include <string>
using namespace std;
model::model()
{
    cout<<"This is a sample solver for diffusion problems:"<<endl;
    cout<<"Please specify the name of input file:"<<endl;
}
void model::ReadModelData(char* filename){
    FILE *fp;  
    char inputline[BUFSIZ];
    fp = fopen(filename, "r");
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%lf", &(this->dx));
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%lf", &(this->xdomain));
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%lf", &(this->ydomain));  
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%lf", &(this->diffusion_Coeff));
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%lf", &(this->dt)); 
    fgets(inputline, BUFSIZ, fp);
    sscanf(inputline, "%d %lf", &(this->outFreq), &(this->totalT)); 
    this->nx=int(xdomain/dx);
    this->ny=int(ydomain/dx);
    data=new double*[this->nx];
    for (int i=0;i<nx;++i){
        data[i]=new double[this->ny];
    }
    
}
void model::printModelSetups(){
    cout<<"PARAMETERS:\n";
    cout<<"dx: "<<this->dx<<endl;
    cout<<"domain size in x-dir: "<<this->xdomain<<endl;
    cout<<"domain size in y-dir: "<<this->ydomain<<endl;
    cout<<"Diffusion Coeff.: "<<this->diffusion_Coeff<<endl;
    cout<<"time increment: "<<this->dt<<endl;
    cout<<"Output Frequency: "<<this->outFreq<<endl;
    cout<<"Total Time: "<<this->totalT<<endl;
}
void model::Initialize(){
        for(int i=0;i<this->nx;++i){
            for(int j=0;j<this->ny;++j){
                this->data[i][j]=100;
          }
        }
        double r=2.,cx=5.,cy=5.;
        double r2=pow(r,2);
        for(int i=0;i<this->nx;++i){
            for(int j=0;j<this->ny;++j){
                double p2=pow(i*this->dx-cx,2) + pow(j*this->dx-cy,2);
                if(p2<r2){
                    this->data[i][j]=400;
                } 
          }
        }
}
void model::Explicit_Solver(){
    double** data_tmp;
    data_tmp=new double*[this->nx];
    for (int i=0;i<nx;++i){
        data_tmp[i]=new double[this->ny];
    }
    for(int i=0;i<this->nx;++i){
        for(int j=0;j<this->ny;++j){
            data_tmp[i][j]=this->data[i][j];
        }
    }
    for(int i=1;i<this->nx-1;++i){
        for(int j=1;j<this->ny-1;++j){
            double top=data_tmp[i][j+1];
            double bottom=data_tmp[i][j-1];
            double left=data_tmp[i-1][j];
            double right=data_tmp[i+1][j];
            this->data[i][j]=data_tmp[i][j]+this->diffusion_Coeff*this->dt*(top+bottom+left+right-4*data_tmp[i][j]);
        }
    }     
    delete[] data_tmp;   
}
int model::getSteps(){
    return int(this->totalT/this->dt);
}
int model::getOutFreq(){
    return this->outFreq;
}
void model::dumpVTK(int time){
    char filename[100];
    snprintf(filename,100,"Output_%04d.vtk",time);
    ofstream writevtk(filename);
    writevtk<<"# vtk DataFile Version 3.0"<<endl;
    writevtk<<"my simple VTK writer\n";
    writevtk<<"ASCII\n";
    writevtk<<"DATASET STRUCTURED_POINTS"<<endl;
    char str[100];
    snprintf(str,100,"DIMENSIONS %4d %4d %4d\n",int(this->xdomain/this->dx),int(this->ydomain/this->dx),1);
    writevtk<<str;
    writevtk<<"ASPECT_RATIO 1 1 1\n";
    writevtk<<"ORIGIN 0 0 0\n";
    writevtk<<"POINT_DATA ";
    writevtk<<int(this->xdomain/this->dx)*int(this->ydomain/this->dx)<<endl;
    writevtk<<"SCALARS Temps float 1\n";
    writevtk<<"LOOKUP_TABLE default\n";
    for (int i=0;i<this->nx;++i){
        for(int j=0;j<this->ny;++j){
            writevtk<<this->data[i][j]<<" ";
        }
        writevtk<<"\n";
    }
}
model::~model()
{
}