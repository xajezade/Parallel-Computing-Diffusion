#include "object.h"
#include <cmath>  
#include <iostream>
#include <fstream>
#include <stdio.h>  
#include <string>
#include <omp.h>
#define PAD 8
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
    data=new double**[this->nx];
    for (int i=0;i<nx;++i){
        data[i]=new double*[this->ny];
    }
    for (int i=0;i<nx;++i){
        for (int j=0;j<nx;++j){
            data[i][j]=new double[PAD];
        }
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
                this->data[i][j][0]=100;
          }
        }
        double r=this->xdomain/4,cx=this->xdomain/2,cy=this->ydomain/2;
        double r2=pow(r,2);
        for(int i=0;i<this->nx;++i){
            for(int j=0;j<this->ny;++j){
                double p2=pow(i*this->dx-cx,2) + pow(j*this->dx-cy,2);
                if(p2<r2){
                    this->data[i][j][0]=400;
                } 
          }
        }
}
void model::Explicit_Solver(){
    int i,j;
    double top, bottom,left,right;
    double delt=this->diffusion_Coeff*this->dt;
    #pragma omp parallel private(top,bottom,left,right,i,j) firstprivate(delt) 
    {
        double** data_tmp;
        int numprocs=omp_get_num_threads();
        int id=omp_get_thread_num();
        int step1=id*int(this->nx/numprocs);
        int stepN=(id+1)*int(this->nx/numprocs);
        data_tmp=new double*[int(this->nx/numprocs)+2];
        for (i=0;i<int(this->nx/numprocs)+2;++i){
            data_tmp[i]=new double[this->ny];
        }
        for(i=1;i<int(this->nx/numprocs)+1;++i){
            for(int j=0;j<this->ny;++j){
                data_tmp[i][j]=this->data[i+step1-1][j][0];
            }
        }
        if (id==0){
            for(int j=0;j<this->ny;++j){
                data_tmp[0][j]=this->data[0][j][0];
            }
        } 
        else{
            for(int j=0;j<this->ny;++j){
                data_tmp[0][j]=this->data[step1-1][j][0];
            }            
        }

        if (id==numprocs-1){
            for(int j=0;j<this->ny;++j){
                data_tmp[int(this->nx/numprocs)+1][j]=this->data[this->nx-1][j][0];
            }
        } 
        else{
            for(int j=0;j<this->ny;++j){
                data_tmp[int(this->nx/numprocs)+1][j]=this->data[int(this->nx/numprocs)+step1][j][0];
            }            
        }

        int stmp=step1-1;
        if(id==0) step1=1;
        if(id==numprocs-1) stepN=this->nx-1;
        for(i=step1;i<stepN;++i){
            for(j=1;j<this->ny-1;++j){
                top=data_tmp[i-stmp][j+1];//data_tmp[i][j+1];
                bottom=data_tmp[i-stmp][j-1];//data_tmp[i][j-1];
                left=data_tmp[i-stmp-1][j];//data_tmp[i-1][j];
                right=data_tmp[i-stmp+1][j];//data_tmp[i+1][j];
                //this->data[i][j]=data_tmp[i][j]+delt*(top+bottom+left+right-4*data_tmp[i][j]);
                this->data[i][j][0]=data_tmp[i-stmp][j]+delt*(top+bottom+left+right-4*data_tmp[i-stmp][j]);
            }
        } 
        for (i=0;i<int(this->nx/numprocs)+2;++i){
            delete[] data_tmp[i];
        }
        delete[] data_tmp;
    }
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
            writevtk<<this->data[i][j][0]<<" ";
        }
        writevtk<<"\n";
    }
}
model::~model()
{
}