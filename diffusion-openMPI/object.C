#include "object.h"
#include <cmath>  
#include <iostream>
#include <fstream>
#include <stdio.h>  
#include <string>
#include <mpi.h>
using namespace std;
model::model()
{
    cout<<"This is a sample solver for diffusion problems!"<<endl;
}
void model::ReadModelData(char* filename,int rank,int num_procs){
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
    this->nx=int(xdomain/dx/num_procs)+2;
    this->ny=int(ydomain/dx);
    data=new double*[this->nx];
    for (int i=0;i<nx;++i){
        data[i]=new double[this->ny];
    }
    buffer_left= new double[this->ny];
    buffer_right= new double[this->ny];
    buffer_left_new= new double[this->ny];
    buffer_right_new= new double[this->ny];
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
void model::Initialize(int rank,int num_procs){
        for(int i=0;i<this->nx;++i){
            for(int j=0;j<this->ny;++j){
                this->data[i][j]=100;
          }
        }
        double r=this->xdomain/4.,cx=this->xdomain/2,cy=this->ydomain/2;
        double r2=pow(r,2);
        for(int i=1;i<this->nx-1;++i){
            for(int j=0;j<this->ny;++j){
                double p2=pow((i+rank*this->nx)*this->dx-cx,2) + pow(j*this->dx-cy,2);
                if(p2<r2){
                    this->data[i][j]=400;
                } 
          }
        }
        //if (rank>0 && rank<num_procs-1){
        for(int j=0;j<this->ny;++j){
                data[0][j]=this->data[1][j];
                data[this->nx-1][j]=this->data[this->nx-2][j];
        }
        //}
        /*if (rank==0){
        for(int j=0;j<this->ny;++j){
                 data[0][j]=this->data[1][j];
        }
        }
        if (rank==num_procs-1){
        for(int j=0;j<this->ny;++j){
                data[this->nx-1][j]=this->data[this->nx-2][j];
        }
        }*/

        for(int j=0;j<this->ny;++j){
            buffer_left[j]=this->data[0][j];
            buffer_left_new[j]=this->data[0][j];
            buffer_right[j]=this->data[this->nx-1][j];
            buffer_right_new[j]=this->data[this->nx-1][j];
        }



}
void model::comm_field(int rank,int num_procs){
    int rank_next = (rank + 1) % num_procs;
    int rank_prev = rank == 0 ? num_procs-1 : rank-1;
    MPI_Status status;
    if (rank %2 ==0){
        MPI_Send( (void *)(this->buffer_right), this->ny, MPI_DOUBLE, rank_next, 1 , MPI_COMM_WORLD);
        MPI_Recv( (void *)(this->buffer_left_new), this->ny ,MPI_DOUBLE ,rank_prev , 1 , MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv( (void *)(this->buffer_left_new), this->ny ,MPI_DOUBLE ,rank_prev , 1 , MPI_COMM_WORLD, &status);
        MPI_Send( (void *)(this->buffer_right), this->ny, MPI_DOUBLE, rank_next, 1 , MPI_COMM_WORLD);
    }
    if (rank %2 ==0){
        MPI_Send( (void *)(this->buffer_left), this->ny, MPI_DOUBLE, rank_prev, 1 , MPI_COMM_WORLD);
        MPI_Recv( (void *)(this->buffer_right_new), this->ny ,MPI_DOUBLE ,rank_next , 1 , MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv( (void *)(this->buffer_right_new), this->ny ,MPI_DOUBLE ,rank_next , 1 , MPI_COMM_WORLD, &status);
        MPI_Send( (void *)(this->buffer_left), this->ny, MPI_DOUBLE, rank_prev, 1 , MPI_COMM_WORLD);
    }
    for(int j=0;j<this->ny;++j){
        this->data[0][j]=buffer_left_new[j];
        this->data[this->nx-1][j]=buffer_right_new[j];
    }
    /*for(int j=0;j<this->ny;++j){
        buffer_left[j]=buffer_left_new[j];
        buffer_right[j]=buffer_right_new[j];
    }*/
}
void model::Explicit_Solver(int rank,int num_procs){
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

    double top, bottom,left,right;
    for(int i=1;i<this->nx-1;++i){
        for(int j=1;j<this->ny-1;++j){
            top=data_tmp[i][j+1];
            bottom=data_tmp[i][j-1];
            left=data_tmp[i-1][j];
            right=data_tmp[i+1][j];
            this->data[i][j]=data_tmp[i][j]+this->diffusion_Coeff*this->dt*(top+bottom+left+right-4*data_tmp[i][j]);
        }
    }
    for(int j=0;j<this->ny;++j){
            this->data[0][j]=this->data[1][j];
            this->data[this->nx-1][j]=this->data[this->nx-2][j];
    }
    for(int j=0;j<this->ny;++j){
        buffer_left[j]=this->data[0][j];
        buffer_right[j]=this->data[this->nx-1][j];
    }   
    for (int i=0;i<nx;++i){
        delete[] data_tmp[i];
    }    
    delete[] data_tmp;  
}
int model::getSteps(){
    return int(this->totalT/this->dt);
}
int model::getOutFreq(){
    return this->outFreq;
}
void model::dumpVTK(int time,int rank, int num_procs){
    char filename[100];
    snprintf(filename,100,"Output_CPU_%d_%04d.vtk",rank,time);
    ofstream writevtk(filename);
    writevtk<<"# vtk DataFile Version 3.0"<<endl;
    writevtk<<"my simple VTK writer\n";
    writevtk<<"ASCII\n";
    writevtk<<"DATASET STRUCTURED_POINTS"<<endl;
    char str[100];
    snprintf(str,100,"DIMENSIONS %4d %4d %4d\n",int(this->ydomain/this->dx),int(this->xdomain/this->dx/num_procs)+2,1);
    writevtk<<str;
    writevtk<<"ASPECT_RATIO 1 1 1\n";
    writevtk<<"ORIGIN 0 "<<rank*(this->nx-2)<<" 0\n";
    writevtk<<"POINT_DATA ";
    writevtk<<(int(this->nx))*int(this->ydomain/this->dx)<<endl;
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
