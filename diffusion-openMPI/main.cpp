#include "object.C"
#include <mpi.h>
int main(int argc, char **argv){
    int num_procs;
    int rank;
    MPI_Init( &argc ,  &argv);
    MPI_Comm_size( MPI_COMM_WORLD , &num_procs);
    MPI_Comm_rank( MPI_COMM_WORLD , &rank);
    model* diffusion=new model;
    diffusion->ReadModelData((char *) "input.txt",rank,num_procs);
    if (rank == 0){
        diffusion->printModelSetups();
    }
    diffusion->Initialize(rank,num_procs);
    diffusion->comm_field(rank,num_procs);
    for (int i=0;i<diffusion->getSteps();++i){
        diffusion->comm_field(rank,num_procs);
        diffusion->Explicit_Solver(rank,num_procs);
        if (i%diffusion->getOutFreq()==0){
            diffusion->dumpVTK(i,rank,num_procs);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
