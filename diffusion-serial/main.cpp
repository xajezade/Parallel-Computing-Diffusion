#include "object.C"

int main(){
    model* diffusion=new model;
    char* filename;
    filename= new char[100];
    cin.getline(filename,100);
    diffusion->ReadModelData(filename);
    diffusion->printModelSetups();
    diffusion->Initialize();
    for (int i=0;i<diffusion->getSteps();++i){
        diffusion->Explicit_Solver();
        if (i%diffusion->getOutFreq()==0){
            diffusion->dumpVTK(i);
        }
    }
    return 0;
}