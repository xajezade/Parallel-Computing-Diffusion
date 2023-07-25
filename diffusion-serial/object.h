class model
{
private:
    double dx;
    double xdomain;
    double ydomain;
    double diffusion_Coeff;
    double dt;
    int outFreq;
    double totalT;
    int nx,ny;
    double** data;

public:
    model();
    ~model();
    void ReadModelData(char* filename);
    void printModelSetups();
    void Initialize();
    int getSteps();
    int getOutFreq();
    void Explicit_Solver();
    void dumpVTK(int time);
};