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
    double* buffer_left;
    double* buffer_right;
    double* buffer_left_new;
    double* buffer_right_new;

public:
    model();
    ~model();
    void ReadModelData(char* filename,int rank,int num_procs);
    void printModelSetups();
    void Initialize(int rank,int num_procs);
    int getSteps();
    int getOutFreq();
    void Explicit_Solver(int rank,int num_procs);
    void comm_field(int rank,int num_procs);
    void dumpVTK(int time,int rank, int num_procs);
};