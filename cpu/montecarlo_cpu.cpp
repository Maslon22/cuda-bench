#include <random>
double montecarlo_cpu(int N){
    std::mt19937 g(0);
    std::uniform_real_distribution<double>d(0,1);
    int in=0;
    for(int i=0;i<N;i++){
        double x=d(g),y=d(g);
        if(x*x+y*y<=1) in++;
    }
    return 4.0*in/N;
}