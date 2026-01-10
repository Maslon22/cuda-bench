#include "../common/precision.hpp"
#include <random>

real montecarlo_cpu(int N){
    std::mt19937 g(0);
    std::uniform_real_distribution<real> d(0,1);
    int in = 0;
    for(int i=0;i<N;i++){
        real x = d(g), y = d(g);
        if(x*x + y*y <= 1) in++;
    }
    return real(4) * in / N;
}
