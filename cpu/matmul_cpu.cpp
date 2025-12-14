#include <algorithm>
void matmul_cpu(const float*A,const float*B,float*C,int N){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j];
            C[i*N+j]=s;
        }
}

void matmul_cpu_blocked(const float*A,const float*B,float*C,int N){
    const int BS=64;
    for(int ii=0;ii<N;ii+=BS)
        for(int jj=0;jj<N;jj+=BS)
            for(int kk=0;kk<N;kk+=BS)
                for(int i=ii;i<std::min(ii+BS,N);i++)
                    for(int j=jj;j<std::min(jj+BS,N);j++){
                        float s=C[i*N+j];
                        for(int k=kk;k<std::min(kk+BS,N);k++)
                            s+=A[i*N+k]*B[k*N+j];
                        C[i*N+j]=s;
                    }
}