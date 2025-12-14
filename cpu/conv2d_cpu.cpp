void conv2d_cpu(const float*i,float*o,const float*k,int W,int H){
    for(int y=1;y<H-1;y++)
        for(int x=1;x<W-1;x++){
            float s=0;
            for(int ky=-1;ky<=1;ky++)
                for(int kx=-1;kx<=1;kx++)
                    s+=i[(y+ky)*W+x+kx]*k[(ky+1)*3+(kx+1)];
            o[y*W+x]=s;
        }
}