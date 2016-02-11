__global__ void inout(short int * out, short int * in, short Res)
{
  const int i = threadIdx.x;
  out[i] = in[i] * Res/1000000;
}
