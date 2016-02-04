// Simple gpu kernel to compute the the derivative of
// using a constant dx value
__global__ void derivative_constant_dx(float* f, float dx, float *dfdx)
{  
  int i   = threadIdx.x;


  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (i < 4) {
    s_f[sj][si-4]  = s_f[sj][si+mx-5];
    s_f[sj][si+mx] = s_f[sj][si+1];   
  }

  __syncthreads();

  df[globalIdx] = 
    ( c_ax * ( s_f[sj][si+1] - s_f[sj][si-1] )
    + c_bx * ( s_f[sj][si+2] - s_f[sj][si-2] )
    + c_cx * ( s_f[sj][si+3] - s_f[sj][si-3] )
    + c_dx * ( s_f[sj][si+4] - s_f[sj][si-4] ) );
}

// Simple gpu kernel to compute the derivative
// using a varied x value
__global__ void derivative_variable_dx(float* f, float* x, float *dfdx)
{  
  int i   = threadIdx.x;

}