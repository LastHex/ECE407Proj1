// __global__ void update (float *u, float *u_prev, int N, float dx, float dt, float c)
// {
//         // Each thread will load one element
//         int i = threadIdx.x;
//         int I = threadIdx.x + BLOCKSIZE * blockIdx.x;
//         __shared__ float u_shared[BLOCKSIZE];
//
//         if (I>=N){
// 			return;
// 		}
//
// 		u_shared[i] = u[I];
//         __syncthreads();
//
// 		if (I>0)
//         {
// 			u[I] = u_shared[i] ‐ c*dt/dx*(u_shared[i] ‐ u_shared[i‐1]);
// 		}
// }


// Simple gpu kernel to compute the derivative with a constant dx
__global__ void derivative(float* f, float* f_shifted, float* dfdx, float dx, int n)
{
    unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int s_i = threadIdx.x;
	
	__shared__ float shared_dx;
	__shared__ float shared_f[512];
	__shared__ float shared_f_shifted[512];
	
	// Assigned to shared memory
	s_dx = dx;
	shared_f[s_i] = f[g_i];
	shared_f_shifted[s_i] = f_shifted[g_i];
	
	__syncthreads();
	
    dfdx[i] = (shared_f_shifted[s_i] - shared_f[s_i]) / shared_dx;
}

// Filter out values that are not big enough (use for very large derivatives) i.e. large positive spikes
__global__ void get_only_large_deriv(float* f, int* thresholded_f, float threshold)
{
    __shared__ int tf[512];
	__shared__ float shared_threshold;

	shared_threshold = threshold;

    unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b_i = threadIdx.x;

    tf[b_i] = f[g_i];

    tf[b_i] = (tf[b_i] >= shared_threshold) ? 1 : 0;

    __syncthreads();

    thresholded_f[g_i] = tf[tid];

}

// Filter out values that are not small enough (use for very small derivatives) i.e. large negative spikes
__global__ void get_only_small_deriv(float* f, bool* thresholded_f, float threshold)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    thresholded_f[i] = (f[i] <= threshold) ? 1 : 0;
}
