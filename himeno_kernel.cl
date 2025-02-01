/* -*- mode:c -*- */
#define MR(m,n,r,c,d) m[(n) * mimax * mkmax * mkmax + (r) * mjmax * mkmax + (c) * mkmax + (d)]

__kernel void mat_set_init(
    __global float *Mat) {
  const int mimax = get_global_size(0);
  const int mjmax = get_global_size(1);
  const int mkmax = get_global_size(2);
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  MR(Mat,0,i,j,k)= (float)(i*i)
    /(float)((mimax - 1)*(mimax - 1));
}

__kernel void mat_set(
    __global float *Mat,
    int n,
    float value) {
  const int mimax = get_global_size(0);
  const int mjmax = get_global_size(1);
  const int mkmax = get_global_size(2);
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  MR(Mat,n,i,j,k) = value;
}

__kernel void jacobi1(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global const float* p,
    __global const float* bnd,
    __global const float* wrk1,
    __global float* wrk2,
    __global float* gosa,
    float omega) {
  const int mimax = get_global_size(0);
  const int mjmax = get_global_size(1);
  const int mkmax = get_global_size(2);
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  if (i < 1 || i >= mimax - 1 ||
      j < 1 || j >= mjmax - 1 ||
      k < 1 || k >= mkmax - 1) return;
  const float s0 =
      MR(a,0,i,j,k)*MR(p,0,i+1, j,   k)
    + MR(a,1,i,j,k)*MR(p,0,i,   j+1, k)
    + MR(a,2,i,j,k)*MR(p,0,i,   j,   k+1)
    + MR(b,0,i,j,k)
    *( MR(p,0,i+1,j+1,k) - MR(p,0,i+1,j-1,k)
     - MR(p,0,i-1,j+1,k) + MR(p,0,i-1,j-1,k) )
    + MR(b,1,i,j,k)
    *( MR(p,0,i,j+1,k+1) - MR(p,0,i,j-1,k+1)
     - MR(p,0,i,j+1,k-1) + MR(p,0,i,j-1,k-1) )
    + MR(b,2,i,j,k)
    *( MR(p,0,i+1,j,k+1) - MR(p,0,i-1,j,k+1)
     - MR(p,0,i+1,j,k-1) + MR(p,0,i-1,j,k-1) )
    + MR(c,0,i,j,k) * MR(p,0,i-1,j,  k)
    + MR(c,1,i,j,k) * MR(p,0,i,  j-1,k)
    + MR(c,2,i,j,k) * MR(p,0,i,  j,  k-1)
    + MR(wrk1,0,i,j,k);
  const float ss = (s0*MR(a,3,i,j,k) - MR(p,0,i,j,k))*MR(bnd,0,i,j,k);
  MR(gosa,0,i,j,k) = ss*ss;
  MR(wrk2,0,i,j,k) = MR(p,0,i,j,k) + omega*ss;
}

__kernel void sum(
    __global const float *input,
    __global float *output,
    __local float *local_sum) {
  const int global_size = get_global_size(0);
  const int global_id = get_global_id(0);
  const int local_size = get_local_size(0);
  const int local_id = get_local_id(0);

  local_sum[local_id] = input[global_id];
  barrier(CLK_LOCAL_MEM_FENCE);

  int stride0 = local_size;
  for (int stride = stride0 / 2; stride > 0; stride0 = stride, stride /= 2) {
    if (local_id < stride) {
      local_sum[local_id] += local_sum[local_id + stride];
      if ((stride * 2) < stride0) {
        if (local_id == 0) {
          local_sum[local_id] += local_sum[local_id + (stride0 -1)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    output[global_id] = local_sum[0];
  }
}
