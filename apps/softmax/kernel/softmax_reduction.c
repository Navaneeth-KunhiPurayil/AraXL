#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#include "../softmax/lib/exp.h"

// Code implementing softmax without stripmining
/*
void softmax_vec_reduction(const double *i, const double *o, uint64_t channels,
                 uint64_t innerSize) {

  size_t avl = innerSize;
  size_t vl;

  double *i_ = (double *) i;
  double *o_ = (double *) o;

  vl = vsetvl_e64m1(avl); // For now assuming avl fits VRF, so vl = avl

  vfloat64m1_t vec_zero = vfmv_v_f_f64m1(0, vl);
  
  vfloat64m1_t vec_res;
  vfloat64m1_t vec_a = vle64_v_f64m1(i_, vl);
  i_ += vl;
  
  for (uint64_t c=0; c<channels; c+=1) {
    // Find max
    vfloat64m1_t vec_red_max;
    vec_red_max = vfredmax_vs_f64m1_f64m1(vec_red_max, vec_a, vec_zero, vl);
    
    if (c > 0) {
      vse64_v_f64m1(o_, vec_res, vl);
      o_ += vl;
    }

    double max = vfmv_f_s_f64m1_f64(vec_red_max);
    vfloat64m1_t vec_b = vfsub_vf_f64m1(vec_a, max, vl);
    
    // Find exp
    vfloat64m1_t vec_c = __exp_1xf64(vec_b, vl);
    
    // Sum and divide
    vfloat64m1_t vec_red_sum;
    vec_red_sum = vfredusum_vs_f64m1_f64m1(vec_red_sum, vec_c, vec_zero, vl);
    
    // Load next row
    if (c+1 < channels) {
      vec_a = vle64_v_f64m1(i_, vl);
    }

    double sum = vfmv_f_s_f64m1_f64(vec_red_sum);
    double sum_inv = 1.0/sum;
    
    vec_res = vfmul_vf_f64m1(vec_c, sum_inv, vl);
    i_ += vl;
    
  }

  vse64_v_f64m1(o_, vec_res, vl);
}
*/

// Code implementing softmax with stripmining
// Assumes row major order for input
// Row wise softmax operation
void softmax_vec_reduction(const double *i, const double *o, uint64_t channels, uint64_t innerSize) {

vfloat64m1_t buf_a;

double *i_ = (double *)i;
double *o_ = (double *)o;

double *is = (double *)i;
double *os = (double *)o;

size_t vl, avl;

for (int c=0; c<channels; c++) {
i_ = is;
avl = innerSize;

// Load the first portion of the long vector
vl = vsetvl_e64m1(avl);
buf_a = vle64_v_f64m1(i_, vl);
i_ += vl;

// Stripmining
avl -= vl;
for (; avl > 0; avl-=vl) {
    // Load the next remaining vector
    vl = vsetvl_e64m1(avl);
    vfloat64m1_t buf_b = vle64_v_f64m1(i_, vl);

    // Do a vector-vector max operation
    buf_a = vfmax_vv_f64m1(buf_a, buf_b, vl);

    // Update vector length
    i_ += vl;
}

// Reduce the max present in buf_a
vfloat64m1_t vec_zero = vfmv_v_f_f64m1(0, vl);

vfloat64m1_t vec_red_max;
vec_red_max = vfredmax_vs_f64m1_f64m1(vec_red_max, buf_a, vec_zero, vl);

double max = vfmv_f_s_f64m1_f64(vec_red_max);

// Reset avl, i_
avl = innerSize;
double *i1_ = (double *)is;
vfloat64m1_t buf_d = vfmv_v_f_f64m1(0, vl);

// Stripmine and find exponentials
for (; avl > 0; avl-= vl) {
    vl = vsetvl_e64m1(avl);
    vfloat64m1_t buf_a = vle64_v_f64m1(i1_, vl);
    vfloat64m1_t buf_b = vfsub_vf_f64m1(buf_a, max, vl);
    
    // Find exp
    vfloat64m1_t buf_c = __exp_1xf64(buf_b, vl);

    buf_d = vfadd_vv_f64m1(buf_c, buf_d, vl);
    vse64_v_f64m1(i1_, buf_c, vl);
    i1_ += vl;
}

// Reset avl, i_
avl = innerSize;
double *i2_ = (double *)is;
o_ = (double *)os;
vl = vsetvl_e64m1(avl);

// Reduction to find sum of exponentials
vfloat64m1_t vec_red_sum;
vec_red_sum = vfredusum_vs_f64m1_f64m1(vec_red_sum, buf_d, vec_zero, vl);

double sum = vfmv_f_s_f64m1_f64(vec_red_sum);
double sum_inv = 1.0/sum;

// Stripmining to the last multiplications
for (; avl > 0; avl-= vl) {
    vl = vsetvl_e64m1(avl);
    vfloat64m1_t buf_a = vle64_v_f64m1(i2_, vl);
    i2_ += vl;
    vfloat64m1_t buf_b = vfmul_vf_f64m1(buf_a, sum_inv, vl);
    vse64_v_f64m1(o_, buf_b, vl);
    o_ += vl;
}

// Update is, os
is += innerSize;
os += innerSize;
}

}

