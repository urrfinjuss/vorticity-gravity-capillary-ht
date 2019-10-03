#include "header.h"

static long int n;
static work_ptr loc;
static __float128 *x;
static __float128 *r;
static __float128 *p;
static __float128 *Ap;
static __float128 a, d, g;
static __float128 residual;

void init_cg_method(params_ptr in, work_ptr wrk) {
  loc = wrk;
  n = in->n/2 + 1;
  //lambda = loc->lambda;
  x = fftwq_malloc(n*sizeof(__float128));
  r = fftwq_malloc(n*sizeof(__float128));
  p = fftwq_malloc(n*sizeof(__float128));
  Ap = fftwq_malloc(n*sizeof(__float128));
}
// ---------------  CG classic  ---------------- //
long int setup_cg_step(__float128 tolerance, __float128 *x) {
  memset(x, 0, n*sizeof(__float128));
  for (long int j = 0; j < n; j++) {
    r[j] = -(loc->F[j]);
    p[j] = r[j];
  }
  g = dot(r, r); 
  residual = sqrtq(g);  
  if (residual <= tolerance) return 1;
  else return 0;
}

long int iterate_cg_method(__float128 tolerance, long int maxit, __float128 *x) {
  long int step_number = 0;
  if (setup_cg_step(tolerance, x)) return step_number;
  else while ( step_number <= maxit) {
    apply_linearized(p, Ap);
    a = g/dot(p, Ap);
    for (long int j = 0; j < n; j++) { 
      x[j] = x[j] + a*p[j];
      r[j] = r[j] - a*Ap[j];
    }
    //printf("At CG step %4u residual is %15.8Qe\n", step_number, residual);
    step_number++;
    residual = sqrtq(dot(r, r));
    if (residual <= tolerance) {
      if (VERBOSE_LINEAR_SOLVE) printf("At CG step %4u residual is %15.8Qe\n", step_number, residual);
      break; 
    }
    d = dot(r, r);
    for (long int j = 0; j < n; j++) {
      p[j] = r[j] + (d/g)*p[j];
    }
    g = d;
  }
  if (residual <= tolerance) return step_number;
  else return -1;
}

