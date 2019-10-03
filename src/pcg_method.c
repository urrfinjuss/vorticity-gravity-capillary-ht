#include "header.h"

static long int n;
static work_ptr loc;
static __float128 *x;
static __float128 *r;
static __float128 *z;
static __float128 *p;
static __float128 *H;
static __float128 *Ap;
static __float128 a, b, t[2], rho;
static __float128 residual;

void init_pcg_method(params_ptr in, work_ptr wrk) {
  loc = wrk;
  n = in->n/2 + 1;
  //lambda = loc->lambda;
  x = fftwq_malloc(n*sizeof(__float128));
  r = fftwq_malloc(n*sizeof(__float128));
  z = fftwq_malloc(n*sizeof(__float128));
  p = fftwq_malloc(n*sizeof(__float128));
  H = fftwq_malloc(n*sizeof(__float128));
  Ap = fftwq_malloc(n*sizeof(__float128));
  for (long int j = 0; j < n; j++) {
    //H[j] = 1.0Q/(j - loc->lambda);
    H[j] = 1.0Q/(j + loc->lambda);
    H[j] = 1.0Q/sqrtq(powq(j,2) + powq(loc->lambda,2));
    //H[j] = 1.0Q;
  }
}
// ---------------  PCG classic  ---------------- //
long int setup_pcg_step(__float128 tolerance, __float128 *x) {
  memset(x, 0, n*sizeof(__float128));
  for (long int j = 0; j < n; j++) {
    r[j] = -(loc->F[j]);
    p[j] = r[j];
  }
  rho = dot(r, r); 
  residual = sqrtq(rho);  
  if (residual <= tolerance) return 1;
  else return 0;
}

long int iterate_pcg_method(__float128 tolerance, long int maxit, __float128 *x) {
  long int step_number = 0;
  if (setup_pcg_step(tolerance, x)) return step_number;
  else while ( step_number <= maxit) {
    for (long int j = 0; j < n; j++) {
      z[j] = H[j]*r[j];
    }
    t[1] = dot(z, r);
    if (step_number == 0) b = 0.Q;
    else b = t[1]/t[0];
    for (long int j = 0; j < n; j++) {
      p[j] = z[j] + b*p[j];
    }
    apply_linearized(p, Ap);
    a = t[1]/dot(p, Ap);
    for (long int j = 0; j < n; j++) { 
      x[j] = x[j] + a*p[j];
      r[j] = r[j] - a*Ap[j];
    }
    t[0] = t[1];
    rho = dot(r, r);
    step_number++;
    residual = sqrtq(rho);
    if (residual <= tolerance) {
      if (VERBOSE_LINEAR_SOLVE) printf("At PCG step %4u residual is %15.8Qe\n", step_number, residual);
      break; 
    }
  }
  if (residual <= tolerance) return step_number;
  else return -1;
}

