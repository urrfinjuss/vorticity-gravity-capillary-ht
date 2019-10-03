#include "header.h"

static long int maxit, n;
static __float128 nl_error;
static __float128 nl_toler, min_error;
static __float128 *dyk;
static work_ptr loc;


void init_newton(params_ptr in, work_ptr wrk) {
  maxit = 512;
  nl_toler = 0.0002Q; // was 2e-3
  min_error = 1.0E-26Q;
  loc = wrk;
  n = (in->n/2) + 1;
  dyk = fftwq_malloc(n*sizeof(__float128));
}

void newton_method_cg() {
  long int cg_steps = 0, k = 0;
  nonlinear_operator(loc->YYk, loc->F);
  nl_error = sqrtq(dot(loc->F, loc->F));
  ifft_even(loc->YYk, loc->YY);
  loc->hmax = 0.5Q*(loc->YY[n-1] - loc->YY[0])/pi;
  while (1) {
    //printf("Newton iteration %4d nonlinear residual is %15.8Qe\n", k, nl_error);
    write_newton_step_to_log(k+1, cg_steps, nl_error, loc->hmax);
    cg_steps += iterate_cg_method(nl_toler*nl_error, 640, dyk);
    for (int j = 0; j < n; j++) {
      loc->YYk[j] += dyk[j];
    }
    ifft_even(loc->YYk, loc->YY);
    loc->hmax = 0.5Q*(loc->YY[n-1] - loc->YY[0])/pi;
    nonlinear_operator(loc->YYk, loc->F);
    nl_error = sqrtq(dot(loc->F, loc->F));
    k++;
    if (nl_error < min_error) break;
    //write_to_log(nl_error, 0.Q, cg_steps);
  }
  printf("Newton iteration %4d nonlinear residual is %15.8Qe\n", k, nl_error);
  loc->nl_error = nl_error;
  //char fname[80];
  //sprintf(fname, "./library/HL%.16Qf.spc", loc->hmax);
  //writer_out(loc->YYk, fname, nl_error, loc->YYk[0], loc->hmax, fabsq(loc->YYk[n-1]));
  
}

void newton_method_pcg() {
  long int pcg_steps = 0, k = 0;
  nonlinear_operator(loc->YYk, loc->F);
  nl_error = sqrtq(dot(loc->F, loc->F));
  ifft_even(loc->YYk, loc->YY);
  loc->hmax = 0.5Q*(loc->YY[n-1] - loc->YY[0])/pi;
  while (1) {
    //printf("Newton iteration %4d nonlinear residual is %15.8Qe\n", k, nl_error);
    write_newton_step_to_log(k+1, pcg_steps, nl_error, loc->hmax);
    pcg_steps += iterate_pcg_method(nl_toler*nl_error, 640, dyk);
    for (int j = 0; j < n; j++) {
      loc->YYk[j] += dyk[j];
    }
    ifft_even(loc->YYk, loc->YY);
    loc->hmax = 0.5Q*(loc->YY[n-1] - loc->YY[0])/pi;
    nonlinear_operator(loc->YYk, loc->F);
    nl_error = sqrtq(dot(loc->F, loc->F));
    k++;
    if (nl_error < min_error) break;
    //write_to_log(nl_error, 0.Q, cg_steps);
  }
  printf("Newton iteration %4d nonlinear residual is %15.8Qe\n", k, nl_error);
  loc->nl_error = nl_error;
  //char fname[80];
  //sprintf(fname, "./library/HL%.16Qf.spc", loc->hmax);
  //writer_out(loc->YYk, fname, nl_error, loc->YYk[0], loc->hmax, fabsq(loc->YYk[n-1]));
}
