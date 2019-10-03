#include "header.h"

static fftwq_plan plan_br1, plan_br2, plan_dst;
static fftwq_plan plan_fc1, plan_bc1;
static int n_even, n_odd, nthreads, n_full;
static __float128 *tmp[4];
static __float128 *odd[4];
//static __float128 l, a, b;
static __float128 tau;
static fftwq_complex *tmpc[2];
static char value[32];


void init(int *argc, char **argv, params_ptr in, work_ptr wrk)
{
  in->w_ptr = wrk; 
  load_parameters(argc, argv, in, wrk);
  init_global_arrays(in, wrk);  // does arctan transform
  mp_setup_grid(in);		// overwrite with HT transform
  init_fftwq_plans();
  init_operator_arrays(in, wrk);
  if (CG) init_cg_method(in, wrk);
  if (PCG) init_pcg_method(in, wrk);
  init_newton(in, wrk);
  nthreads = in->nthreads;
  load_data(in, wrk);
}


void init_global_arrays(params_ptr par, work_ptr wrk) {
  n_full = par->n; 
  n_even = (par->n)/2 + 1;
  n_odd  = (par->n)/2 - 1;
  nthreads = par->nthreads;
  wrk->L = fftwq_malloc(n_even*sizeof(__float128));
  wrk->G = fftwq_malloc(n_even*sizeof(__float128));
  wrk->F = fftwq_malloc(n_even*sizeof(__float128));
  wrk->V = fftwq_malloc(n_even*sizeof(__float128));
  wrk->u = fftwq_malloc(n_even*sizeof(__float128));
  wrk->du = fftwq_malloc(n_even*sizeof(__float128));
  wrk->d2u = fftwq_malloc(n_even*sizeof(__float128));
  wrk->d3u = fftwq_malloc(n_even*sizeof(__float128));
  wrk->Z = fftwq_malloc(n_full*sizeof(fftwq_complex));
  wrk->Zk = fftwq_malloc(n_full*sizeof(fftwq_complex));
  wrk->dZ = fftwq_malloc(n_full*sizeof(fftwq_complex));
  wrk->XX  = fftwq_malloc(n_odd*sizeof(__float128));
  wrk->YY = fftwq_malloc(n_even*sizeof(__float128));
  wrk->XXk = fftwq_malloc(n_odd*sizeof(__float128));
  wrk->YYk = fftwq_malloc(n_even*sizeof(__float128));
  wrk->EtaX = fftwq_malloc(n_even*sizeof(__float128));
  //l = 1.Q/par->L;
  //a = 2.0Q*l/(1.0Q + powq(l,2));
  //b = (1.0Q - powq(l,2))/(1.0Q + powq(l,2));
  tau = wrk->tau; 
}

void init_fftwq_plans() {
  if (nthreads != 1) {
    fftwq_init_threads();
    fftwq_plan_with_nthreads(nthreads);
  }
  sprintf(value,"extra/r2r.fftw.%d", nthreads);
  for (int j = 0; j < 4; j++) {
    tmp[j] = fftwq_malloc(n_even*sizeof(__float128));
    odd[j] = fftwq_malloc( n_odd*sizeof(__float128));
  }
  tmpc[0] = fftwq_malloc(n_full*sizeof(fftwq_complex));
  tmpc[1] = fftwq_malloc(n_full*sizeof(fftwq_complex));

  if (fftwq_import_wisdom_from_filename(value) == 0) {
    printf("Creating plans\n");
    fftwq_set_timelimit(10.);
    plan_br1 = fftwq_plan_r2r_1d(n_even, tmp[1], tmp[0], FFTW_REDFT00, FMODE);
    plan_br2 = fftwq_plan_r2r_1d(n_even, tmp[3], tmp[2], FFTW_REDFT00, FMODE);
    plan_fc1 = fftwq_plan_dft_1d(n_full, tmpc[0], tmpc[1], FFTW_FORWARD, FMODE);
    plan_bc1 = fftwq_plan_dft_1d(n_full, tmpc[0], tmpc[1], FFTW_BACKWARD, FMODE);
    fftwq_export_wisdom_to_filename(value);
    err_msg("Plans created. Rerun the computation");
  } else {
    fftwq_set_timelimit(10.);
    plan_br1 = fftwq_plan_r2r_1d(n_even, tmp[1], tmp[0], FFTW_REDFT00, FMODE);
    plan_br2 = fftwq_plan_r2r_1d(n_even, tmp[3], tmp[2], FFTW_REDFT00, FMODE);
    plan_fc1 = fftwq_plan_dft_1d(n_full, tmpc[0], tmpc[1], FFTW_FORWARD, FMODE);
    plan_bc1 = fftwq_plan_dft_1d(n_full, tmpc[0], tmpc[1], FFTW_BACKWARD, FMODE);
  }
  plan_dst = fftwq_plan_r2r_1d(n_odd,  odd[1], odd[0], FFTW_RODFT00, FFTW_ESTIMATE);
}

void ifft_even(__float128 *in, __float128 *out) {
  memcpy(tmp[1], in, n_even*sizeof(__float128));
  fftwq_execute(plan_br1);
  memcpy(out, tmp[0], n_even*sizeof(__float128));
}

/*
void ifft_even_two(__float128 *in, __float128 *out) {
  memcpy(tmp[3], in, n_even*sizeof(__float128));
  fftwq_execute(plan_br2);
  memcpy(out, tmp[2], n_even*sizeof(__float128));
}
*/
void ifft_odd(__float128 *in, __float128 *out) {
  memcpy(odd[1], in, n_odd*sizeof(__float128));
  fftwq_execute(plan_dst);
  memcpy(out, odd[0], n_odd*sizeof(__float128));
}

void write_cmplx2(fftwq_complex *Arr, char* str) {
  FILE *fh = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh, "%29.21Qe\t%29.21Qe\n", crealq(Arr[j]), cimagq(Arr[j]));
  }
  fclose(fh);
}

void fft_cmplx(fftwq_complex *in, fftwq_complex *out, const int direction) {
  if (direction == 1) {
    write_cmplx2(in, "in.txt");
    memcpy(tmpc[0], in, n_full*sizeof(fftwq_complex));
    write_cmplx2(tmpc[0], "tmpc0.txt");
    fftwq_execute(plan_fc1);
    write_cmplx2(tmpc[1], "tmpc1.txt");
    memcpy(out, tmpc[1], n_full*sizeof(fftwq_complex));
  } else if (direction == -1) {
    memcpy(tmpc[0], in, n_full*sizeof(fftwq_complex));
    fftwq_execute(plan_bc1);
    memcpy(out, tmpc[1], n_full*sizeof(fftwq_complex));
  } else {
    printf("unknown flag in fft_cmplx() call.\n");
    exit(0);
  }
}



