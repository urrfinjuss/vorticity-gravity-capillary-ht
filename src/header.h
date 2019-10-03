#define MPFR_WANT_FLOAT128
#include <stdio.h>
#include <quadmath.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <pthread.h>
#include <fftw3.h>
#include <mpfr.h>

#define FMODE FFTW_ESTIMATE
#define VERBOSE_LINEAR_SOLVE 0
#define PCG 	1
#define CG 	0
#define pi 	M_PIq
#define ODD	1
#define EVEN 	0
#define MPFR_PREC	1024
#define FORWARD		1
#define BACKWARD	-1

typedef struct vector {
  fftwq_complex *r, *v;
} vect;

typedef struct work {
  int n;
  fftwq_complex 	*Z, *Zk, *dZ;
  __float128 		*YY, *YYk;
  __float128 		*XX, *XXk;
  __float128 		*EtaX;
  __float128 		lambda, tau, sigma, xi;
  __float128 		hmax, nl_error;
  __float128		max_angle, xmax, ymax;
  __float128 		*du, *u, *d2u, *d3u;
  __float128 		*G, *V, *L, *F;
} work, *work_ptr;

typedef struct control {
  __float128 	H;  // energy
  __float128 	M;  // mean level
  __float128 	P;  // momentum
} ctrl, *ctrl_ptr;

typedef struct params {
  int n, d, skip, nthreads;
  __float128 g, c, w, xi, alfa;
  char runname[80];
  char libpath[80];
  char logpath[80];
  char resname[80];
  char cmpname[80];
  __float128 	L, qc;
  ctrl 		current; 
  work_ptr 	w_ptr;
} params, *params_ptr;

typedef struct std_constants {
  __float128 y0, c, b0;
} constants, *constants_ptr;


extern void writer_dfull(work_ptr wrk, char* str);
extern void find_der(work_ptr wrk);
extern void err_msg(char* str);
extern void init(int *argc, char **argv, params_ptr in, work_ptr wrk) ;
extern void hilbert(fftwq_complex *in, fftwq_complex *out);
extern void project(fftwq_complex *in, fftwq_complex *out);
extern void pproject(fftwq_complex *in, fftwq_complex *out);
extern void ffilter(fftwq_complex *in);
extern void reconstruct_surface(work_ptr in);
extern void reconstruct_potential(work_ptr in); 
extern void update_dfunc(work_ptr in);
extern void fft(fftwq_complex *in, fftwq_complex *out);
extern void ifft(fftwq_complex *in, fftwq_complex *out);
extern void fft_even(__float128 *in, __float128 *out); 
extern void der(fftwq_complex *in, fftwq_complex *out);
extern void rhs(work_ptr in, int m);
extern void rk4(params_ptr in);
extern void generate_derivatives(work_ptr in, int m);
extern void clean_spectrum(fftwq_complex *in, fftwq_complex *out);
extern void prepare_rhs(work_ptr wrk);
extern void iterate(work_ptr wrk);
extern void iterate_cg(work_ptr wrk);
extern void iterate_cr(work_ptr wrk);
extern void prep_control_params(params_ptr in);
extern void hamiltonian(params_ptr in);
extern void momentum(params_ptr in);

//  --  memory.c
extern void init(int *argc, char **argv, params_ptr in, work_ptr wrk);
extern void init_global_arrays(params_ptr par, work_ptr wrk);
extern void init_fftwq_plans();
extern void ifft_even(__float128 *in, __float128 *out);
extern void ifft_odd(__float128 *in, __float128 *out);
extern void fft_cmplx(fftwq_complex *in, fftwq_complex *out, const int direction);

//  --  io.c
extern void load_parameters(int *argc, char **argv, params_ptr in, work_ptr wrk);
extern void load_data(params_ptr in, work_ptr wrk);
extern void write_to_log(__float128 NV, __float128 hmax, int counter);
extern void write_newton_step_to_log(int c1, int c2, __float128 NV, __float128 hmax);
extern void writer_full(work_ptr wrk, char* str, __float128 c, __float128 y0, __float128 w);
extern void writer_out(__float128* in, char* str, __float128 residual, __float128 y0, __float128 hmax, __float128 minmod, __float128 c, __float128 cs, __float128 w, __float128 l);
extern void write_surface(work_ptr wrk, char* str, __float128 c, __float128 y0, __float128 w);
extern void read_data(params_ptr in);
extern void debug_write(__float128 *in, char *str);
extern void debug_write_hmore(work_ptr wrk, __float128 *in, char* str);
extern void debug_write_cmplx(work_ptr wrk, fftwq_complex *in, char* str);
extern void debug_write_cmplx_more(work_ptr wrk, fftwq_complex *in, char* str);
extern void debug_write_cmplx_hmore(work_ptr wrk, fftwq_complex *in, char* str);
extern void write_cmplx(work_ptr wrk, char* str);

//  --  operators.c
extern void init_operator_arrays(params_ptr par, work_ptr wrk);
extern void operator_absk(__float128 *in, __float128 *out);
extern void linear(__float128 *in, __float128 *out);
extern void gravity(__float128 *in, __float128 *out);
extern void vorticity(__float128 *in, __float128 *out);
extern void nonlinear_operator(__float128 *in, __float128 *out);
extern void nonlinear_operator_ref(__float128 *in, __float128 *out);
extern void linear_gravity(__float128 *in, __float128 *out);
extern void linearized_vorticity(__float128 *in, __float128 *out);
extern void apply_linearized(__float128 *in, __float128 *out);
extern void apply_linearized_ref(__float128 *in, __float128 *out);
extern void convert_constants(params_ptr in, constants_ptr out);
extern __float128 mean_level(__float128 *yk);
extern __float128 dot(__float128 *in1, __float128 *in2);
extern __float128 get_nonlinear_residual(__float128 *ink);

/*  --  extra functions to get max angle  -- */ 
extern void seek_max_angle(params_ptr par, work_ptr wrk);
extern void restore_map(params_ptr par, work_ptr wrk, char *fname);
extern fftwq_complex complex_interp(fftwq_complex *ink, __float128 q_tmp);

//  --  cg_method.c
extern void init_cg_method(params_ptr in, work_ptr wrk);
extern long int iterate_cg_method(__float128 tolerance, long int maxit, __float128 *x);

//  --  pcg_newton.c
extern void init_pcg_method(params_ptr in, work_ptr wrk);
extern long int iterate_pcg_method(__float128 tolerance, long int maxit, __float128 *x);

//  --  newton.c
extern void init_newton(params_ptr in, work_ptr wrk);
extern void newton_method_cg();
extern void newton_method_pcg();

//  -- mp_mapping.c 
extern void mp_setup_grid(params_ptr in);

//  -- mp_special_functions.c
extern void mp_ellf(mpfr_t *out, mpfr_t *phi, mpfr_t *inkc);
extern void mp_rf(mpfr_t *out, mpfr_t *x, mpfr_t *y, mpfr_t *z);
extern void mp_rd(mpfr_t *out, mpfr_t *x, mpfr_t *y, mpfr_t *z);
extern void mp_jacobi_am(mpfr_t *out, mpfr_t *in, mpfr_t *inkc);
extern void mp_jacobi_elliptick(mpfr_t *out, mpfr_t *ink);
extern void mp_jacobi_elliptickc(mpfr_t *out, mpfr_t *inkc);
extern void mp_sn_cn_dn(__float128 *sn, __float128 *cn, __float128 *dn, __float128 *q0, __float128 *kc);
extern void mp_sn_cn_dn_K_Piq(__float128 *sn, __float128 *cn, __float128 *dn, __float128 *K, __float128 *q0, __float128 *kc);


//  -- postprocess.c (proposed new file, presently in operators.c)
// compute max_angle via Newton's method + spectral interpolation
//extern fftwq_complex spectral_interp(__float128 *ink, __float128 q_tmp, const int parity);
