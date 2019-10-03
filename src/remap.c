#include "header.h"

#define MPFR_PREC	1024

void remap_init(int *argc, char **argv, params_ptr in, work_ptr wrk)
{
  in->w_ptr = wrk; 
  load_parameters(argc, argv, in, wrk);
  init_global_arrays(in, wrk);  // does arctan transform
  mp_setup_grid(in);		// overwrite with HT transform
  init_fftwq_plans();
  init_operator_arrays(in, wrk);
  load_data(in, wrk);
}

int main(int argc, char** argv) {
  work 		wrk, wrk_new;
  params	ctrl, ctrl_new;
  constants	std_consts;
  char		fname[80];

  remap_init(&argc, argv, &ctrl, &wrk);
  //n_even = (ctrl.n)/2 + 1;
  memcpy(&ctrl_new, &ctrl, sizeof(params)); 
  ctrl_new.w_ptr = &wrk_new;
  init_global_arrays(&ctrl_new, &wrk_new);
  ctrl_new.L = 1.0e-1Q;
  mp_setup_grid(&ctrl_new);

  printf("Remapping from L = %Qe\tto\tL = %Qe\n", ctrl.L, ctrl_new.L); 
  
  convert_constants(&ctrl, &std_consts);
  wrk.hmax = 0.5Q*(wrk.YY[ctrl.n/2] - wrk.YY[0])/pi;
  wrk.nl_error = get_nonlinear_residual(wrk.YYk);
  /* Write Spectrum to file */
  sprintf(fname, "%s/HL%025.21Qf.mod.spc", ctrl.libpath, wrk.hmax);
  printf("Data written to %s\n", fname);
  writer_out(wrk.YYk, fname, wrk.nl_error, std_consts.y0, wrk.hmax, fabsq(wrk.YYk[ctrl.n/2]), ctrl.c, std_consts.c, ctrl.w, ctrl.L);
  writer_out(wrk.YYk, ctrl.resname, wrk.nl_error, std_consts.y0, wrk.hmax, fabsq(wrk.YYk[ctrl.n/2]), ctrl.c, std_consts.c, ctrl.w, ctrl.L);

  sprintf(fname, "%s/HL%025.21Qf.mod.txt", ctrl.libpath, wrk.hmax);
  printf("Data written to %s\n", fname);
  write_surface(&wrk, fname, std_consts.c, std_consts.y0, ctrl.w);
  printf("Success\n");

  restore_map(&ctrl, &wrk, "what?where?when?");
  printf("Test EllipticF:\n");
  
  mpfr_t  EllF, Phi, kp, K2, mp_PI;
  mpfr_inits2(MPFR_PREC, EllF, Phi, kp, K2, mp_PI, NULL);
  mpfr_set_float128(kp, ctrl_new.L, MPFR_RNDN);
  mp_jacobi_elliptick(&K2, &kp);
  
  mpfr_const_pi(mp_PI, MPFR_RNDN);
  mpfr_printf("K2 = %.32Re\n", K2);
  __float128 	new_q;
  fftwq_complex	z;

  for (int j = 0; j < ctrl_new.n/2 + 1; j++) {
    mpfr_set_float128(Phi, wrk.u[j], MPFR_RNDN);
    mpfr_add(Phi, Phi, mp_PI, MPFR_RNDN);
    mpfr_div_ui(Phi, Phi, 2, MPFR_RNDN);
    mp_ellf(&EllF, &Phi, &kp);
    mpfr_div(EllF, EllF, K2, MPFR_RNDN);
    mpfr_sub_ui(EllF, EllF, 1, MPFR_RNDN);
    new_q = M_PIq*mpfr_get_float128(EllF, MPFR_RNDN);
    z = complex_interp(wrk.Zk, new_q - M_PIq);
    printf("%4d:\tq = %.32Qe\tY = %.32Qe\n", j, new_q, cimagq(z));
  }
  

  mpfr_clears(EllF, Phi, kp, K2, mp_PI, NULL);
}
