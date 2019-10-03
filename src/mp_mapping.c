#include "header.h"

void mp_complement(mpfr_t *outk, mpfr_t *ink) {
  mpfr_mul(*outk, *ink, *ink, MPFR_RNDN);
  mpfr_ui_sub(*outk, 1, *outk, MPFR_RNDN);
  mpfr_sqrt(*outk, *outk, MPFR_RNDN);
}

void mp_setup_grid(params_ptr in) {
  __float128 	k0; 
  mpfr_t 	x, y, mp_l, mp_am, mp_theta, mp_kc;
  mpfr_t	mp_u, mp_q, mp_tmp, mp_pi, K, K1;
  work_ptr      wrk = in->w_ptr;
  int 		N, n_even;

  N = in->n; n_even = N/2 + 1;
  mpfr_inits2(MPFR_PREC, mp_kc, K, K1, NULL);
 
  mpfr_set_float128(mp_kc, in->L, MPFR_RNDN);
  mp_jacobi_elliptick(&K, &mp_kc);
  mp_jacobi_elliptickc(&K1, &mp_kc);
  k0 = mpfr_get_float128(K, MPFR_RNDN);

  mpfr_clears(K, NULL);

  in->qc = (M_PIq/k0)*mpfr_get_float128(K1, MPFR_RNDN);
  k0 = k0*M_2_PIq;
  wrk->u[0] = -M_PIq;
  wrk->du[0] = k0;

  mpfr_inits2(MPFR_PREC, x, y, mp_l, mp_am, mp_theta, NULL);
  mpfr_inits2(MPFR_PREC, mp_q, mp_tmp, mp_u, mp_pi, K, NULL);
  mpfr_set_float128(mp_tmp, M_PIq*k0, MPFR_RNDN);
  mpfr_const_pi(mp_pi, MPFR_RNDN);

  for (long int j = 1; j < n_even-1; j++) {
    // MPFR q //
    mpfr_mul_si(mp_q, mp_tmp, j, MPFR_RNDN);
    mpfr_div_ui(mp_q, mp_q, N, MPFR_RNDN);
    mp_jacobi_am(&mp_am, &mp_q, &mp_kc);  
    // the gridpoints (MPFR)
    mpfr_mul_ui(mp_u, mp_am, 2, MPFR_RNDN);
    mpfr_sub(mp_u, mp_u, mp_pi, MPFR_RNDN);
    wrk->u[j] = mpfr_get_float128(mp_u, MPFR_RNDN);
    // geometric form (MPFR)		VII
    mpfr_sin_cos(x, y, mp_am, MPFR_RNDN);
    mpfr_div(y, y, mp_kc, MPFR_RNDN);
    mpfr_atan2(mp_theta, x, y, MPFR_RNDN);
    mpfr_sin(y, mp_theta, MPFR_RNDN);
    mpfr_div(y, y, mp_kc, MPFR_RNDN);
    mpfr_div(mp_l, x, y, MPFR_RNDN);
    // set in quad precision 
    wrk->du[j] = mpfr_get_float128(mp_l, MPFR_RNDN);
    wrk->du[j] = k0*(wrk->du[j]);
  }
  wrk->u[N/2] = 0.Q;
  wrk->du[N/2] = (in->L)*k0;
  mpfr_clears(x, y, mp_theta, mp_l, mp_am, mp_kc, mp_q, mp_tmp, K, K1, NULL);
}

void mp_remapping(params_ptr in1, params_ptr in2) {


}





