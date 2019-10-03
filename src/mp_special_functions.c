#include "header.h"

#define LANDEN_DEPTH	192


void mp_sn_cn_dn(__float128 *sn, __float128 *cn, __float128 *dn, __float128 *q0, __float128 *kc) {
  mpfr_t  mp_am, mp_kc, mp_arg;
  mpfr_t  mp_sn, mp_cn, mp_dn;
  /* initialize MPFR variables */ 
  mpfr_inits2(MPFR_PREC, mp_kc, mp_am, mp_arg, NULL);
  mpfr_inits2(MPFR_PREC, mp_sn, mp_cn, mp_dn, NULL);
  /* set MPFR variables from __float128 */
  mpfr_set_float128(mp_kc, *kc, MPFR_RNDN);
  mpfr_set_float128(mp_arg, *q0, MPFR_RNDN);
  /* compute elliptic functions in MPFR */
  mp_jacobi_am(&mp_am, &mp_arg, &mp_kc);
  mpfr_sin_cos(mp_sn, mp_cn, mp_am, MPFR_RNDN);
  /* compute elliptic dn in MPFR */
  mpfr_div(mp_arg, mp_cn, mp_kc, MPFR_RNDN);
  mpfr_atan2(mp_arg, mp_sn, mp_arg, MPFR_RNDN);  
  mpfr_sin(mp_arg, mp_arg, MPFR_RNDN);
  mpfr_mul(mp_dn, mp_sn, mp_kc, MPFR_RNDN);
  mpfr_div(mp_dn, mp_dn, mp_arg, MPFR_RNDN);
  /* set __float128 to respective values*/
  *sn = mpfr_get_float128(mp_sn, MPFR_RNDN);
  *cn = mpfr_get_float128(mp_cn, MPFR_RNDN);
  *dn = mpfr_get_float128(mp_dn, MPFR_RNDN);
  /* deallocate MPFR variables */
  mpfr_clears(mp_arg, mp_am, mp_kc, NULL);
  mpfr_clears(mp_sn, mp_cn, mp_dn, NULL);
}


void mp_sn_cn_dn_K_Piq(__float128 *sn, __float128 *cn, __float128 *dn, __float128 *K, __float128 *q0, __float128 *kc) {
  mpfr_t  mp_am, mp_kc, mp_arg, mp_K;
  mpfr_t  mp_sn, mp_cn, mp_dn;
  /* initialize MPFR variables */ 
  mpfr_inits2(MPFR_PREC, mp_kc, mp_am, mp_arg, mp_K, NULL);
  mpfr_inits2(MPFR_PREC, mp_sn, mp_cn, mp_dn, NULL);
  /* set MPFR variables from __float128 */
  mpfr_set_float128(mp_kc, *kc, MPFR_RNDN);
  mpfr_set_float128(mp_arg, (*q0)*M_1_PIq, MPFR_RNDN);
  /* compute elliptic functions in MPFR */
  mp_jacobi_elliptick(&mp_K, &mp_kc);
  mpfr_mul(mp_arg, mp_arg, mp_K, MPFR_RNDN);
  mp_jacobi_am(&mp_am, &mp_arg, &mp_kc);
  mpfr_sin_cos(mp_sn, mp_cn, mp_am, MPFR_RNDN);
  /* compute elliptic dn in MPFR */
  mpfr_div(mp_arg, mp_cn, mp_kc, MPFR_RNDN);
  mpfr_atan2(mp_arg, mp_sn, mp_arg, MPFR_RNDN);  
  mpfr_sin(mp_arg, mp_arg, MPFR_RNDN);
  mpfr_mul(mp_dn, mp_sn, mp_kc, MPFR_RNDN);
  mpfr_div(mp_dn, mp_dn, mp_arg, MPFR_RNDN);
  /* set __float128 to respective values*/
  *sn = mpfr_get_float128(mp_sn, MPFR_RNDN);
  *cn = mpfr_get_float128(mp_cn, MPFR_RNDN);
  *dn = mpfr_get_float128(mp_dn, MPFR_RNDN);
  *K  = mpfr_get_float128(mp_K, MPFR_RNDN);
  /* deallocate MPFR variables */
  mpfr_clears(mp_K, mp_arg, mp_am, mp_kc, NULL);
  mpfr_clears(mp_sn, mp_cn, mp_dn, NULL);
}

void mp_jacobi_am(mpfr_t *out, mpfr_t *in, mpfr_t *inkc) {
  int 				i, ii, l;
  mpfr_t			a, b, c, d, emc, u;
  mpfr_t			sn, cn, dn;
  mpfr_t			x, y;
  mpfr_t			em[LANDEN_DEPTH], en[LANDEN_DEPTH];
  mpfr_t			ca;

  /* Initialize MPFR variables */
  mpfr_inits2(MPFR_PREC, sn, cn, dn, ca, x, y, NULL);
  mpfr_inits2(MPFR_PREC, a, b, c, d, emc, u, NULL);
  for (int j = 0; j < LANDEN_DEPTH; j++) mpfr_inits2(MPFR_PREC, em[j], en[j], NULL);
  mpfr_set_float128(ca, 1.0E-27Q, MPFR_RNDN);

  /* set copies of the variables */
  mpfr_sqr(emc, *inkc, MPFR_RNDN);
  mpfr_set(u, *in, MPFR_RNDN);
  
  if ( mpfr_cmp_si(emc, 0) == 0) {
    mpfr_sech(cn, u, MPFR_RNDN);
    mpfr_set(dn, cn, MPFR_RNDN);
    mpfr_tanh(sn, u, MPFR_RNDN);
  } else {

    mpfr_set_ui(a, 1, MPFR_RNDN);   // a = 1
    mpfr_set_ui(dn, 1, MPFR_RNDN);  // dn = 1
    l = LANDEN_DEPTH-1;

    for (i = 0; i < LANDEN_DEPTH; i++) {
      mpfr_set(em[i], a, MPFR_RNDN);   	// m_array[i] = a
      mpfr_sqrt(emc, emc, MPFR_RNDN);  	// m_comp = sqrt(m_comp)
      mpfr_set(en[i], emc, MPFR_RNDN); 	// n_array[i] = m_comp
      mpfr_add(c, a, emc, MPFR_RNDN);	// c = a + m_comp
      mpfr_div_ui(c, c, 2, MPFR_RNDN);  // c = c/2
      
      mpfr_sub(x, a, emc, MPFR_RNDN);		// x = a - m_comp
      mpfr_abs(x, x, MPFR_RNDN);    		// x = |a - m_comp|
      mpfr_mul(y, ca, a, MPFR_RNDN);  		// y = CA*a
      if (mpfr_cmp(x, y) <= 0) {
        l = i;					// l = i
      	break;					// break
      }
      mpfr_mul(emc, emc, a, MPFR_RNDN);		// m_comp = m_comp * a
      mpfr_set(a, c, MPFR_RNDN);		// a = c
    }
    mpfr_mul(u, u, c, MPFR_RNDN);		// u_copy = c * u_copy
    mpfr_sin_cos(sn, cn, u, MPFR_RNDN);		// sn = sin(u_copy) 
    						// cn = cos(u_copy)
    if (mpfr_cmp_si(sn, 0) != 0 ) {		// if (sn != 0)
      mpfr_div(a, cn, sn, MPFR_RNDN);		// a = cn/sn
      mpfr_mul(c, c, a, MPFR_RNDN);		// c = c * a

      for (ii = l; 0 <= ii; ii--) {		// 
        mpfr_set(b, em[ii], MPFR_RNDN);		// b = m_array[i]
	mpfr_mul(a, a, c, MPFR_RNDN);		// a = c * a
	mpfr_mul(c, c, dn, MPFR_RNDN);		// c = c * dn

        mpfr_add(x, b, a, MPFR_RNDN);		// x = b + a
	mpfr_add(y, en[ii], a, MPFR_RNDN);	// y = n_array[i] + a
	mpfr_div(dn, y, x, MPFR_RNDN);		// dn = ( n_array[i] + a ) / ( b + a )
	mpfr_div(a, c, b, MPFR_RNDN);		// a  = c / b
      }
      mpfr_mul(a, c, c, MPFR_RNDN);		// a = c * c
      mpfr_add_ui(a, a, 1, MPFR_RNDN);		// a = c * c + 1
      mpfr_sqrt(a, a, MPFR_RNDN);		// a = sqrt( c * c + 1 )
      mpfr_ui_div(a, 1, a, MPFR_RNDN);		// a = 1 / sqrt ( c * c + 1 )

      if (mpfr_cmp_si(sn, 0) > 0) mpfr_set(sn, a, MPFR_RNDN);
      else mpfr_neg(sn, a, MPFR_RNDN);

      mpfr_mul(cn, c, sn, MPFR_RNDN); 		// cn = c * sn
    }
  } 
  mpfr_atan2(*out, sn, cn, MPFR_RNDN);
  mpfr_clears(sn, cn, dn, x, y, NULL);
  mpfr_clears(a, b, c, d, emc, u, NULL);

  for (int j = LANDEN_DEPTH-1; 0 <= j; j--) mpfr_clears(em[j], en[j], NULL);
}

void mp_jacobi_elliptick(mpfr_t *out, mpfr_t *ink) {
  mpfr_t 	one, tmp;
  
  mpfr_inits2(MPFR_PREC, one, tmp, NULL);
  mpfr_set_ui(one, 1, MPFR_RNDN);

  mpfr_agm(*out, one, *ink, MPFR_RNDN);
  mpfr_const_pi(tmp, MPFR_RNDN);
  mpfr_div(*out, tmp, *out, MPFR_RNDN);
  mpfr_div_ui(*out, *out, 2, MPFR_RNDN);
  
  mpfr_clears(one, tmp, NULL);
}

void mp_jacobi_elliptickc(mpfr_t *out, mpfr_t *inkc) {
  mpfr_t 	one, tmp;
  
  mpfr_inits2(MPFR_PREC, one, tmp, NULL);
  mpfr_set_ui(one, 1, MPFR_RNDN);

  mpfr_fms(tmp, *inkc, *inkc, one, MPFR_RNDN);
  mpfr_neg(tmp, tmp, MPFR_RNDN);
  mpfr_sqrt(tmp, tmp, MPFR_RNDN);

  mpfr_agm(*out, one, tmp, MPFR_RNDN);
  mpfr_const_pi(tmp, MPFR_RNDN);
  mpfr_div(*out, tmp, *out, MPFR_RNDN);
  mpfr_div_ui(*out, *out, 2, MPFR_RNDN);
  
  mpfr_clears(one, tmp, NULL);
}

/* Carlson's elliptic integrals R_f and R_d */ 

void mp_rf(mpfr_t *out, mpfr_t *x, mpfr_t *y, mpfr_t *z) {
  /*
   * Computes Carlson's elliptic integral of the first kind, R_f(x,y,z). 
   * x, y and z must be nonnegative, and at most one can be zero.
   *
   * Algorithm from Numerical Recipies 3rd Edition, page 312 
   * programmed with GNU MPFR
   */
   mpfr_t 	ErrTol, OneThird;
   mpfr_t	C1, C2, C3, C4;
   mpfr_t	xt, yt, zt, e2, e3;
   mpfr_t	delx, dely, delz, maxt;
   mpfr_t	sqrtx, sqrty, sqrtz, alamb, ave;

   mpfr_inits2(MPFR_PREC, ErrTol, OneThird, C1, C2, C3, C4, NULL);
   mpfr_set_ui(OneThird, 1, MPFR_RNDN); mpfr_div_ui(OneThird, OneThird, 3, MPFR_RNDN);
   mpfr_set_ui(C1, 1, MPFR_RNDN); mpfr_div_ui(C1, C1, 24, MPFR_RNDN);
   mpfr_set_ui(C2, 1, MPFR_RNDN); mpfr_div_ui(C2, C2, 10, MPFR_RNDN);
   mpfr_set_ui(C3, 3, MPFR_RNDN); mpfr_div_ui(C3, C3, 44, MPFR_RNDN);
   mpfr_set_ui(C4, 1, MPFR_RNDN); mpfr_div_ui(C4, C4, 14, MPFR_RNDN);
   mpfr_set_d(ErrTol, 2.56e-6, MPFR_RNDN);
   mpfr_inits2(MPFR_PREC, xt, yt, zt, e2, e3, NULL);
   mpfr_inits2(MPFR_PREC, delx, dely, delz, maxt, NULL);
   mpfr_inits2(MPFR_PREC, sqrtx, sqrty, sqrtz, alamb, ave, NULL);
   /* the actual algorithm */
   mpfr_set(xt, *x, MPFR_RNDN);
   mpfr_set(yt, *y, MPFR_RNDN);
   mpfr_set(zt, *z, MPFR_RNDN);

   mpfr_set_ui(maxt, 1, MPFR_RNDN);
   while (mpfr_cmp(maxt, ErrTol) > 0 ) {
     mpfr_sqrt(sqrtx, xt, MPFR_RNDN); 
     mpfr_sqrt(sqrty, yt, MPFR_RNDN);
     mpfr_sqrt(sqrtz, zt, MPFR_RNDN);

     mpfr_add(alamb, sqrty, sqrtz, MPFR_RNDN);
     mpfr_fmma(alamb, sqrtx, alamb, sqrty, sqrtz, MPFR_RNDN);

     mpfr_add(xt, xt, alamb, MPFR_RNDN);
     mpfr_add(yt, yt, alamb, MPFR_RNDN);
     mpfr_add(zt, zt, alamb, MPFR_RNDN);
     mpfr_div_ui(xt, xt, 4, MPFR_RNDN);
     mpfr_div_ui(yt, yt, 4, MPFR_RNDN);
     mpfr_div_ui(zt, zt, 4, MPFR_RNDN);

     mpfr_add(ave, xt, yt, MPFR_RNDN);
     mpfr_add(ave, ave, zt, MPFR_RNDN);
     mpfr_mul(ave, ave, OneThird, MPFR_RNDN); 

     mpfr_sub(delx, ave, xt, MPFR_RNDN);
     mpfr_sub(dely, ave, yt, MPFR_RNDN);
     mpfr_sub(delz, ave, zt, MPFR_RNDN);
     mpfr_div(delx, delx, ave, MPFR_RNDN);
     mpfr_div(dely, dely, ave, MPFR_RNDN);
     mpfr_div(delz, delz, ave, MPFR_RNDN);

     mpfr_abs(maxt, delx, MPFR_RNDN);
     mpfr_abs(e2,   dely, MPFR_RNDN);
     mpfr_abs(e3,   delz, MPFR_RNDN);
     mpfr_max(maxt, maxt, e2, MPFR_RNDN);
     mpfr_max(maxt, maxt, e3, MPFR_RNDN);
   }
   mpfr_fmms(e2, delx, dely, delz, delz, MPFR_RNDN);
   mpfr_mul(e3, delx, dely, MPFR_RNDN);
   mpfr_mul(e3, e3, delz, MPFR_RNDN);

   
   mpfr_fmms(*out, C1, e2, C3, e3, MPFR_RNDN);
   mpfr_sub(*out, *out, C2, MPFR_RNDN);
   mpfr_fmma(*out, *out, e2, C4, e3, MPFR_RNDN);
   mpfr_add_ui(*out, *out, 1, MPFR_RNDN);
   mpfr_sqrt(ave, ave, MPFR_RNDN);
   mpfr_div(*out, *out, ave, MPFR_RNDN);

   mpfr_clears(ErrTol, OneThird, C1, C2, C3, C4, NULL);
   mpfr_clears(xt, yt, zt, e2, e3, NULL);
   mpfr_clears(delx, dely, delz, maxt, NULL);
   mpfr_clears(sqrtx, sqrty, sqrtz, alamb, ave, NULL);
}

void mp_rd(mpfr_t *out, mpfr_t *x, mpfr_t *y, mpfr_t *z) {
  /*
   * Computes Carlson's elliptic integral of the first kind, R_d(x,y,z). 
   * x, y and z must be nonnegative, and at most one can be zero.
   *
   * Algorithm from Numerical Recipies 3rd Edition, page 312-313 
   * programmed with GNU MPFR
   */
   mpfr_t 	ErrTol;
   mpfr_t	C1, C2, C3, C4, C5, C6;
   mpfr_t	xt, yt, zt, fac, sum;
   mpfr_t	ea, eb, ec, ed, ee;
   mpfr_t	delx, dely, delz, maxt;
   mpfr_t	sqrtx, sqrty, sqrtz, alamb, ave;

   mpfr_inits2(MPFR_PREC, ErrTol, C1, C2, C3, C4, C5, C6, NULL);
   mpfr_set_ui(C1, 3, MPFR_RNDN); mpfr_div_ui(C1, C1, 14, MPFR_RNDN);
   mpfr_set_ui(C2, 1, MPFR_RNDN); mpfr_div_ui(C2, C2, 6, MPFR_RNDN);
   mpfr_set_ui(C3, 9, MPFR_RNDN); mpfr_div_ui(C3, C3, 22, MPFR_RNDN);
   mpfr_set_ui(C4, 3, MPFR_RNDN); mpfr_div_ui(C4, C4, 26, MPFR_RNDN);
   mpfr_set_ui(C5, 9, MPFR_RNDN); mpfr_div_ui(C5, C5, 88, MPFR_RNDN);
   mpfr_set_ui(C6, 9, MPFR_RNDN); mpfr_div_ui(C6, C6, 52, MPFR_RNDN);
   mpfr_set_d(ErrTol, 2.56e-6, MPFR_RNDN);
   mpfr_inits2(MPFR_PREC, xt, yt, zt, fac, sum, NULL);
   mpfr_inits2(MPFR_PREC, ea, eb, ec, ed, ee, NULL);
   mpfr_inits2(MPFR_PREC, delx, dely, delz, maxt, NULL);
   mpfr_inits2(MPFR_PREC, sqrtx, sqrty, sqrtz, alamb, ave, NULL);
   /* the actual algorithm */
   mpfr_set(xt, *x, MPFR_RNDN);
   mpfr_set(yt, *y, MPFR_RNDN);
   mpfr_set(zt, *z, MPFR_RNDN);
   mpfr_set_ui(sum, 0, MPFR_RNDN);
   mpfr_set_ui(fac, 1, MPFR_RNDN);

   mpfr_set_ui(maxt, 1, MPFR_RNDN);
   while (mpfr_cmp(maxt, ErrTol) > 0 ) {
     mpfr_sqrt(sqrtx, xt, MPFR_RNDN); 
     mpfr_sqrt(sqrty, yt, MPFR_RNDN);
     mpfr_sqrt(sqrtz, zt, MPFR_RNDN);

     mpfr_add(alamb, sqrty, sqrtz, MPFR_RNDN);
     mpfr_fmma(alamb, sqrtx, alamb, sqrty, sqrtz, MPFR_RNDN);
    
     mpfr_fmma(ave, sqrtz, zt, sqrtz, alamb, MPFR_RNDN);
     mpfr_div(ave, fac, ave, MPFR_RNDN);
     mpfr_add(sum, sum, ave, MPFR_RNDN);
     mpfr_div_ui(fac, fac, 4, MPFR_RNDN);

     mpfr_add(xt, xt, alamb, MPFR_RNDN);
     mpfr_add(yt, yt, alamb, MPFR_RNDN);
     mpfr_add(zt, zt, alamb, MPFR_RNDN);
     mpfr_div_ui(xt, xt, 4, MPFR_RNDN);
     mpfr_div_ui(yt, yt, 4, MPFR_RNDN);
     mpfr_div_ui(zt, zt, 4, MPFR_RNDN);

     mpfr_mul_ui(ave, zt, 3, MPFR_RNDN);
     mpfr_add(ave, ave, yt, MPFR_RNDN);
     mpfr_add(ave, ave, xt, MPFR_RNDN);
     mpfr_div_ui(ave, ave, 5, MPFR_RNDN);

     mpfr_sub(delx, ave, xt, MPFR_RNDN);
     mpfr_sub(dely, ave, yt, MPFR_RNDN);
     mpfr_sub(delz, ave, zt, MPFR_RNDN);
     mpfr_div(delx, delx, ave, MPFR_RNDN);
     mpfr_div(dely, dely, ave, MPFR_RNDN);
     mpfr_div(delz, delz, ave, MPFR_RNDN);

     mpfr_abs(maxt, delx, MPFR_RNDN);
     mpfr_abs(ea,   dely, MPFR_RNDN);
     mpfr_abs(eb,   delz, MPFR_RNDN);
     mpfr_max(maxt, maxt, ea, MPFR_RNDN);
     mpfr_max(maxt, maxt, eb, MPFR_RNDN);
   }

   mpfr_mul(ea, delx, dely, MPFR_RNDN);
   mpfr_mul(eb, delz, delz, MPFR_RNDN);
   mpfr_sub(ec, ea, eb, MPFR_RNDN);
   mpfr_mul_ui(ed, eb, 6, MPFR_RNDN);
   mpfr_sub(ed, ea, ed, MPFR_RNDN);
   mpfr_mul_ui(ee, ec, 2, MPFR_RNDN);
   mpfr_add(ee, ed, ee, MPFR_RNDN);
   
   mpfr_mul(alamb, delz, C4, MPFR_RNDN);
   mpfr_fmms(alamb, ea, alamb, C3, ec, MPFR_RNDN);
   mpfr_fmma(alamb, C2, ee, delz, alamb, MPFR_RNDN);

   mpfr_mul(maxt, delz, C6, MPFR_RNDN);
   mpfr_fmms(maxt, C5, ed, maxt, ee, MPFR_RNDN);
   mpfr_sub(maxt, maxt, C1, MPFR_RNDN);

   mpfr_fmma(*out, ed, maxt, delz, alamb, MPFR_RNDN);
   mpfr_add_ui(*out, *out, 1, MPFR_RNDN);
   mpfr_div(*out, *out, ave, MPFR_RNDN);
   mpfr_sqrt(ave, ave, MPFR_RNDN);
   mpfr_div(*out, *out, ave, MPFR_RNDN);
   
   mpfr_mul_ui(sum, sum, 3, MPFR_RNDN);
   mpfr_fma(*out, *out, fac, sum, MPFR_RNDN);

   mpfr_clears(ErrTol, C1, C2, C3, C4, C5, C6, NULL);
   mpfr_clears(xt, yt, zt, fac, sum, NULL);
   mpfr_clears(ea, eb, ec, ed, ee, NULL);
   mpfr_clears(delx, dely, delz, maxt, NULL);
   mpfr_clears(sqrtx, sqrty, sqrtz, alamb, ave, NULL);
}

/* Legendre elliptic integrals */

void mp_ellf(mpfr_t *out, mpfr_t *phi, mpfr_t *inkc) {
   /* Legendre elliptical integral of the second kind E(\phi, \sqrt{1 - k'^2}), evaluated
    * using Carlson's function's R_D and R_F. The argument ranges 0 \leq \phi \leq \pi/2
    * 0 \leq k sin \phi \leq 1
    */
   mpfr_t 	s, c, ak, tmp;
   mpfr_inits2(MPFR_PREC, s, c, ak, tmp, NULL);
   
   mpfr_mul(ak, *inkc, *inkc, MPFR_RNDN);
   mpfr_ui_sub(ak, 1, ak, MPFR_RNDN);
   mpfr_sqrt(ak, ak, MPFR_RNDN);

   mpfr_sin_cos(s, c, *phi, MPFR_RNDN); 
   mpfr_mul(c, c, c, MPFR_RNDN); 

   mpfr_mul(tmp, s, ak, MPFR_RNDN);
   mpfr_mul(tmp, tmp, tmp, MPFR_RNDN);
   mpfr_ui_sub(tmp, 1, tmp, MPFR_RNDN);

   mpfr_set_ui(ak, 1, MPFR_RNDN);

   mp_rf(out, &c, &tmp, &ak);
   mpfr_mul(*out, s, *out, MPFR_RNDN);
   mpfr_clears(s, c, ak, tmp, NULL); 
}

/* High Level function, it should be programmed in a different module */
/*
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

  in->qc = M_PIq*mpfr_get_float128(K1, MPFR_RNDN)/k0;
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
*/





