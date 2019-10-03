#include "header.h"

static int n_even, n_full; 
static __float128 *tmp[6];
static __float128 *aux[6];
static __float128 *yk, *mlt, *rk;
static __float128 overN;
static __float128 D;
static __float128 tau;
static work_ptr loc;
static constants_ptr local_consts;

void init_operator_arrays(params_ptr par, work_ptr wrk) {
  n_even = (par->n)/2 + 1;
  n_full = par->n;
  overN = 1.0Q/(par->n); 
  for (int j = 0; j < 6; j++) {
    tmp[j] = fftwq_malloc(sizeof(__float128)*n_even);
    aux[j] = fftwq_malloc(sizeof(__float128)*n_even);
  }
  mlt = fftwq_malloc(sizeof(__float128)*n_even);
  rk = fftwq_malloc(sizeof(__float128)*n_even);
  loc = wrk;
  D = par->qc;
  mlt[0] = 0.Q;
  printf("Vertical Box size:\tqc = %38.31Qe\n", D);
  for (int j = 0; j < n_even; j++) {
    rk[j] = 1.0Q*tanhq(D*j);
    mlt[j] = 1.0Q*j*tanhq(D*j);
  }
  tau = wrk->tau;
}

void operator_absk(__float128 *in, __float128 *out){
  ifft_even(in, tmp[0]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = tmp[0][j]*mlt[j]*overN;
  }
  ifft_even(tmp[0], out);
}

void linear(__float128 *in, __float128 *out) {
  ifft_even(in, tmp[0]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = tmp[0][j]*loc->du[j]*overN;
  }
  ifft_even(tmp[0], tmp[0]);
  for (int j = 0; j < n_even; j++) {
    out[j] = mlt[j]*in[j] - loc->lambda*tmp[0][j];
  }
}

void gravity(__float128 *in, __float128 *out) {
  ifft_even(in, tmp[3]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = mlt[j]*in[j];
    tmp[1][j] = powq(tmp[3][j], 2)*overN;
  }
  ifft_even(tmp[0], tmp[0]);
  ifft_even(tmp[1], tmp[1]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = tmp[3][j]*tmp[0][j]*overN;
  }
  ifft_even(tmp[0], tmp[0]);
  for (int j = 0; j < n_even; j++) {
    out[j] = 0.5Q*mlt[j]*tmp[1][j] + tmp[0][j];
  }
}

void vorticity(__float128 *in, __float128 *out) {
  ifft_even(in, tmp[4]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = powq(tmp[4][j], 2)*(loc->du[j])*overN;
    tmp[1][j] = powq(tmp[4][j], 3)*overN;
    tmp[2][j] = mlt[j]*in[j];
    tmp[3][j] = powq(tmp[4][j], 2)*overN;
  }
  ifft_even(tmp[0], tmp[0]);
  ifft_even(tmp[1], tmp[1]);
  ifft_even(tmp[2], tmp[2]);
  ifft_even(tmp[3], tmp[3]);
  for (int j = 0; j < n_even; j++) {
    tmp[3][j] = mlt[j]*tmp[3][j];
    out[j] = tmp[0][j] + mlt[j]*tmp[1][j]/3.0Q; 
  }
  ifft_even(tmp[3], tmp[3]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = powq(tmp[4][j], 2)*tmp[2][j]*overN;  
    tmp[1][j] = tmp[4][j]*tmp[3][j]*overN; 
  }
  ifft_even(tmp[0], tmp[0]);
  ifft_even(tmp[1], tmp[1]);
  for (int j = 0; j < n_even; j++) out[j] += tmp[0][j] - tmp[1][j];
}

void nonlinear_operator(__float128 *in, __float128 *out) {
  linear(in, aux[0]);
  gravity(in, aux[1]);
  vorticity(in, aux[2]);
  for (int j = 0; j < n_even; j++) {
    out[j] = aux[0][j] - (loc->tau)*aux[1][j] - 0.5Q*powq(loc->xi, 2)*aux[2][j];
  }
}

void nonlinear_operator_ref(__float128 *in, __float128 *out) {
  linear(in, aux[0]);
  gravity(in, aux[1]);
  for (int j = 0; j < n_even; j++) {
    out[j] = aux[0][j] - (loc->tau)*aux[1][j];
    out[j] = -out[j];
  }
}

/*-------  Linearized Operators -------*/

void linear_gravity(__float128 *in, __float128 *out) {
  yk = loc->YYk;
  ifft_even(yk, tmp[0]);
  ifft_even(in, tmp[1]);
  for (int j = 0; j < n_even; j++) {
    tmp[2][j] = mlt[j]*yk[j];
    tmp[3][j] = mlt[j]*in[j];
    tmp[4][j] = tmp[0][j]*tmp[1][j]*overN;
  }
  ifft_even(tmp[2], tmp[2]);
  ifft_even(tmp[3], tmp[3]);
  ifft_even(tmp[4], tmp[4]);
  for (int j = 0; j < n_even; j++) {
    tmp[0][j] = tmp[0][j]*tmp[3][j]*overN;  
    tmp[1][j] = tmp[1][j]*tmp[2][j]*overN;  
  }
  ifft_even(tmp[0], tmp[0]);
  ifft_even(tmp[1], tmp[1]);
  for (int j = 0; j < n_even; j++) {
    out[j] = tmp[0][j] + tmp[1][j] + mlt[j]*tmp[4][j];
  }
}

void linearized_vorticity(__float128 *in, __float128 *out) {
  yk = loc->YYk;
  ifft_even(yk, tmp[0]);
  ifft_even(in, tmp[1]);
  for (int j = 0; j < n_even; j++) {
    tmp[2][j] = 2.Q*tmp[0][j]*tmp[1][j]*(loc->du[j])*overN;
    tmp[3][j] = powq(tmp[0][j], 2)*tmp[1][j]*overN;
  }
  ifft_even(tmp[2], tmp[2]);
  ifft_even(tmp[3], tmp[3]);
  for (int j = 0; j < n_even; j++) {
    out[j] = tmp[2][j] + mlt[j]*tmp[3][j]; 
    // find [ydy, y]
    tmp[2][j] = tmp[0][j]*tmp[1][j]*overN;
    tmp[3][j] = mlt[j]*yk[j];
    tmp[4][j] = tmp[2][j];
  }
  ifft_even(tmp[3], tmp[3]);
  ifft_even(tmp[4], tmp[4]);
  for (int j = 0; j < n_even; j++) {
    tmp[4][j] = mlt[j]*tmp[4][j];
  }
  ifft_even(tmp[4], tmp[4]);
  for (int j = 0; j < n_even; j++) {
    tmp[5][j] = tmp[2][j]*tmp[3][j] - tmp[0][j]*tmp[4][j]*overN; 
  }
  ifft_even(tmp[5], tmp[5]);
  for (int j = 0; j < n_even; j++) {
    out[j] += 2.0Q*tmp[5][j];
    // find [y^2, dy]
    tmp[2][j] = powq(tmp[0][j], 2)*overN;
    tmp[3][j] = mlt[j]*in[j];
  }
  ifft_even(tmp[2], tmp[4]);
  ifft_even(tmp[3], tmp[3]);
  for (int j = 0; j < n_even; j++) {
    tmp[4][j] = mlt[j]*tmp[4][j];
  }
  ifft_even(tmp[4], tmp[4]);
  for (int j = 0; j < n_even; j++) {
    tmp[2][j] = tmp[2][j]*tmp[3][j] - tmp[1][j]*tmp[4][j]*overN;
  } 
  ifft_even(tmp[2], tmp[2]);
  for (int j = 0; j < n_even; j++) {
    out[j] += tmp[2][j];
  }
}

void apply_linearized(__float128 *in, __float128 *out) {
  linear(in, aux[0]);
  linear_gravity(in, aux[1]);
  linearized_vorticity(in, aux[2]);
  for (int j = 0; j < n_even; j++) {
    out[j] = aux[0][j] - (loc->tau)*aux[1][j] - 0.5Q*powq(loc->xi, 2)*aux[2][j];
  }
}

void apply_linearized_ref(__float128 *in, __float128 *out) {
  linear(in, aux[0]);
  linear_gravity(in, aux[1]);
  for (int j = 0; j < n_even; j++) {
    out[j] = aux[0][j] - (loc->tau)*aux[1][j];
    out[j] = -out[j];
  }
}

/* -------------------  Conversion of Constants  ---------------------  */

void convert_constants(params_ptr in, constants_ptr out) {
  __float128 w = in->w, c = in->c;
  local_consts = out;
  ifft_even(loc->YYk, tmp[0]);
  ifft_even(loc->du, tmp[1]);
  for (int j = 0; j < n_even; j++) {
    tmp[2][j] = tmp[0][j]*(c + 0.5Q*w*tmp[0][j])*(loc->du[j]);
  }
  ifft_even(tmp[2], tmp[2]);
  out->y0 = -w*(tmp[2][0]/tmp[1][0]);
  out->c  = in->c + (in->w)*out->y0;
  printf("Form 1: c = c-tilde + omega y_0 = %.12Qe\n", out->c);
  out->b0 = powq(in->c, 2) - powq(out->c, 2) - 2.0Q*out->y0;
}

__float128 get_nonlinear_residual(__float128 *ink) {
  nonlinear_operator(loc->YYk, tmp[0]);
  return sqrtq(dot(tmp[0], tmp[0]));
}

__float128 mean_level(__float128 *yk) {
  ifft_even(yk, tmp[0]);
  for (int j = 0; j < n_even; j++) {
    tmp[1][j] = mlt[j]*yk[j];
  }
  ifft_even(tmp[1], tmp[1]);
  for (int j = 0; j < n_even; j++) {
    tmp[1][j] = tmp[0][j]*(1.0Q + tmp[1][j])*overN;
  }
  ifft_even(tmp[1], tmp[1]);
  return tmp[1][0];
}

// -------------------------------------------------------------- //

__float128 dot(__float128 *in1, __float128 *in2) {
  __float128 aux = 0.Q;
  for (int j = 0; j < n_even; j++) aux += 2.Q*in1[j]*in2[j];
  aux += - (in1[0]*in2[0] + in1[n_even-1]*in2[n_even-1]);
  return aux;
}

/* unused and deprecated */
__float128 dot_H(__float128 *in1, __float128 *in2) {
  __float128 aux = 0.Q;
  for (int j = 0; j < n_even; j++) aux += 2.Q*in1[j]*in2[j]/(mlt[j] - tau);
  aux += in1[0]*in2[0]/tau - in1[n_even-1]*in2[n_even-1]/(mlt[n_even-1]-tau);
  return aux;
}

/*
void get_initial_guess(work_ptr wrk, fftwq_complex *in, __float128 *q_cr) {
  int 		Jmax = 0;
  __float128 	Amax = 0.Q;
  
  for (int j = 0; j < (n_even-0); j++) {
    if (Amax < cimagq( clogq( wrk->du[j] + in[j]))) {
      Amax = cimagq( clogq( wrk->du[j] + in[j]));
      Jmax = j;
    }
  }
  printf("Crude: %.12Qe\t%.12Qe\n", Amax, 1.0Q*Jmax);
  *q_cr = pi*(2.Q*Jmax*overN);
}
*/


void get_initial_guess(work_ptr wrk, __float128 *q_cr) {
  int 		Jmax = 0;
  __float128 	Amax = 0.Q;

  debug_write_cmplx_hmore(wrk, wrk->Z, "half_Z.txt");  // wrk->Z has Z-tilde
  wrk->dZ[0] = 0.Q;
  for (int j = 1; j < n_full/2; j++) {
    wrk->dZ[j] = 1.0IQ*j*wrk->Zk[j];
    wrk->dZ[n_full-j] = -1.0IQ*j*wrk->Zk[n_full-j];
  }

  fft_cmplx(wrk->dZ, wrk->dZ, BACKWARD);
  fftwq_complex ctmp;
  for (int j = 0; j < n_even; j++) {
    ctmp = wrk->dZ[j] + wrk->du[j];
    wrk->EtaX[j] = cimagq(clogq(ctmp))*180.0Q/pi;
  }
  debug_write_hmore(wrk, wrk->EtaX, "half_EtaX.txt"); 

  /* not the global Max, but rather the first max! */
  for (int j = n_even-1; j > 0; j--) {
    if (Amax < cimagq( clogq( wrk->du[j] + wrk->dZ[j]))) {
      Amax = cimagq( clogq( wrk->du[j] + wrk->dZ[j]));
      Jmax = j;
    }
    if (Amax > cimagq( clogq( wrk->du[j] + wrk->dZ[j]))) break;
  }
  printf("Local Max at x = %.32Qe\tEtaX = %.32Qe\n", wrk->u[Jmax]+crealq(wrk->Z[Jmax]), Amax*180.0Q/pi);
  //printf("Crude: %.12Qe\t%.12Qe\n", Amax, 1.0Q*Jmax);
  *q_cr = pi*(2.Q*Jmax*overN);
  //*q_cr = 1.0Q*Jmax;
}




void seek_max_angle(params_ptr par, work_ptr wrk) {
  //int 		n_full = par->n;

  fftwq_complex *d1Z = fftwq_malloc(n_full*sizeof(fftwq_complex));
  fftwq_complex *d2Z = fftwq_malloc(n_full*sizeof(fftwq_complex));
  fftwq_complex *d3Z = fftwq_malloc(n_full*sizeof(fftwq_complex));

  fftwq_complex *dZ = fftwq_malloc(n_full*sizeof(fftwq_complex));
  d1Z[0] = 0.Q;
  d2Z[0] = 0.Q;
  d3Z[0] = 0.Q;
  for (int j = 1; j < n_even; j++) {
    d1Z[j] =  1.IQ*j*wrk->Zk[j];
    d2Z[j] = -1.Q*j*j*wrk->Zk[j];
    d3Z[j] = -1.IQ*j*j*j*wrk->Zk[j];
    d1Z[n_full-j] = -1.IQ*j*wrk->Zk[n_full-j];
    d2Z[n_full-j] = -1.Q*j*j*wrk->Zk[n_full-j];
    d3Z[n_full-j] =  1.IQ*j*j*j*wrk->Zk[n_full-j];
  }
  /*
  fft_cmplx(d1Z, dZ, BACKWARD);
  fft_cmplx(d2Z, d2Z, BACKWARD);
  fft_cmplx(d3Z, d3Z, BACKWARD);
  fft_cmplx(d1Z, d1Z, BACKWARD);
  fft_cmplx(d2Z, d2Z, BACKWARD);
  fft_cmplx(d3Z, d3Z, BACKWARD);
  */
 
  /*
  debug_write_cmplx(wrk, dZ, "dz.txt"); 
  for (int j = 0; j < n_even; j++) {
    dZ[j] = wrk->u[j];
  }
  debug_write_cmplx(wrk, dZ, "u.txt"); 
  for (int j = 0; j < n_even; j++) {
    dZ[j] = wrk->du[j];
  }
  debug_write_cmplx(wrk, dZ, "du.txt");
  */
  //debug_write_cmplx(wrk, d2Z, "d2Z.txt"); 
  //debug_write_cmplx(wrk, d3Z, "d3Z.txt"); 
  
  
  fftwq_complex	z, d1z, d2z, d3z;
  __float128	sn, cn, dn, am, K;
  __float128	d1u, d2u, d3u;
  __float128 	f, df;
  __float128 	q_tmp = 0.6Q;

  /* Get a crude estimate first */
  get_initial_guess(wrk, &q_tmp);
  //if (q_tmp > 0) q_tmp += -pi;
  printf("Crude Estimate:\tq0 = %.12Qe\n", q_tmp);
  //exit(0);

  mp_sn_cn_dn_K_Piq( &sn, &cn, &dn, &K, &q_tmp, &(par->L));
  d1u = 2.Q*(K/pi)*dn;
  d2u = -2.Q*powq(K/pi,2)*(1.Q - dn*dn)*cn/sn;
  d1z = complex_interp(d1Z, q_tmp - pi); 
  d2z = complex_interp(d2Z, q_tmp - pi); 
  f  = cimagq((d2u + d2z)/(d1u + d1z));
  
  /**/
  printf("at %.12Qe\tPhase = %.12Qe\tdPhase = %.12Qe\n", q_tmp, cimagq(clogq(d1u+d1z)), f);
  /**/

  int iter = 0;
  while (1) {
    mp_sn_cn_dn_K_Piq( &sn, &cn, &dn, &K, &q_tmp, &(par->L));
    d1u = 2.Q*(K/pi)*dn;
    d2u = -2.Q*powq(K/pi,2)*(1.Q - dn*dn)*cn/sn;
    d3u = -2.Q*(1.Q - par->L)*(1.Q + par->L)*powq(K/pi,3)*(cn*cn - sn*sn)*dn;
    d1z = complex_interp(d1Z, q_tmp - pi); 
    d2z = complex_interp(d2Z, q_tmp - pi); 
    d3z = complex_interp(d3Z, q_tmp - pi); 
  
    f  = cimagq((d2u + d2z)/(d1u + d1z));
    df = cimagq((d3u + d3z)/(d1u + d1z) - cpowq( (d2u+d2z)/(d1u + d1z), 2));
    printf("Newton Step %2d:\tq = %19.12Qe\t|F(q)| = %19.12Qe\n",iter, q_tmp, fabsq(f));
    if (fabsq(f) < 1e-25Q) break;
    q_tmp = q_tmp - f/df;
    if (q_tmp >  pi) q_tmp += -pi;
    if (q_tmp < -pi) q_tmp +=  pi;
    iter++;
    if (iter == 100) break;
  }
  z = complex_interp(wrk->Zk, q_tmp - pi); 
  am = atan2q(sn,cn);
  wrk->max_angle = cimagq(clogq(d1u+d1z))*180.Q/pi;
  wrk->xmax = -pi + 2.Q*am + crealq(z); 
  wrk->ymax = cimagq(z) - local_consts->y0;
  printf("Found Max Angle %16.9Qf degrees at (x,y) = (%23.16Qe, %23.16Qe)\n", wrk->max_angle, wrk->xmax, wrk->ymax);

}

void restore_map(params_ptr par, work_ptr wrk, char* fname) {
  int n_odd = n_even-2;
  //int n_full = par->n;
  
  for (int j = 0; j < n_odd; j++) {
    wrk->XXk[j] = 1.Q*wrk->YYk[j+1]*mlt[j+1]/(j+1);
  }
  ifft_odd(wrk->XXk, wrk->XX);
  ifft_even(wrk->YYk, wrk->YY);

  wrk->Zk[0] = 1.IQ*wrk->YYk[0];
  for (int j = 1; j < n_even; j++) {
    wrk->Zk[j] =   1.IQ*(-rk[j] + 1.Q)*wrk->YYk[j];
    wrk->Zk[n_full-j] = 1.IQ*(rk[j] + 1.Q)*wrk->YYk[j];
  }
  fft_cmplx(wrk->Zk, wrk->Z, BACKWARD);
}

fftwq_complex complex_interp(fftwq_complex *ink, __float128 q_tmp) {
  int k = n_even - 1;
  //int n_full = 2*k;
  fftwq_complex	w = cexpq(1.IQ*(q_tmp));
  fftwq_complex	z1 = ink[k];
  fftwq_complex	z2 = ink[k];

  for (int j = 1; j < n_full/2; j++) {
      z1 = ink[n_full/2-j] - z1*w;   
      z2 = ink[n_full/2+j] - z2*conjq(w);
  }
  return ink[0] - z1*w - z2*conjq(w);
}

