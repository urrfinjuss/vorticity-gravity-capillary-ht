#include "header.h"

static int N, n_even, n_full;
static FILE *fh_out;
static __float128 L;
static params_ptr loc_par;
static char strlog[80], line[480];

#define NUM_PARAMS 11

void load_parameters(int *argc, char **argv, params_ptr in, work_ptr wrk) {
  char 	str[480], value[480], param[480];
  int 	num_p = 0;
 
  loc_par = in; 
  sprintf(str, "Usage:\n\t%s input", argv[0]);
  if (*argc != 2) err_msg(str);
  sprintf(str, "%s", argv[1]);

  FILE *fh = fopen(str,"r"); 
  if (fh == NULL) err_msg("Cannot open file");
  while (fgets(line, 480, fh)!=NULL) {
    sscanf(line, "%s\t%s", param, value);
    if (strcmp(param,"runname=") == 0) { sprintf(in->runname,"%s", value); num_p++; }
    if (strcmp(param,"libpath=") == 0) { sprintf(in->libpath,"%s", value); num_p++; }
    if (strcmp(param,"logpath=") == 0) { sprintf(in->logpath,"%s", value); num_p++; }
    if (strcmp(param,"resname=") == 0) { sprintf(in->resname,"%s", value); num_p++; }
    if (strcmp(param,"npoints=") == 0) { in->n = atoi(value); num_p++; }
    if (strcmp(param,"gravity=") == 0) { in->g = strtoflt128 (value, NULL); num_p++; }
    if (strcmp(param,"vrtcity=") == 0) { in->w = strtoflt128 (value, NULL); num_p++; }
    if (strcmp(param,"velocit=") == 0) { in->c = strtoflt128 (value, NULL); num_p++; }
    if (strcmp(param,"perturb=") == 0) { in->alfa = strtoflt128 (value, NULL); num_p++; }
    if (strcmp(param,"nthread=") == 0) { in->nthreads = atoi(value); num_p++; }
    if (strcmp(param,"refin_l=") == 0) { in->L = strtoflt128 (value, NULL); num_p++; }
    sprintf(param, "foo");
  }
  fclose(fh);
  if (num_p != NUM_PARAMS) {
    printf("Found %d entries.\t", num_p);
    err_msg("Missing parameters in conf file\n");
  }
  /* Write the log of iteration convergence to a text file */
  sprintf(strlog, "%s/c%.16Qf.log", in->logpath, in->c); 
  printf("Writing Log to %s\n", strlog);
  FILE *fhlog = fopen(strlog,"w"); 
  fprintf(fhlog, "#input parameters read:\n" );
  fprintf(fhlog, "#runname = %s\n", in->runname );
  fprintf(fhlog, "#libpath = %s\n", in->libpath );
  fprintf(fhlog, "#logpath = %s\n", in->logpath );
  fprintf(fhlog, "#resname = %s\n", in->resname );
  fprintf(fhlog, "#npoints = %d\n", in->n );
  fprintf(fhlog, "#gravity = %.24Qe\n", in->g );
  fprintf(fhlog, "#vrtcity = %.24Qe\n", in->w );
  fprintf(fhlog, "#velocit = %.24Qe\n", in->c );
  fprintf(fhlog, "#perturb = %.24Qe\n", in->alfa );
  fprintf(fhlog, "#nthread = %d\n", in->nthreads );
  fprintf(fhlog, "#refin_l = %.24Qe\n#\n", in->L );
  fprintf(fhlog, "# 1. Newton Cycle 2. CG cycle 3. Residual 4. H/L\n\n");
  fclose(fhlog);
  /* Set static variables */
  N = in->n; n_even = N/2 + 1; L = in->L;
  wrk->tau = (in->g)/powq(in->c, 2);
  wrk->xi = (in->w)/(in->c);
  wrk->lambda = wrk->tau + wrk->xi;
  n_full = in->n;
}

void load_data(params_ptr in, work_ptr wrk) {
  FILE *fh = fopen(in->resname,"r");
  int jj = 0, wave_num = 0; 
  if (fh != NULL) {
    char v1[160], v2[160];
    while (fgets(line, 480, fh)!=NULL) {
      if (jj == 0) {
        if (!fgets(line, 480, fh)) printf("Warning NULL pointer in fgets.\n");
        if (!fgets(line, 480, fh)) printf("Warning NULL pointer in fgets.\n");
        if (!fgets(line, 480, fh)) printf("Warning NULL pointer in fgets.\n");
        if (!fgets(line, 480, fh)) printf("Warning NULL pointer in fgets.\n");
      }
      sscanf(line, "%s\t%s\n",  v1, v2);
      wave_num = atoi(v1);
      if (jj == n_even)    break;
      if (wave_num >= 0) wrk->YYk[jj] = strtoflt128(v2, NULL);
      else break;
      jj++;      
    }
    fclose(fh);
    printf("Lines read = %d of %d expected\n", jj, n_even);
    if ( jj != n_even) {
      printf("Restarting from coarser grid.\n");
      printf("Missing Fourier modes are set to zero.\n");
      for (int j = jj; j < n_even; j++) wrk->YYk[j] = 0.Q;
    }
  } else err_msg("Could not open restart file\n"); 
  for(int j = n_even/2; j < n_even; j++)  {
    wrk->YYk[j] = (wrk->YYk[j])*cexpq((in->alfa)*j); 
  }
  ifft_even(wrk->YYk,wrk->YY);
}

void write_newton_step_to_log(int c1, int c2, __float128 NV, __float128 hmax) {
  printf("Newton iteration %4d nonlinear residual is %15.8Qe\n", c1-1, NV);
  FILE *fhlog = fopen(strlog,"a"); 
  fprintf(fhlog, "%d\t%d\t", c1-1, c2);
  fprintf(fhlog, "%.32Qe\t%.32Qe\n", NV, hmax);
  fclose(fhlog);
}

void write_to_log(__float128 NV, __float128 hmax, int counter) {
  FILE *fhlog = fopen(strlog, "a"); 
  fprintf(fhlog, "# Residual = %Qe at %3d CG step, ", NV, counter);
  fprintf(fhlog, "H/L = %.32Qe\n", hmax);
  fclose(fhlog);
}


void writer_out(__float128* in, char* str, __float128 residual, __float128 y0, __float128 hmax, __float128 minmod, __float128 c, __float128 cs, __float128 w, __float128 l) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "# 1.k 2.y_k\tH/L = %38.31Qe\tMin Fourier mode = %16.10Qe\n", hmax, minmod);
  fprintf(fh_out, "# c-tilde = %20.12Qe\tc = %20.12Qe\tw = %20.12Qe\ty0 = %38.31Qe\n", c, cs, w, y0);
  fprintf(fh_out, "# N = %d\tl = %20.12Qe\tL2_Residual = %20.12Qe\n\n", N, l, residual);
  for (int j = 0; j < N; j++) {
    if (j > N/2) fprintf(fh_out, "%10d\t%38.31Qe\n", j-N, in[N-j]);
    else fprintf(fh_out, "%10d\t%38.31Qe\n", j, in[j]);
  }
  fclose(fh_out);
}

void writer_full(work_ptr wrk, char* str, __float128 c, __float128 y0, __float128 w) {
  int m;
  __float128 u;
  fh_out = fopen(str, "w");
  fprintf(fh_out, "# 1.u 2. x-tilde 3. y 4. EtaX\n");
  fprintf(fh_out, "# N = %ld\tL = %38.31Qe\n", N, L);
  fprintf(fh_out, "# H/L = %29.22Qe\tc = %29.22Qe\ty0 = %29.22Qe\tOmega = %29.22Qe\n\n", wrk->hmax, c, y0, w);
  fprintf(fh_out, "%38.31Qe\t%38.31Qe\t%38.31Qe\t%38.31Qe\n", -pi, 0.Q, wrk->YY[0]-y0, 0.Q);
  for (int j = 1; j < n_even-1; j++) {
    u = wrk->u[j];
    fprintf(fh_out, "%38.31Qe\t%38.31Qe\t%38.31Qe\t%38.31Qe\n", u, wrk->XX[j-1], wrk->YY[j]-y0, wrk->EtaX[j]);
  } 
  fprintf(fh_out, "%38.31Qe\t%38.31Qe\t%38.31Qe\t%38.31Qe\n", 0.Q, 0.Q, wrk->YY[n_even-1]-y0, 0.Q);
  for (int j = 1; j < n_even-1; j++) {
    m = n_even - j;
    u = -1.Q*(wrk->u[m-1]);
    fprintf(fh_out, "%38.31Qe\t%38.31Qe\t%38.31Qe\t%38.31Qe\n", u, -1.Q*(wrk->XX[m-2]), wrk->YY[m-1]-y0, -wrk->EtaX[m-1]);
  }
  fclose(fh_out);
}

/* extra for debug and etc */
void err_msg(char* str)
{
  printf("%s\n",str);
  exit(1);
}

void debug_write(__float128 *in, char* str) {
  fh_out = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh_out, "%10d\t%38.31Qe\n", j, in[j]);
  }
  fclose(fh_out);
}

void debug_write_hmore(work_ptr wrk, __float128 *in, char* str) {
  fh_out = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh_out, "%10d\t%38.31Qe\t%38.31Qe\n", j, wrk->u[j], in[j]);
  }
  fclose(fh_out);
}

void debug_write_cmplx_hmore(work_ptr wrk, fftwq_complex *in, char* str) {
  fh_out = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh_out, "%10d\t%38.31Qe\t%38.31Qe\t%38.31Qe\n", j, wrk->u[j], crealq(in[j]), cimagq(in[j]));
  }
  fclose(fh_out);
}

void debug_write_cmplx(work_ptr wrk, fftwq_complex *in, char* str) {
  fh_out = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh_out, "%29.21Qe\t%29.21Qe\t%29.21Qe\n", wrk->u[j], crealq(in[j]), cimagq(in[j]));
  }
  for (int j = 0; j < n_even-2; j++) {
    fprintf(fh_out, "%29.21Qe\t%29.21Qe\t%29.21Qe\n", -wrk->u[n_even-j-2], crealq(in[n_even+j]), cimagq(in[n_even+j]));
  }
  fclose(fh_out);
}

void debug_write_cmplx_more(work_ptr wrk, fftwq_complex *in, char* str) {
  fh_out = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh_out, "%4d\t%29.21Qe\t%29.21Qe\t%29.21Qe\n", j, wrk->u[j], crealq(in[j]), cimagq(in[j]));
  }
  for (int j = 0; j < n_even-2; j++) {
    fprintf(fh_out, "%4d\t%29.21Qe\t%29.21Qe\t%29.21Qe\n", j+n_even,  -wrk->u[n_even-j-2], crealq(in[n_even+j]), cimagq(in[n_even+j]));
  }
  fclose(fh_out);
}


void write_surface(work_ptr wrk, char* str, __float128 c, __float128 y0, __float128 w) {
  __float128 *mlt = fftwq_malloc(sizeof(__float128)*n_even);
  __float128 D = loc_par->qc;
  for (int j = 0; j < n_even; j++) mlt[j] = 1.0Q*j*tanhq(D*j);

  for (int j = 0; j < n_even-2; j++) {
    wrk->XXk[j] = 1.Q*wrk->YYk[j+1]*mlt[j+1]/(j+1);
  }
  ifft_odd(wrk->XXk, wrk->XX);
  ifft_even(wrk->YYk, wrk->YY);
  wrk->hmax = 0.5Q*(wrk->YY[n_even-1] - wrk->YY[0])/pi;
  writer_full(wrk, str, c, y0, w);

  fftwq_free(mlt);
}

void write_cmplx(work_ptr wrk, char* str) {
  FILE *fh = fopen(str, "w");
  for (int j = 0; j < n_even; j++) {
    fprintf(fh, "%29.21Qe\t%29.21Qe\t%29.21Qe\n", wrk->u[j], crealq(wrk->Z[j]), cimagq(wrk->Z[j]));
  }
  for (int j = 0; j < n_even-2; j++) {
    fprintf(fh, "%29.21Qe\t%29.21Qe\t%29.21Qe\n", -wrk->u[n_even-j-2], crealq(wrk->Z[n_even+j]), cimagq(wrk->Z[n_even+j]));
  }
  fclose(fh);
}





