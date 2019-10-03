#include "header.h"

int main(int argc, char** argv)
{
  work	  	wrk;
  params  	ctrl;
  constants	std_consts;
  char    	fname[80];

  init(&argc, argv, &ctrl, &wrk);
  if (PCG) newton_method_pcg();
  if (CG) newton_method_cg();

  convert_constants(&ctrl, &std_consts);

  /* Write Spectrum to file */
  sprintf(fname, "%s/HL%025.21Qf.spc", ctrl.libpath, wrk.hmax);
  printf("Data written to %s\n", fname);
  writer_out(wrk.YYk, fname, wrk.nl_error, std_consts.y0, wrk.hmax, fabsq(wrk.YYk[ctrl.n/2]), ctrl.c, std_consts.c, ctrl.w, ctrl.L);
  writer_out(wrk.YYk, ctrl.resname, wrk.nl_error, std_consts.y0, wrk.hmax, fabsq(wrk.YYk[ctrl.n/2]), ctrl.c, std_consts.c, ctrl.w, ctrl.L);

  sprintf(fname, "%s/HL%025.21Qf.txt", ctrl.libpath, wrk.hmax);
  printf("Data written to %s\n", fname);
  write_surface(&wrk, fname, std_consts.c, std_consts.y0, ctrl.w);
  
  //printf("Mean Level is %24.16Qe and also is %24.16Qe\n", mean_level(wrk.YYk), std_consts.y0);
  printf("H/Lambda = \t%29.21Qe\nc = \t\t%29.21Qe\nomega = \t%24.16Qe\n", wrk.hmax, std_consts.c, ctrl.w);
  printf("Fourier Coefficient at max wavenumber %d is %.12Qe\n", ctrl.n/2, fabsq(wrk.YYk[ctrl.n/2]));
 
  sprintf(fname, "%s/steep.txt", ctrl.logpath);
  FILE *fh = fopen(fname,"r"); 
  if (fh == NULL) {
    printf("Writing wave parameters to NEW steep.txt\n");
    fh = fopen(fname,"w");
    //fprintf(fh, "# 1. H/L 2. c 3. b0 4. Vorticity 5. Conformal Depth 6. Max Angle 7. x_max\n\n");
    fprintf(fh, "# 1. H/L 2. c 3. Vorticity 4. Max Angle 5. (x_max,y_max) \n\n");
    fclose(fh);
  } else {
    printf("Writing wave parameters to steep.txt\n");
    fclose(fh);
  }
  restore_map(&ctrl, &wrk, fname);
  seek_max_angle(&ctrl, &wrk);
  fh = fopen(fname, "a");
  fprintf(fh, "%29.22Qe\t", wrk.hmax);
  fprintf(fh, "%29.22Qe\t", std_consts.c);
  //fprintf(fh, "%29.22Qe\t", std_consts.b0);
  fprintf(fh, "%29.22Qe\t", ctrl.w);
  //fprintf(fh, "%s\t", "Inf");
  fprintf(fh, "%29.22Qe\t", wrk.max_angle);
  fprintf(fh, "%29.22Qe\t", wrk.xmax);
  fprintf(fh, "%29.22Qe\t", wrk.ymax);
  fprintf(fh, "\n");
  fclose(fh);

  if (ctrl.nthreads != 0) fftwq_cleanup_threads();
}
