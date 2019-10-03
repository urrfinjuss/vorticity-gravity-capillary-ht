AC_INIT([travelling-wave-ht], [1.0], [urrfinjuss@gmail.com], [generic-fluid-ht], [])

AM_INIT_AUTOMAKE([-Werror -Wall foreign subdir-objects])
#: ${CFLAGS="-Ofast"}
#AM_SILENT_RULES([yes])
AC_PROG_CC
AC_CHECK_LIB([fftw3q],[fftwq_malloc],[],[AC_MSG_ERROR([FFTW __float128 not installed])])
AC_CHECK_LIB([fftw3q_threads],[fftwq_init_threads],[],[AC_MSG_ERROR([FFTW __float128 threads not installed])])
AC_CHECK_LIB([mpfr],[mpfr_inits2],[],[AC_MSG_ERROR([GNU MPFR not installed])])
AC_CHECK_LIB([mpfr],[mpfr_get_float128],[],[AC_MSG_ERROR([GNU MPFR does not support __float128 type])])
AC_CHECK_LIB([mpc],[mpc_init2],[],[AC_MSG_ERROR([GNU MPC not installed])])
AC_CHECK_LIB([quadmath],[floorq],[],[AC_MSG_ERROR([libquadmath missing])])
AC_CHECK_LIB([m],[pow],[],[AC_MSG_ERROR([Could not find pow])])
AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([
src/Makefile
Makefile
])

# enable debugging
AC_ARG_ENABLE([debug], 
  [AS_HELP_STRING([--enable-debug], [enable debug messages])],
[ AC_DEFINE([DEBUG], [1], [Enable extra debugging information]) ], 
[ AC_DEFINE([DEBUG], [0], [Disable extra debugging information]) ] 
)



AC_OUTPUT
