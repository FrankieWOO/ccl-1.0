AC_INIT([ccl], [1.0], [yuchen.zhao@kcl.ac.uk])
AC_CONFIG_AUX_DIR([buid-aux])
AC_CONFIG_HEADERS([include/ccl_config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_PROG_AR
LT_INIT

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

AC_C_CONST
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_TYPE_SIZE_T

AC_HEADER_STDBOOL
AC_HEADER_STDC
AC_STRUCT_TM

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile])
AC_OUTPUT
