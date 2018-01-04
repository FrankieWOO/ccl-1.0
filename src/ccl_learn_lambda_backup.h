#ifndef __CCL_LEARN_LAMBDA_H
#define __CCL_LEARN_LAMBDA_H

#include <../include/ccl_math.h>
#include <../include/ccl_learn_alpha.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define NUM_CONSTRAINT 2
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*JACOBIAN)(const double*,const int,double*);
void Jacobian(const double* X,const int size,double* out);
int ccl_learn_lambda_model_alloc(LEARN_A_MODEL *model);
int ccl_learn_lambda_model_free(LEARN_A_MODEL *model);
void ccl_learn_lambda(const double * Un,const double *X,void (*J_func)(const double*,const int,double*),const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal);
void search_learn_lambda(const double *BX, const double *RnVn, LEARN_A_MODEL* model);
void obj_AUn_lambda (const LEARN_A_MODEL* model, const double* W, const double* BX,const double * RnUn,double* fun_out);
void ccl_get_rotation_matrix_lambda(const double*theta,const double*currentRn,const LEARN_A_MODEL* model,const int alpha_id,double*Rn);
void ccl_solve_lm_lambda(const LEARN_A_MODEL* model,const  double* RnUn,const  double* BX, const OPTION option,SOLVE_LM_WS * lm_ws_param, double* W);
void findjac_lambda(const LEARN_A_MODEL* model, const int dim_x,const double* BX, const double * RnUn,const double *y,const double*x,double epsx,double* J);
int ccl_solve_lm_ws_lambda_alloc(const LEARN_A_MODEL *model,SOLVE_LM_WS * lm_ws);
void predict_proj_lambda(double* x, LEARN_A_MODEL model,void (*J_func)(const double*,const int,double*),double* centres,double variance,double* Iu, double*A);
#ifdef __cplusplus
}
#endif
#endif


