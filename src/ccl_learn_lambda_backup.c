#include <ccl_learn_lambda.h>
void Jacobian(const double* X, const int size, double *out){
//    out[0] = 2*X[0];
//    out[1] = 2*X[1];
//    out[2] = 0;
//    out[3] = 1;
//    out[4] = 2*X[0];
//    out[5] = -1;
    out[0] = -sin(X[0])-sin(X[0]+X[1]);
    out[1] = -sin(X[0]+X[1]);
    out[2] = cos(X[0])+cos(X[0]+X[1]);
    out[3] = cos(X[0]+X[1]);
}

int ccl_learn_lambda_model_alloc(LEARN_A_MODEL *model){
    int i;
    for (i=0;i<NUM_CONSTRAINT;i++){
        model->w[i]  = malloc(model->dim_r*model->dim_b*sizeof(double*));
    }
}
int ccl_learn_lambda_model_free(LEARN_A_MODEL *model){
    int k;
    for (k=0;k<NUM_CONSTRAINT;k++){
        free(model->w[k]);
    }
}
void ccl_learn_lambda(const double * Un,const double *X,void (*J_func)(const double*,const int,double*),const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal){
    LEARN_A_MODEL model;
    double * centres,*var_tmp,*vec, *BX, *RnVn,*Rn,*Vn;
    double variance;
    int i,lambda_id;
    gsl_matrix *Un_,*X_;

    //     J_fun       = &Jacobian;
    model.dim_b = dim_b;
    model.dim_r = dim_r;
    model.dim_x = dim_x;
    model.dim_n = dim_n;
    model.dim_t = dim_r-1;
    model.dim_u = dim_u;
    // calculate model.var
    var_tmp = malloc(dim_u*sizeof(double));
    ccl_mat_var(Un,dim_u,dim_n,0,var_tmp);
    model.var = ccl_vec_sum(var_tmp,dim_u);
    free(var_tmp);

    Un_     = gsl_matrix_alloc(dim_u,dim_n);
    memcpy(Un_->data,Un,dim_u*dim_n*sizeof(double));
    X_ = gsl_matrix_alloc(dim_x,dim_n);
    memcpy(X_->data,X,dim_x*dim_n*sizeof(double));


    optimal.nmse = 1000000;

    gsl_vector* X_col = gsl_vector_alloc(dim_x);
    gsl_vector* Un_col = gsl_vector_alloc(dim_u);
    gsl_matrix* Vn_container = gsl_matrix_alloc(dim_r,dim_n);
    gsl_matrix_set_zero(Vn_container);
    double*     norm   = malloc(dim_n*sizeof(double));
    double*     J_x    = malloc(dim_r*dim_x*sizeof(double));
    gsl_vector* Jx_Un  = gsl_vector_alloc(dim_r);
    int*     id_keep   = malloc(dim_n*sizeof(double));
    for (i=0;i<dim_n;i++){
        gsl_matrix_get_col(X_col,X_,i);
        gsl_matrix_get_col(Un_col,Un_,i);
        J_func(X_col->data,dim_x,J_x);
        //        print_mat_d(J_x,dim_r,dim_x);
        ccl_dot_product(J_x,dim_r,dim_x,Un_col->data,dim_u,1,Jx_Un->data);
        gsl_matrix_set_col(Vn_container,i,Jx_Un);
        norm[i] = gsl_blas_dnrm2(Jx_Un);
    }
    // here dim_n maybe change.
    int new_dim_n = ccl_find_index_double(norm,dim_n,2,1E-3,id_keep);
    model.dim_n = new_dim_n;


    Vn = malloc(dim_r*model.dim_n*sizeof(double));
    gsl_matrix Vn_ = gsl_matrix_view_array(Vn,dim_r,model.dim_n).matrix;
    gsl_matrix*X_new = gsl_matrix_alloc(dim_x,model.dim_n);
    gsl_matrix*Un_new = gsl_matrix_alloc(dim_u,model.dim_n);
    for (i=0;i<new_dim_n;i++){
        gsl_vector* Vn_tmp = gsl_vector_alloc(dim_r);
        gsl_matrix_get_col(Vn_tmp,Vn_container,id_keep[i]);
        gsl_matrix_set_col(&Vn_,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
        Vn_tmp = gsl_vector_alloc(dim_x);
        gsl_matrix_get_col(Vn_tmp,X_,id_keep[i]);
        gsl_matrix_set_col(X_new,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
        Vn_tmp = gsl_vector_alloc(dim_u);
        gsl_matrix_get_col(Vn_tmp,Un_,id_keep[i]);
        gsl_matrix_set_col(Un_new,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
    }
    gsl_vector_free(Jx_Un);
    gsl_vector_free(X_col);
    gsl_vector_free(Un_col);
    gsl_matrix_free(Vn_container);
    gsl_matrix_free(Un_);
    free(J_x);
    free(norm);
    free(id_keep);

    //  prepare for BX
    centres = malloc(dim_x*dim_b*sizeof(double));
    generate_kmeans_centres(X_new->data,dim_x,model.dim_n,dim_b,centres);
    var_tmp = malloc(dim_b*dim_b*sizeof(double));
    vec     = malloc(dim_b*sizeof(double));
    ccl_mat_distance(centres,dim_x,dim_b,centres,dim_x,dim_b,var_tmp);

    for (i=0;i<dim_b*dim_b;i++){
        var_tmp[i] = sqrt(var_tmp[i]);
    }
    ccl_mat_mean(var_tmp,dim_b,dim_b,1,vec);
    variance = pow(gsl_stats_mean(vec,1,dim_b),2);
    BX = malloc(dim_b*model.dim_n*sizeof(double));
    ccl_gaussian_rbf(X_new->data,dim_x,model.dim_n,centres,dim_x,dim_b,variance,BX);
    gsl_matrix_free(X_new);
    gsl_matrix_free(Un_new);
    free(var_tmp);
    free(vec);

    Rn = malloc(dim_r*dim_r*sizeof(double));
    gsl_matrix Rn_ = gsl_matrix_view_array(Rn,dim_r,dim_r).matrix;
    gsl_matrix_set_identity(&Rn_);

    RnVn = malloc(dim_r*model.dim_n*sizeof(double));
    memcpy(RnVn,Vn,dim_r*model.dim_n*sizeof(double));
    ccl_learn_alpha_model_alloc(&model);


    for(lambda_id=0;lambda_id<dim_r;lambda_id++){
        model.dim_k = lambda_id+1;
        //        model.w[lambda_id] = malloc((dim_u-model.dim_k)*dim_b*sizeof(double*));
        if(dim_r-model.dim_k==0){
            model.dim_k = lambda_id;
            break;
        }
        else{
            search_learn_alpha(BX,RnVn,&model);
            double* theta = malloc(model.dim_n*model.dim_t*sizeof(double));
            double *W_BX = malloc((dim_u-model.dim_k)*dim_n*sizeof(double));
            double *W_BX_T = malloc(model.dim_n*(dim_u-model.dim_k)*sizeof(double));
            ccl_dot_product(model.w[lambda_id],dim_u-model.dim_k,dim_b,BX,dim_b,model.dim_n,W_BX);
            ccl_mat_transpose(W_BX,dim_u-model.dim_k,dim_n,W_BX_T);
            if (model.dim_k ==1){
                memcpy(theta,W_BX_T,model.dim_n*(dim_u-model.dim_k)*sizeof(double));
            }
            else{
                gsl_matrix* ones = gsl_matrix_alloc(model.dim_n,model.dim_k-1);
                gsl_matrix_set_all(ones,1);
                gsl_matrix_scale(ones,M_PI/2);
                mat_hotz_app(ones->data,model.dim_n,model.dim_k-1,W_BX_T,model.dim_n,dim_u-model.dim_k,theta);

                gsl_matrix_free(ones);
            }
            for(i=0;i<model.dim_n;i++){
                gsl_matrix theta_mat = gsl_matrix_view_array(theta,model.dim_n,model.dim_t).matrix;
                gsl_vector *vec      = gsl_vector_alloc(model.dim_t);
                gsl_matrix_get_row(vec,&theta_mat,i);
                ccl_get_rotation_matrix_lambda(vec->data,Rn,&model,lambda_id,Rn);
                gsl_vector_free(vec);
                vec                  = gsl_vector_alloc(dim_r);
                gsl_matrix_get_col(vec,&Vn_,i);
                ccl_dot_product(Rn,dim_r,dim_r,vec->data,dim_r,1,vec->data);
                gsl_matrix RnVn_     = gsl_matrix_view_array(RnVn,dim_r,model.dim_n).matrix;
                gsl_matrix_set_col(&RnVn_,i,vec);
                gsl_vector_free(vec);
            }
            if(model.nmse > optimal.nmse && model.nmse > 1E-5){
                model.dim_k = lambda_id;
                printf("\n I am out...\n");//optimal;
                break;
            }
            else{
                printf("\n copy model -> optimal\n");//optimal;
            }
            free(W_BX);
            free(W_BX_T);
            free(theta);
        }
    }
    double* A = malloc(model.dim_k*model.dim_r*sizeof(double));
    gsl_matrix* Iu = gsl_matrix_alloc(model.dim_r,model.dim_r);
    gsl_matrix_set_identity(Iu);
    gsl_vector* x = gsl_vector_alloc(model.dim_x);
    gsl_matrix_get_col(x,X_,5);
    predict_proj_lambda(x->data, model,Jacobian,centres,variance,Iu->data,A);
    print_mat_d(A,model.dim_k,dim_r);
    free(Vn);
    free(Rn);
    free(BX);
    free(RnVn);
    free(centres);
    gsl_matrix_free(X_);
    ccl_learn_alpha_model_free(&model);
}
void search_learn_lambda(const double* BX, const double* RnVn,LEARN_A_MODEL* model){
    OPTION option;
    option.MaxIter = 1000;
    option.Tolfun  = 1E-6;
    option.Tolx    = 1E-6;
    option.Jacob   = 0;
    model->nmse = 100000;
    int i,j,n_param;
    SOLVE_LM_WS lm_ws_param;
    gsl_rng * r;
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r,1);
    n_param = (model->dim_r-model->dim_k)*model->dim_b;
    for (i=0;i<5;i++){
        gsl_vector * nmse_ = gsl_vector_alloc(model->dim_n);
        gsl_vector * W = gsl_vector_alloc((model->dim_r-model->dim_k)*model->dim_b);
        for (j=0;j< W->size;j++){
            gsl_vector_set(W,j,gsl_rng_uniform(r));
        }
        ccl_solve_lm_lambda(model,RnVn,BX,option,&lm_ws_param,W->data);
        obj_AUn_lambda(model,W->data,BX,RnVn,nmse_->data);
        print_mat_d(W->data,model->dim_n,1);
        gsl_vector_mul(nmse_,nmse_);
        double nmse = ccl_vec_sum(nmse_->data,model->dim_n)/model->dim_n/model->var;
        printf("K=%d, iteration=%d, residual error=%4.2g\n", model->dim_k, i, nmse);
        if(model->nmse > nmse){
            ccl_mat_reshape(W->data,1,(model->dim_r-model->dim_k)*model->dim_b, model->w[model->dim_k-1]);
            model->nmse = nmse;
        }
        if (model->nmse < 1E-5){
            gsl_vector_free(W);
            gsl_vector_free(nmse_);
            break;
        }
        gsl_vector_free(W);
        gsl_vector_free(nmse_);
    }
    gsl_rng_free(r);
}

void obj_AUn_lambda (const LEARN_A_MODEL* model,  const double* W, const double* BX, const double * RnVn,double* fun_out){
    int dim_n,dim_x,dim_b,dim_u,dim_k,dim_t,dim_r,n;
    double * theta,*alpha,*W_BX,*W_BX_T,*W_;
    dim_n = model->dim_n;
    dim_x = model->dim_x;
    dim_b = model->dim_b;
    dim_u = model->dim_u;
    dim_k = model->dim_k;
    dim_t = model->dim_t;
    dim_r = model->dim_r;
    theta = malloc(dim_n*(dim_k-1+dim_r-dim_k)*sizeof(double));
    alpha = malloc(dim_n*dim_r*sizeof(double));
    W_   = malloc((dim_r-dim_k)*dim_b*sizeof(double));
    W_BX = malloc((dim_r-dim_k)*dim_n*sizeof(double));
    W_BX_T = malloc(dim_n*(dim_r-dim_k)*sizeof(double));
    ccl_mat_reshape(W,dim_r-dim_k,dim_b,W_);
    ccl_dot_product(W_,dim_r-dim_k,dim_b,BX,dim_b,dim_n,W_BX);
    ccl_mat_transpose(W_BX,dim_r-dim_k,dim_n,W_BX_T);
    if (dim_k ==1){
        memcpy(theta,W_BX_T,dim_n*(dim_r-dim_k)*sizeof(double));
    }
    else{
        gsl_matrix* ones = gsl_matrix_alloc(dim_n,dim_k-1);
        gsl_matrix_set_all(ones,1);
        gsl_matrix_scale(ones,M_PI/2);
        mat_hotz_app(ones->data,dim_n,dim_k-1,W_BX_T,dim_n,dim_r-dim_k,theta);
        gsl_matrix_free(ones);
    }
    ccl_get_unit_vector_from_matrix(theta,dim_n,dim_t,alpha);
    gsl_matrix a_m = gsl_matrix_view_array(alpha,dim_n,dim_r).matrix;
    gsl_matrix * RnVn_m = gsl_matrix_alloc(dim_r,dim_n);
    memcpy(RnVn_m->data,RnVn,dim_r*dim_n*sizeof(double));
    gsl_vector *a_v = gsl_vector_alloc(dim_r);
    gsl_vector *RnVn_v = gsl_vector_alloc(dim_r);
    for (n=0;n<dim_n;n++){
        gsl_matrix_get_row(a_v,&a_m,n);
        gsl_matrix_get_col(RnVn_v,RnVn_m,n);
        ccl_dot_product(a_v->data,1,dim_r,RnVn_v->data,dim_r,1,&fun_out[n]);
    }
    gsl_matrix_free(RnVn_m);
    gsl_vector_free(a_v);
    gsl_vector_free(RnVn_v);
    free(theta);
    free(alpha);
    free(W_BX);
    free(W_BX_T);
    free(W_);
}
void ccl_get_rotation_matrix_lambda(const double*theta,const double*currentRn,const LEARN_A_MODEL* model,const int alpha_id,double*Rn){
    gsl_matrix * R = gsl_matrix_alloc(model->dim_r,model->dim_r);
    gsl_matrix_set_identity(R);
    int d;
    for(d=alpha_id;d < model->dim_t;d++){
        gsl_matrix * tmp = gsl_matrix_alloc(model->dim_r,model->dim_r);
        ccl_make_given_matrix(theta[d],d,d+1,model->dim_r,tmp->data);
        ccl_dot_product(R->data,model->dim_r,model->dim_r,tmp->data,model->dim_r,model->dim_r,R->data);
        gsl_matrix_free(tmp);
    }
    ccl_dot_product(R->data,model->dim_r,model->dim_r,currentRn,model->dim_r,model->dim_r,Rn);
    gsl_matrix_free(R);
}
void ccl_solve_lm_lambda(const LEARN_A_MODEL* model,const  double* RnUn,const  double* BX, const OPTION option,SOLVE_LM_WS * lm_ws_param, double* W){
    ccl_solve_lm_ws_lambda_alloc(model,lm_ws_param);
    memcpy(lm_ws_param->xc,W,lm_ws_param->dim_x*sizeof(double));
    flt_mat(lm_ws_param->xc,lm_ws_param->dim_x,1,lm_ws_param->x);
    gsl_vector ones = gsl_vector_view_array(lm_ws_param->epsx,lm_ws_param->dim_x).vector;
    gsl_vector_set_all(&ones,1);
    gsl_vector_scale(&ones,option.Tolx);
    lm_ws_param->epsf = option.Tolfun;
    obj_AUn_lambda(model,lm_ws_param->x,BX,RnUn,lm_ws_param->r);
    //print_mat_d(lm_ws_param->r,lm_ws_param->dim_n,1);
    int dim = (model->dim_r-model->dim_k)*model->dim_b;
    findjac_lambda(model,dim,BX,RnUn,lm_ws_param->r,lm_ws_param->x,lm_ws_param->epsx[0],lm_ws_param->J);
    gsl_vector* r_T = gsl_vector_alloc(lm_ws_param->dim_n);
    memcpy(r_T->data,lm_ws_param->r,lm_ws_param->dim_n*sizeof(double));
    ccl_mat_transpose(lm_ws_param->r,lm_ws_param->dim_n,1,r_T->data);
    ccl_dot_product(r_T->data,1,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,&lm_ws_param->S);
    gsl_vector_free(r_T);
    double* J_T = malloc(lm_ws_param->dim_n*lm_ws_param->dim_x*sizeof(double));
    memcpy(J_T,lm_ws_param->J,lm_ws_param->dim_x*lm_ws_param->dim_n*sizeof(double));
    ccl_mat_transpose(lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,J_T);
    ccl_dot_product(J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->A);
    ccl_dot_product(J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,lm_ws_param->v);
    gsl_matrix A_ = gsl_matrix_view_array(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x).matrix;
    gsl_vector A_diag = gsl_matrix_diagonal(&A_).vector;
    int i,j,k;
    gsl_matrix D_ = gsl_matrix_view_array(lm_ws_param->D,lm_ws_param->dim_x,lm_ws_param->dim_x).matrix;
    gsl_matrix_set_zero(&D_);
    k=0;
    for (i = 0;i<lm_ws_param->dim_x;i++){
        for (j=0;j<lm_ws_param->dim_x;j++){
            gsl_matrix_set(&D_,i,j,A_diag.data[k]);
            k++;
        }
    }
    for (i=0;i<lm_ws_param->dim_x;i++){
        if (gsl_matrix_get(&D_,i,i)==0){
            gsl_matrix_set(&D_,i,i,1);
        }
    }
    lm_ws_param->Rlo = 0.25;
    lm_ws_param->Rhi = 0.75;
    lm_ws_param->l   = 1;
    lm_ws_param->lc  = 0.75;
    // Main iterations
    lm_ws_param->iter = 0;
    double * d_T = malloc(lm_ws_param->dim_x*sizeof(double));
    gsl_vector d = gsl_vector_view_array(lm_ws_param->d,lm_ws_param->dim_x).vector;
    gsl_vector_set_all(&d,option.Tolx);
    while (lm_ws_param->iter<option.MaxIter
           && ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x)
           && ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n)){
        gsl_matrix * D_pinv   = gsl_matrix_alloc(lm_ws_param->dim_x,lm_ws_param->dim_x);
        gsl_matrix_memcpy(D_pinv,&D_);
        gsl_matrix_scale(D_pinv,lm_ws_param->l);
        ccl_mat_add(D_pinv->data,lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x);
        ccl_MP_pinv(D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,D_pinv->data);
        ccl_dot_product(D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->v,lm_ws_param->dim_x,1,lm_ws_param->d);
        gsl_matrix_free(D_pinv);
        memcpy(lm_ws_param->xd,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
        ccl_mat_sub(lm_ws_param->xd,lm_ws_param->d,lm_ws_param->dim_x,1);
        obj_AUn_lambda(model,lm_ws_param->xd,BX,RnUn,lm_ws_param->rd);
        int dim = (model->dim_r - model->dim_k)*model->dim_b;
        findjac_lambda(model,dim,BX,RnUn,lm_ws_param->r,lm_ws_param->x,lm_ws_param->epsx[0],lm_ws_param->J);
        //print_mat_d(lm_ws_param->J,23,3);
        double * rd_T = malloc(lm_ws_param->dim_n*sizeof(double));
        memcpy(rd_T,lm_ws_param->rd,lm_ws_param->dim_n*sizeof(double));
        ccl_mat_transpose(lm_ws_param->rd,lm_ws_param->dim_n,1,rd_T);
        ccl_dot_product(rd_T,1,lm_ws_param->dim_n,lm_ws_param->rd,lm_ws_param->dim_n,1,&lm_ws_param->Sd);
        double * tmp = malloc(lm_ws_param->dim_x*sizeof(double));
        memcpy(tmp,lm_ws_param->v,lm_ws_param->dim_x*sizeof(double));
        gsl_vector tmp_vec = gsl_vector_view_array(tmp,lm_ws_param->dim_x).vector;
        gsl_vector_scale(&tmp_vec,2);
        gsl_vector * A_d = gsl_vector_alloc(lm_ws_param->dim_x);
        ccl_dot_product(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->d,lm_ws_param->dim_x,1,A_d->data);
        ccl_mat_sub(tmp_vec.data,A_d->data,lm_ws_param->dim_x,1);
        memcpy(d_T,lm_ws_param->d,lm_ws_param->dim_x);
        ccl_mat_transpose(lm_ws_param->d,lm_ws_param->dim_x,1,d_T);
        ccl_dot_product(d_T,1,lm_ws_param->dim_x,tmp_vec.data,lm_ws_param->dim_x,1,&lm_ws_param->dS);
        lm_ws_param->R = (lm_ws_param->S - lm_ws_param->dS)/lm_ws_param->dS;
        free(tmp);
        free(rd_T);
        gsl_vector_free(A_d);

        if(lm_ws_param->R>lm_ws_param->Rhi){
            lm_ws_param->l = lm_ws_param->l/2;
            if(lm_ws_param->l < lm_ws_param->lc){
                lm_ws_param->l = 0;
            }
        }
        else if(lm_ws_param->R < lm_ws_param->Rlo){
            double d_Tv[] = {0};
            ccl_dot_product(d_T,1,lm_ws_param->dim_x,lm_ws_param->v,lm_ws_param->dim_x,1,d_Tv);
            lm_ws_param->nu = (lm_ws_param->Sd-lm_ws_param->S)/d_Tv[0]+2;
            if (lm_ws_param->nu < 2){
                lm_ws_param->nu = 2;
            }
            else if (lm_ws_param->nu > 10){
                lm_ws_param->nu = 10;
            }
            if (lm_ws_param->l == 0){
                gsl_matrix * A_inv = gsl_matrix_alloc(lm_ws_param->dim_x,lm_ws_param->dim_x);
                memcpy(A_inv->data,lm_ws_param->A,lm_ws_param->dim_x*lm_ws_param->dim_x*sizeof(double));
                gsl_vector * A_inv_diag = gsl_vector_alloc(lm_ws_param->dim_x);
                ccl_MP_pinv(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x,A_inv->data);
                gsl_vector vec_= gsl_matrix_diagonal(A_inv).vector;
                gsl_vector_memcpy(A_inv_diag,&vec_);
                for (i=0;i<lm_ws_param->dim_x;i++){
                    gsl_vector_set(A_inv_diag,i,fabs(A_inv_diag->data[i]));
                }
                lm_ws_param->lc = 1/gsl_vector_max(A_inv_diag);
                lm_ws_param->l = lm_ws_param->lc;
                lm_ws_param->nu = lm_ws_param->nu/2;
                gsl_vector_free(A_inv_diag);
                free(A_inv);
            }
            lm_ws_param->l = lm_ws_param->nu*lm_ws_param->l;
        }
        lm_ws_param->iter ++;
        if(lm_ws_param->Sd < lm_ws_param->S){
            lm_ws_param->S = lm_ws_param->Sd;
            memcpy(lm_ws_param->x,lm_ws_param->xd,lm_ws_param->dim_x*sizeof(double));
            memcpy(lm_ws_param->r,lm_ws_param->rd,lm_ws_param->dim_n*sizeof(double));
            ccl_dot_product(J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->A);
            ccl_dot_product(J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,lm_ws_param->v);
        }
    }
    memcpy(lm_ws_param->xf,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
    memcpy(W,lm_ws_param->xf,lm_ws_param->dim_x*sizeof(double));
    if(lm_ws_param->iter == option.MaxIter) printf("Solver terminated because max iteration reached\n");
    else if (ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x)) printf("Solver terminated because |dW| < min(dW)\n");
    else if (ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n)) printf("Solver terminated because |F(dW)| < min(F(dW))\n");
    else printf("Problem solved\n");
    free(d_T);
    free(J_T);
    ccl_solve_lm_ws_free(lm_ws_param);
}
void findjac_lambda(const LEARN_A_MODEL* model, const int dim_x,const double* BX, const double * RnUn,const double *y,const double*x,double epsx,double* J){
    gsl_matrix J_ = gsl_matrix_view_array(J,model->dim_n,dim_x).matrix;
    gsl_matrix_set_zero(&J_);
    int k;
    double dx;
    dx = epsx*0.25;
    gsl_vector  *y_  = gsl_vector_alloc(model->dim_n);
    memcpy(y_->data,y,model->dim_n*sizeof(double));
    for (k=0;k<dim_x;k++){
        gsl_vector *yd = gsl_vector_alloc(model->dim_n);
        gsl_vector *xd = gsl_vector_alloc(dim_x);
        memcpy(xd->data,x,dim_x*sizeof(double));
        gsl_vector_set(xd,k,gsl_vector_get(xd,k)+dx);
        obj_AUn_lambda(model,xd->data,BX,RnUn,yd->data);
        gsl_vector_sub(yd,y_);
        gsl_vector_scale(yd,1/dx);
        gsl_matrix_set_col(&J_,k,yd);
        gsl_vector_free(xd);
        gsl_vector_free(yd);
    }
    gsl_vector_free(y_);
}
int ccl_solve_lm_ws_lambda_alloc(const LEARN_A_MODEL *model,SOLVE_LM_WS * lm_ws){
    int num_w_param;
    num_w_param = (model->dim_r-model->dim_k)*model->dim_b;
    lm_ws->A = malloc(num_w_param*num_w_param*sizeof(double));
    lm_ws->D = malloc(num_w_param*num_w_param*sizeof(double));
    lm_ws->epsx = malloc(num_w_param*sizeof(double));
    lm_ws->J = malloc(model->dim_n*num_w_param*sizeof(double));
    lm_ws->r = malloc(model->dim_n*sizeof(double));
    lm_ws->rd = malloc(model->dim_n*sizeof(double));
    lm_ws->v = malloc(num_w_param*sizeof(double));
    lm_ws->x = malloc(num_w_param*sizeof(double));
    lm_ws->xc= malloc(num_w_param*sizeof(double));
    lm_ws->xd= malloc(num_w_param*sizeof(double));
    lm_ws->xf= malloc(num_w_param*sizeof(double));
    lm_ws->d= malloc(num_w_param*sizeof(double));
    lm_ws->dim_b = model->dim_b;
    lm_ws->dim_u = model->dim_u;
    lm_ws->dim_n = model->dim_n;
    lm_ws->dim_x = num_w_param;
    lm_ws->dim_k = model->dim_k;
}
void predict_proj_lambda(double* x, LEARN_A_MODEL model,void (*J_func)(const double*,const int,double*),double* centres,double variance,double* Iu, double*A){
    gsl_matrix* Rn = gsl_matrix_alloc(model.dim_r,model.dim_r);
    memcpy(Rn->data,Iu,model.dim_r*model.dim_r*sizeof(double));
    gsl_matrix* lambda = gsl_matrix_alloc(model.dim_k,model.dim_r);
    gsl_matrix_set_all(lambda,0);
    int k;
    double * BX, *W_BX,*W_BX_T,*theta,*alpha,*J_x;
    BX = malloc(model.dim_b*1*sizeof(double));
    theta = malloc(1*model.dim_t*sizeof(double));
    alpha = malloc(1*model.dim_r*sizeof(double));
    gsl_vector* lambda_vec = gsl_vector_alloc(model.dim_r);
    for (k=1;k<model.dim_k+1;k++){
        W_BX = malloc((model.dim_r-k)*1*sizeof(double));
        W_BX_T = malloc(1*(model.dim_r-k)*sizeof(double));
        ccl_gaussian_rbf(x,model.dim_x,1,centres,model.dim_x,model.dim_b,variance,BX);
        ccl_dot_product(model.w[k-1],model.dim_r-k,model.dim_b,BX,model.dim_b,1,W_BX);
        ccl_mat_transpose(W_BX,model.dim_r-k,1,W_BX_T);
        free(W_BX);
        if (k ==1){
            memcpy(theta,W_BX_T,1*(model.dim_r-k)*sizeof(double));
        }
        else{
            gsl_matrix* ones = gsl_matrix_alloc(1,k);
            gsl_matrix_set_all(ones,1);
            gsl_matrix_scale(ones,M_PI/2);
            mat_hotz_app(ones->data,1,k,W_BX_T,model.dim_n,model.dim_r-k,theta);
            free(W_BX_T);
            gsl_matrix_free(ones);
        }
        ccl_get_unit_vector_from_matrix(theta,1,model.dim_t,alpha);
        ccl_dot_product(alpha,k,model.dim_r,Rn->data,model.dim_r,model.dim_r,lambda_vec->data);
        gsl_matrix_set_row(lambda,k-1,lambda_vec);
        ccl_get_rotation_matrix_lambda(theta,Rn->data,&model,k-1,Rn->data);
    }
    memcpy(A,lambda->data,model.dim_k*model.dim_r*sizeof(double));
    J_x = malloc(model.dim_r*model.dim_x*sizeof(double));
//    J_func(x,model.dim_x,J_x);
//    print_mat_d(alpha,1,model.dim_r);
//    ccl_dot_product(lambda->data,model.dim_k,model.dim_r,J_x,model.dim_r,model.dim_x,A);
    free(BX);
    free(theta);
    free(alpha);
    free(J_x);
    gsl_vector_free(lambda_vec);
    gsl_matrix_free(lambda);
}

