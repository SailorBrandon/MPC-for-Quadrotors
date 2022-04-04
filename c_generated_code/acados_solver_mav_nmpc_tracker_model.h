/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_mav_nmpc_tracker_model_H_
#define ACADOS_SOLVER_mav_nmpc_tracker_model_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define MAV_NMPC_TRACKER_MODEL_NX     12
#define MAV_NMPC_TRACKER_MODEL_NZ     0
#define MAV_NMPC_TRACKER_MODEL_NU     4
#define MAV_NMPC_TRACKER_MODEL_NP     0
#define MAV_NMPC_TRACKER_MODEL_NBX    0
#define MAV_NMPC_TRACKER_MODEL_NBX0   12
#define MAV_NMPC_TRACKER_MODEL_NBU    4
#define MAV_NMPC_TRACKER_MODEL_NSBX   0
#define MAV_NMPC_TRACKER_MODEL_NSBU   0
#define MAV_NMPC_TRACKER_MODEL_NSH    1
#define MAV_NMPC_TRACKER_MODEL_NSG    0
#define MAV_NMPC_TRACKER_MODEL_NSPHI  0
#define MAV_NMPC_TRACKER_MODEL_NSHN   0
#define MAV_NMPC_TRACKER_MODEL_NSGN   0
#define MAV_NMPC_TRACKER_MODEL_NSPHIN 0
#define MAV_NMPC_TRACKER_MODEL_NSBXN  0
#define MAV_NMPC_TRACKER_MODEL_NS     1
#define MAV_NMPC_TRACKER_MODEL_NSN    0
#define MAV_NMPC_TRACKER_MODEL_NG     0
#define MAV_NMPC_TRACKER_MODEL_NBXN   0
#define MAV_NMPC_TRACKER_MODEL_NGN    0
#define MAV_NMPC_TRACKER_MODEL_NY0    16
#define MAV_NMPC_TRACKER_MODEL_NY     16
#define MAV_NMPC_TRACKER_MODEL_NYN    6
#define MAV_NMPC_TRACKER_MODEL_N      20
#define MAV_NMPC_TRACKER_MODEL_NH     1
#define MAV_NMPC_TRACKER_MODEL_NPHI   0
#define MAV_NMPC_TRACKER_MODEL_NHN    0
#define MAV_NMPC_TRACKER_MODEL_NPHIN  0
#define MAV_NMPC_TRACKER_MODEL_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct mav_nmpc_tracker_model_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;
    external_function_param_casadi *nl_constr_h_fun_jac_hess;




} mav_nmpc_tracker_model_solver_capsule;

ACADOS_SYMBOL_EXPORT mav_nmpc_tracker_model_solver_capsule * mav_nmpc_tracker_model_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_free_capsule(mav_nmpc_tracker_model_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_create(mav_nmpc_tracker_model_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_reset(mav_nmpc_tracker_model_solver_capsule* capsule);

/**
 * Generic version of mav_nmpc_tracker_model_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_create_with_discretization(mav_nmpc_tracker_model_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_update_time_steps(mav_nmpc_tracker_model_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_update_qp_solver_cond_N(mav_nmpc_tracker_model_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_update_params(mav_nmpc_tracker_model_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_solve(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int mav_nmpc_tracker_model_acados_free(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void mav_nmpc_tracker_model_acados_print_stats(mav_nmpc_tracker_model_solver_capsule * capsule);
                     
ACADOS_SYMBOL_EXPORT ocp_nlp_in *mav_nmpc_tracker_model_acados_get_nlp_in(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *mav_nmpc_tracker_model_acados_get_nlp_out(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *mav_nmpc_tracker_model_acados_get_sens_out(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *mav_nmpc_tracker_model_acados_get_nlp_solver(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *mav_nmpc_tracker_model_acados_get_nlp_config(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *mav_nmpc_tracker_model_acados_get_nlp_opts(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *mav_nmpc_tracker_model_acados_get_nlp_dims(mav_nmpc_tracker_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *mav_nmpc_tracker_model_acados_get_nlp_plan(mav_nmpc_tracker_model_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_mav_nmpc_tracker_model_H_
