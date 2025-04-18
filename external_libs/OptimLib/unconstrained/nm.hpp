/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
  ##   Modified work Copyright (c) 2021-22 Fabio Sigrist. All rights reserved.
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

//ChangedForGPBoost (removed all grad_out parameter names to avoid "unreferenced formal paramter" warning and removed all output to std::cout for CRAN compliance)

/*
 * Nelder-Mead
 */

#ifndef _optim_nm_HPP
#define _optim_nm_HPP

/**
 * @brief The Nelder-Mead Simplex-based Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
nm(Vec_t& init_out_vals, 
   std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
   void* opt_data);

/**
 * @brief The Nelder-Mead Simplex-based Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
nm(Vec_t& init_out_vals, 
   std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
   void* opt_data, 
   algo_settings_t& settings);

//
// internal

namespace internal
{

bool
nm_impl(Vec_t& init_out_vals, 
        std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
        void* opt_data, 
        algo_settings_t* settings_inp);

}

//

inline
bool
internal::nm_impl(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    //
    // NM settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    //ChangedForGPBoost
    //const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const double rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const double rel_sol_change_tol = settings.rel_sol_change_tol;

    // expansion / contraction parameters
    
    const double par_alpha = settings.nm_settings.par_alpha;
    const double par_beta  = (settings.nm_settings.adaptive_pars) ? 0.75 - 1.0 / (2.0*n_vals) : settings.nm_settings.par_beta;
    const double par_gamma = (settings.nm_settings.adaptive_pars) ? 1.0 + 2.0 / n_vals        : settings.nm_settings.par_gamma;
    const double par_delta = (settings.nm_settings.adaptive_pars) ? 1.0 - 1.0 / n_vals        : settings.nm_settings.par_delta;

    const bool vals_bound = settings.vals_bound;
    
    const Vec_t lower_bounds = settings.lower_bounds;
    const Vec_t upper_bounds = settings.upper_bounds;

    const VecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints
    std::function<double (const Vec_t& vals_inp, Vec_t*, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const Vec_t& vals_inp, Vec_t*, void* opt_data) \
    -> double 
    {
        if (vals_bound) {
            Vec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        } else {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };
    
    //
    // setup

    Vec_t simplex_fn_vals(n_vals+1);
    Vec_t simplex_fn_vals_old(n_vals+1);
    Mat_t simplex_points(n_vals+1, n_vals);
    Mat_t simplex_points_old(n_vals+1, n_vals);
    
    simplex_fn_vals(0) = opt_objfn(init_out_vals, nullptr, opt_data);
    simplex_points.row(0) = OPTIM_MATOPS_TRANSPOSE(init_out_vals);

    if (vals_bound) {
        simplex_points.row(0) = OPTIM_MATOPS_TRANSPOSE( transform( OPTIM_MATOPS_TRANSPOSE(simplex_points.row(0)), bounds_type, lower_bounds, upper_bounds) );
    }

    for (size_t i = 1; i < n_vals + 1; ++i) {
        if (init_out_vals(i-1) != 0.0) {
            simplex_points.row(i) = OPTIM_MATOPS_TRANSPOSE( init_out_vals + 0.05*init_out_vals(i-1) * unit_vec(i-1,n_vals) );
        } else {
            simplex_points.row(i) = OPTIM_MATOPS_TRANSPOSE( init_out_vals + 0.00025 * unit_vec(i-1,n_vals) );
            // simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
        }

        simplex_fn_vals(i) = opt_objfn(OPTIM_MATOPS_TRANSPOSE(simplex_points.row(i)),nullptr,opt_data);

        if (vals_bound) {
            simplex_points.row(i) = OPTIM_MATOPS_TRANSPOSE( transform( OPTIM_MATOPS_TRANSPOSE(simplex_points.row(i)), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    double min_val = OPTIM_MATOPS_MIN_VAL(simplex_fn_vals);

    //
    // begin loop

    //ChangedForGPBoost
    //if (print_level > 0) {
    //    std::cout << "\nNelder-Mead: beginning search...\n";

    //    if (print_level >= 3) {
    //        std::cout << "  - Initialization Phase:\n";
    //        std::cout << "    Objective function value at each vertex:\n";
    //        OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE(simplex_fn_vals) << "\n";
    //        std::cout << "    Simplex matrix:\n"; 
    //        OPTIM_MATOPS_COUT << simplex_points << "\n";
    //    }
    //}

    size_t iter = 0;
    double rel_objfn_change = 2*std::abs(rel_objfn_change_tol);
    double rel_sol_change = 2*std::abs(rel_sol_change_tol);

    simplex_fn_vals_old = simplex_fn_vals;
    simplex_points_old = simplex_points;

    bool has_converged = false;
    while (!has_converged) {
        ++iter;
        bool next_iter = false;
        
        // step 1

        // VecInt_t sort_vec = arma::sort_index(simplex_fn_vals); // sort from low (best) to high (worst) values
        VecInt_t sort_vec = get_sort_index(simplex_fn_vals); // sort from low (best) to high (worst) values

        simplex_fn_vals = OPTIM_MATOPS_EVAL(simplex_fn_vals(sort_vec));
        simplex_points = OPTIM_MATOPS_EVAL(OPTIM_MATOPS_ROWS(simplex_points, sort_vec));

        // step 2

        Vec_t centroid = OPTIM_MATOPS_TRANSPOSE( OPTIM_MATOPS_COLWISE_SUM( OPTIM_MATOPS_MIDDLE_ROWS(simplex_points, 0, n_vals-1) ) ) / static_cast<double>(n_vals);

        Vec_t x_r = centroid + par_alpha*(centroid - OPTIM_MATOPS_TRANSPOSE(simplex_points.row(n_vals)));

        double f_r = box_objfn(x_r, nullptr, opt_data);

        if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) {
            // reflected point is neither best nor worst in the new simplex
            simplex_points.row(n_vals) = OPTIM_MATOPS_TRANSPOSE(x_r);
            simplex_fn_vals(n_vals) = f_r;
            next_iter = true;
        }

        // step 3

        if (!next_iter && f_r < simplex_fn_vals(0)) {
            // reflected point is better than the current best; try to go farther along this direction
            Vec_t x_e = centroid + par_gamma*(x_r - centroid);

            double f_e = box_objfn(x_e, nullptr, opt_data);

            if (f_e < f_r) {
                simplex_points.row(n_vals) = OPTIM_MATOPS_TRANSPOSE(x_e);
                simplex_fn_vals(n_vals) = f_e;
            } else {
                simplex_points.row(n_vals) = OPTIM_MATOPS_TRANSPOSE(x_r);
                simplex_fn_vals(n_vals) = f_r;
            }

            next_iter = true;
        }

        // steps 4, 5, 6

        if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) {
            // reflected point is still worse than x_n; contract

            // steps 4 and 5

            if (f_r < simplex_fn_vals(n_vals)) {
                // outside contraction
                Vec_t x_oc = centroid + par_beta*(x_r - centroid);

                double f_oc = box_objfn(x_oc, nullptr, opt_data);

                if (f_oc <= f_r)
                {
                    simplex_points.row(n_vals) = OPTIM_MATOPS_TRANSPOSE(x_oc);
                    simplex_fn_vals(n_vals) = f_oc;
                    next_iter = true;
                }
            } else {
                // inside contraction: f_r >= simplex_fn_vals(n_vals)
                
                // x_ic = centroid - par_beta*(x_r - centroid);
                Vec_t x_ic = centroid + par_beta*( OPTIM_MATOPS_TRANSPOSE(simplex_points.row(n_vals)) - centroid );

                double f_ic = box_objfn(x_ic, nullptr, opt_data);

                if (f_ic < simplex_fn_vals(n_vals))
                {
                    simplex_points.row(n_vals) = OPTIM_MATOPS_TRANSPOSE(x_ic);
                    simplex_fn_vals(n_vals) = f_ic;
                    next_iter = true;
                }
            }
        }

        // step 6

        if (!next_iter) {
            // neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
            for (size_t i = 1; i < n_vals + 1; i++) {
                simplex_points.row(i) = simplex_points.row(0) + par_delta*(simplex_points.row(i) - simplex_points.row(0));
            }

#ifdef OPTIM_USE_OMP
            #pragma omp parallel for
#endif
            for (size_t i = 1; i < n_vals + 1; i++) {
                simplex_fn_vals(i) = box_objfn( OPTIM_MATOPS_TRANSPOSE(simplex_points.row(i)), nullptr, opt_data);
            }
        }

        min_val = OPTIM_MATOPS_MIN_VAL(simplex_fn_vals);

        // double change_val_min = std::abs(min_val - OPTIM_MATOPS_MIN_VAL(simplex_fn_vals));
        // double change_val_max = std::abs(min_val - OPTIM_MATOPS_MAX_VAL(simplex_fn_vals));
    
        // rel_objfn_change = std::max( change_val_min, change_val_max ) / (1.0e-08 + OPTIM_MATOPS_ABS_MAX_VAL(simplex_fn_vals));

        rel_objfn_change = (OPTIM_MATOPS_ABS_MAX_VAL(simplex_fn_vals - simplex_fn_vals_old)) / (1.0e-08 + OPTIM_MATOPS_ABS_MAX_VAL(simplex_fn_vals_old));
        simplex_fn_vals_old = simplex_fn_vals;

        if (rel_sol_change_tol >= 0.0) { 
            rel_sol_change = (OPTIM_MATOPS_ABS_MAX_VAL(simplex_points - simplex_points_old)) / (1.0e-08 + OPTIM_MATOPS_ABS_MAX_VAL(simplex_points_old));
            simplex_points_old = simplex_points;
        }

        has_converged = !(rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max);

        //ChangedForGPBoost
        if (settings_inp) {
            settings_inp->opt_iter = iter - 1;
        }
        //redetermine neighbors for the Vecchia approximation if applicable
        Vec_t gradient_dummy(3);//"hack" for redermininig neighbors for the Vecchia approximation and/or inducing points (i.e. calling RedetermineNearestNeighborsVecchiaInducingPointsFITC())
        gradient_dummy[0] = 1.00000000001e30;
        gradient_dummy[1] = -1.00000000001e30;
        if (has_converged) {
            gradient_dummy[2] = 1.00000000001e30;//hack to force redetermination of nearest neighbors
        }
        else {
            gradient_dummy[2] = -1.00000000001e30;
        }
        double neighbors_have_been_redetermined = opt_objfn(simplex_points.row(index_min(simplex_fn_vals)), &gradient_dummy, opt_data);
        if (neighbors_have_been_redetermined >= 1e30 && neighbors_have_been_redetermined <= 1.00000000002e30) {//hack that indicates that the neighbors have indeed been redetermined
            //recalculated objective values if neighbors have been redetermined and check convergence again
#ifdef OPTIM_USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 1; i < n_vals + 1; i++) {
                simplex_fn_vals(i) = box_objfn(OPTIM_MATOPS_TRANSPOSE(simplex_points.row(i)), nullptr, opt_data);
            }
            rel_objfn_change = (OPTIM_MATOPS_ABS_MAX_VAL(simplex_fn_vals - simplex_fn_vals_old)) / (1.0e-08 + OPTIM_MATOPS_ABS_MAX_VAL(simplex_fn_vals_old));
            has_converged = !(rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max);
        }

        //print trace information
        if ((iter < 10 || (iter % 10 == 0 && iter < 100) || (iter % 100 == 0 && iter < 1000) ||
            (iter % 1000 == 0 && iter < 10000) || (iter % 10000 == 0)) && (iter != iter_max)) {
            gradient_dummy[0] = -1.00000000001e30;//"hack" for printing nice logging information
            gradient_dummy[1] = 1.00000000001e30;
            gradient_dummy[2] = min_val;
            opt_objfn(simplex_points.row(index_min(simplex_fn_vals)), &gradient_dummy, opt_data);
        }
        //OPTIM_NM_TRACE(iter, min_val, rel_objfn_change, rel_sol_change, simplex_fn_vals, simplex_points);
    }//end while loop

    //ChangedForGPBoost
    //if (print_level > 0) {
    //    std::cout << "Nelder-Mead: search completed.\n";
    //}

    //

    Vec_t prop_out = OPTIM_MATOPS_TRANSPOSE(simplex_points.row(index_min(simplex_fn_vals)));
    
    if (vals_bound) {
        prop_out = inv_transform(prop_out, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, prop_out, opt_objfn, opt_data,
                    success, rel_objfn_change, rel_objfn_change_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);

    //
    
    return success;
}

inline
bool
nm(Vec_t& init_out_vals, 
          std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
          void* opt_data)
{
    return internal::nm_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
nm(Vec_t& init_out_vals, 
          std::function<double (const Vec_t& vals_inp, Vec_t*, void* opt_data)> opt_objfn, 
          void* opt_data, 
          algo_settings_t& settings)
{
    return internal::nm_impl(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
