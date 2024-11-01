/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2024 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_VECCHIA_H_
#define GPB_VECCHIA_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <GPBoost/utils.h>
#include <vector>
#include <algorithm>

namespace GPBoost {

	template<typename T_mat>
	void distances_funct(const int& coord_ind_i,
		const std::vector<int>& coords_ind_j,
		const den_mat_t& coords,
		const vec_t& corr_diag, 
		const den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		vec_t& distances,
		string_t dist_function,
		bool distances_saved) {

		if (dist_function == "residual_correlation_FSA") {
			vec_t pp_node(coords_ind_j.size());
			vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(coord_ind_i);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < pp_node.size(); j++) {
				pp_node[j] = chol_ip_cross_cov.col(coords_ind_j[j]).dot(chol_ip_cross_cov_sample);
			}
			den_mat_t corr_mat, coords_i, coords_j;
			std::vector<den_mat_t> corr_mat_deriv;
			coords_i = coords(coord_ind_i, Eigen::all);
			coords_j = coords(coords_ind_j, Eigen::all);
			den_mat_t dist_ij;
			if (distances_saved) {
				dist_ij.resize(coords_ind_j.size(), 1);
#pragma omp parallel for schedule(static)
				for (int j = 0; j < coords_ind_j.size(); j++) {
					dist_ij.coeffRef(j, 0) = (coords_j(j, Eigen::all) - coords_i).lpNorm<2>();
				}
			}
			re_comps_resid_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(dist_ij, coords_i, coords_j,
				corr_mat, corr_mat_deriv.data(), false, true, 1., false);
			double corr_diag_sample = corr_diag(coord_ind_i);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < coords_ind_j.size(); j++) {
				distances[j] = std::sqrt((1. - std::pow(std::abs((corr_mat.data()[j] - pp_node[j]) /
					std::sqrt(corr_diag_sample * corr_diag[coords_ind_j[j]])),(double)0.1)));
				/*distances[j] = std::sqrt((1. - ((corr_mat.data()[j] - pp_node[j]) /
					std::sqrt(corr_diag_sample * corr_diag[coords_ind_j[j]])))*0.5);*/
			}
		}
	}

	template<typename T_mat>
	void CoverTree_kNN(const den_mat_t& coords_mat,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& corr_diag,
		const double base,
		const int start,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		std::map<int, std::vector<int>>& cover_tree,
		int& level,
		bool distances_saved,
		string_t dist_function) {
		//std::chrono::steady_clock::time_point begin, end;//only for debugging
		//double el_time;//only for debugging
		//begin = std::chrono::steady_clock::now();//only for debugging
		den_mat_t coords = coords_mat;
		// Distances already computed
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_cluster_i[0]);
		den_mat_t coords_i, coords_j, corr_mat, dist_ij;
		std::vector<den_mat_t> corr_mat_deriv;//help matrix
		int sample;
		//Select data point with index 0 as root
		int root = start;
		cover_tree.insert({ -1, { root } });
		//max_dist of root
		//double R_max = 1.;// pow(base, 1);
		double R_max = 1.;
		double R_l;
		// Initialize
		std::vector<int> all_indices(coords.rows() - 1);
		std::iota(std::begin(all_indices), std::end(all_indices), 1);
		std::map<int, std::vector<int>> covert_points_old;
		covert_points_old[0] = all_indices;
		level = 0;
		//std::vector<int> new_parents = { root };
		while ((cover_tree.size()-1) != coords.rows()) {
			level += 1;
			R_l = R_max / pow(base, level);
			std::map<int, std::vector<int>> covert_points;
			for (const auto& key_covert_points_old_i : covert_points_old) {
				int key = key_covert_points_old_i.first;
				std::vector<int> covert_points_old_i = key_covert_points_old_i.second;
				// sample new node
				bool not_all_covered = covert_points_old_i.size() > 0;
				cover_tree.insert({ key + start, { key + start } });
				while (not_all_covered) {
					int sample_ind = covert_points_old_i[0];
					cover_tree[key + start].push_back(sample_ind + start);
					//new_parents.push_back(sample_ind);
					// new covered points per node
					std::vector<int> covert_points_old_i_up;
					for (int j : covert_points_old_i) {
						if (j > sample_ind) {
							covert_points_old_i_up.push_back(j);
						}
					}
					vec_t dist_vect((int)covert_points_old_i_up.size());
					distances_funct<T_mat>(sample_ind, covert_points_old_i_up, coords, corr_diag, chol_ip_cross_cov,
						re_comps_resid_cluster_i, dist_vect, dist_function, distances_saved);
					for (int j = 0; j < dist_vect.size(); j++) {
						if (dist_vect[j] <= R_l) {
							if (covert_points.find(sample_ind) == covert_points.end()) {
								covert_points.insert({ sample_ind, { covert_points_old_i_up[j] } });
							}
							else {
								covert_points[sample_ind].push_back(covert_points_old_i_up[j]);
							}
						}
					}
					std::vector<int> covert_points_vect = covert_points[sample_ind];
					covert_points_old_i.erase(covert_points_old_i.begin());
					std::vector<int> diff_vect;
					std::set_difference(covert_points_old_i.begin(), covert_points_old_i.end(),
						covert_points_vect.begin(), covert_points_vect.end(),
						std::inserter(diff_vect, diff_vect.begin()));
					covert_points_old_i = diff_vect;
					not_all_covered = covert_points_old_i.size() > 0;
				}
			}
			covert_points_old.clear();
			covert_points_old = covert_points;
		}
		//for (int i = 0; i < new_parents.size(); i++) {
		//	std::vector<int> id_parent{ new_parents[i]};
		//	cover_tree.insert({ new_parents[i], { new_parents[i]} });
		//	//std::set<std::tuple<int, double>> id_parent{ { new_parents[i], 0.} };
		//	//cover_tree.insert({ new_parents[i], id_parent });
		//}
		//level += 1;
		//Log::REInfo("Depth of CoverTree: %i and size %i", level, cover_tree.size());
		//end = std::chrono::steady_clock::now();//only for debugging
		//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
		//Log::REInfo(" time until CoverTree for kNN = %g ", el_time);
	}//end CoverTree_kNN

//	template<typename T_mat>
//	void find_kNN_CoverTree(const int i, 
//		const int k,
//		const int levels,
//		const bool distances_saved,
//		const double base,
//		const den_mat_t& coords,
//		const den_mat_t& chol_ip_cross_cov,
//		const vec_t& corr_diag,
//		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
//		std::vector<int>& neighbors_i,
//		std::map<int, std::vector<int>>& cover_tree,
//		string_t dist_function) {
//		std::chrono::steady_clock::time_point begin, end;//only for debugging
//		double el_time;//only for debugging
//		if (i == 10011) {
//			begin = std::chrono::steady_clock::now();//only for debugging
//		}
//		int num_smaller_ind = 0;
//		// Initialize distance and help matrices
//		den_mat_t coords_j, corr_mat, dist_ij;
//		std::vector<den_mat_t> corr_mat_deriv;
//		// query point
//		den_mat_t coords_i = coords(i, Eigen::all);
//		vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(i);
//		double corr_diag_sample = corr_diag[i];
//		// Initialize vectors for distance computations
//		vec_t pp_node, dist_vect, dist_vect_before(0);
//		std::vector<double> dist_vect_prev;
//		// Initialize sets
//		std::set<int> Q;
//		std::vector<int> Q_before_vect;
//		std::set<int> Q_i;
//		int root = cover_tree[-1][0];
//		Q_i.insert(root);
//		//Q_i.insert(std::get<0>(*(cover_tree[-1].begin())));
//		std::set<int> Q_i_before;
//		std::vector<int> Q_i_end;
//		std::vector<double> Q_i_dist;
//		//std::vector<int> diff;
//		std::vector<std::set<int>::iterator> diff;
//		std::vector<int> diff_rev = { root };
//		//std::vector<int> diff_rev = { std::get<0>(*(cover_tree[-1].begin())) };
//		//std::vector<double> diff_rev_dist = { 1. };
//		// threshold distance
//		double dist_k_Q_cor = 1.;
//		bool early_stop = false;
//		int size_before;
//		int k_scaled = k;
//		if (i == 10011) {
//			end = std::chrono::steady_clock::now();//only for debugging
//			el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//			Log::REInfo(" time until a = %g ", el_time);
//		}
//		for (int ii = 1; ii < levels; ii++) {
//			// build set of children
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			/*for (int j = 0; j < diff.size(); j++) {
//				Q.erase(diff[j]);
//
//			}*/
//			if (diff.size() > 0) {
//				for (auto it : diff) {
//					Q.erase(it);
//				}
//			}
//			for (int j = 0; j < diff_rev.size(); j++) {
//				//std::vector<int> children_Q_i_j(cover_tree[diff_rev[j]].begin(), cover_tree[diff_rev[j]].end());
//				int child_jj;
//				for (int jj = 0; jj < cover_tree[diff_rev[j]].size(); jj++) {
//					child_jj = cover_tree[diff_rev[j]][jj];
//					if (child_jj < i) {
//						Q.insert(child_jj);
//					}
//					else {
//						break;
//					}
//				}
//				//std::vector<std::tuple<int, double>> children_Q_i_j(cover_tree[diff_rev[j]].begin(), cover_tree[diff_rev[j]].end());
//				//for (int jj = 0; jj < children_Q_i_j.size(); jj++) {
//				//	if (std::get<0>(children_Q_i_j[jj]) < i) {
//				//		Q.insert(std::get<0>(children_Q_i_j[jj]));
//				//		/*if (dist_k_Q_cor < 1.) {
//				//			if (std::abs(std::get<1>(children_Q_i_j[jj]) - diff_rev_dist[j]) < dist_k_Q_cor) {
//				//				Q.insert(std::get<0>(children_Q_i_j[jj]));
//				//			}
//				//		}
//				//		else {
//				//			Q.insert(std::get<0>(children_Q_i_j[jj]));
//				//		}*/
//				//	}
//				//	else {
//				//		break;
//				//	}
//				//}
//			}
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until b = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			std::vector<int> Q_vec(Q.begin(), Q.end());
//			//intersection_ind_1.clear();
//			//intersection_ind_2.clear();
//			//remaining.clear();
//			//remaining_ind.clear();
//			std::vector<int> intersection_ind_1;
//			std::vector<int> intersection_ind_2;
//			std::vector<int> remaining;
//			std::vector<int> remaining_ind;
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until c = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			if (ii > 1) {
//				my_intersection<int>(Q_before_vect.begin(), Q_before_vect.end(),
//					Q_vec.begin(), Q_vec.end(), std::back_inserter(intersection_ind_1),
//					std::back_inserter(intersection_ind_2), std::back_inserter(remaining),
//					std::back_inserter(remaining_ind));
//			}
//			else {
//				remaining = Q_vec;
//				remaining_ind.resize(remaining.size());
//				std::iota(remaining_ind.begin(), remaining_ind.end(), 0);
//			}
//			early_stop = remaining.size() == 0 || ii == (levels - 1);
//			Q_i_before.clear();
//			Q_i_before = Q_i;
//			// Add already computed distances
//			dist_vect.resize(Q_vec.size());
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until d = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			if (intersection_ind_1.size() > 0) {
//#pragma omp parallel for schedule(static)
//				for (int j = 0; j < intersection_ind_1.size(); j++) {
//					dist_vect[intersection_ind_2[j]] = dist_vect_before[intersection_ind_1[j]];
//				}
//			}
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until e = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			if (remaining.size() > 0) {
//				vec_t dist_vect_interim(remaining.size());
//				distances_funct<T_mat>(i, remaining, coords, corr_diag, chol_ip_cross_cov,
//					re_comps_resid_cluster_i, dist_vect_interim, dist_function);
//#pragma omp parallel for schedule(static)
//				for (int j = 0; j < remaining_ind.size(); j++) {
//					dist_vect[remaining_ind[j]] = dist_vect_interim[j];
//				}
//			}
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until f = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			// Find k-th smallest element
//			if (dist_vect.size() < k_scaled) {
//				dist_k_Q_cor = dist_vect.maxCoeff();
//			}
//			else {
//				dist_vect_prev.clear();
//				for (int j = 0; j < dist_vect.size(); j++) {
//					dist_vect_prev.push_back(dist_vect[j]);
//				}
//				std::nth_element(dist_vect_prev.begin(), dist_vect_prev.begin() + k_scaled - 1, dist_vect_prev.end());
//				dist_k_Q_cor = dist_vect_prev[k_scaled - 1];
//				dist_vect_prev.clear();
//			}
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until g = %g ", el_time);
//			}
//			if (i == 10011) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			dist_k_Q_cor += 1/std::pow(base,ii-1);// TODO? -2
//			diff.clear();
//			diff_rev.clear();
//			//diff_rev_dist.clear();
//			if (dist_k_Q_cor >= 1.) {
//				Q_i = Q;
//				if (early_stop) {
//					for (int j = 0; j < dist_vect.size(); j++) {
//						num_smaller_ind += 1;
//						Q_i_end.push_back(Q_vec[j]);
//						Q_i_dist.push_back(dist_vect[j]);
//					}
//				}
//				else {
//					std::set_difference(Q_i.begin(), Q_i.end(), Q_i_before.begin(), Q_i_before.end(),
//						std::inserter(diff_rev, diff_rev.begin()));
//				}
//			}
//			else {
//				std::set<int>::iterator it = Q.begin();
//				std::set<int>::iterator it_i = Q_i.begin();
//				for (int j = 0; j < dist_vect.size(); j++) {
//					if (dist_vect[j] <= dist_k_Q_cor) {
//						if (early_stop) {
//							num_smaller_ind += 1;
//							Q_i_dist.push_back(dist_vect[j]);
//							Q_i_end.push_back(Q_vec[j]);
//						}
//						else {
//							size_before = (int)Q_i.size();
//							Q_i.insert(Q_vec[j]);
//							if (size_before != Q_i.size()) {
//								diff_rev.push_back(Q_vec[j]);
//								//diff_rev_dist.push_back(dist_vect[j]);
//							}
//						}
//					}
//					else if (!early_stop) {
//						size_before = (int)Q_i.size();
//						Q_i.erase(it_i);
//						if (size_before != Q_i.size()) {
//							diff.push_back(it);
//						}
//					}
//					++it;
//					++it_i;
//				}
//			}
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until h = %g ", el_time);
//			}
//			if (i == 10010) {
//				begin = std::chrono::steady_clock::now();//only for debugging
//			}
//			if (early_stop) {
//				break;
//			}
//			dist_vect_before.resize(dist_vect.size());
//			dist_vect_before = dist_vect;
//			Q_before_vect.clear();
//			Q_before_vect = Q_vec;
//			if (i == 10011) {
//				end = std::chrono::steady_clock::now();//only for debugging
//				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//				Log::REInfo(" time until i = %g ", el_time);
//			}
//		}
//		if (i == 10011) {
//			begin = std::chrono::steady_clock::now();//only for debugging
//		}
//		// Sort vector
//		if (num_smaller_ind >= k) {
//			//std::vector<int> sort_vect(Q_i_dist.size());
//			//SortIndeces<double>(Q_i_dist, sort_vect);
//
//			// k-th element
//			//std::nth_element(Q_i_end_tuple.begin(), Q_i_end_tuple.begin() + k - 1, Q_i_end_tuple.end(),
//			//	[](const std::tuple<int, double>& a,
//			//		const std::tuple<int, double>& b) {
//			//			return std::get<1>(a) < std::get<1>(b);  // Compare based on the first element
//			//	});
//
//			// First k-th elements
//			//Q_i_end_tuple.resize(k);
//			// sort by second element
//			//std::sort(Q_i_end_tuple.begin(), Q_i_end_tuple.end(), 
//			//	[](const std::tuple<int, double>& a,
//			//		const std::tuple<int, double>& b) {
//			//			return std::get<1>(a) < std::get<1>(b);
//			//	});
////#pragma omp parallel for schedule(static)
////			for (int ii = 0; ii < k; ii++){
////				neighbors_i[ii] = std::get<1>(*std::next(Q_i_end_tuple.begin(), ii));
////			}
////#pragma omp parallel for schedule(static)
////			for (int ii = 0; ii < k; ii++){
////				neighbors_i[ii] = Q_i_end[sort_vect[ii]];
////			}
//			std::vector<double> nn_dist(k);
//#pragma omp parallel for schedule(static)
//			for (int j = 0; j < k; ++j) {
//				nn_dist[j] = std::numeric_limits<double>::infinity();
//			}
//			for (int jj = 0; jj < Q_i_dist.size(); ++jj) {
//				if (Q_i_dist[jj] < nn_dist[k - 1]) {
//					nn_dist[k - 1] = Q_i_dist[jj];
//					neighbors_i[k - 1] = Q_i_end[jj];
//					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
//				}
//			}
//			if (i == 10010) {
//				for (int ii = 0; ii < k; ii++) {
//					Log::REInfo("knn %i %i %g %g %g %i", ii, neighbors_i[ii],
//						coords.coeffRef(neighbors_i[ii]),
//						coords.coeffRef(neighbors_i[ii]), nn_dist[ii], num_smaller_ind);
//				}
//			}
//		}
//		else {
//			vec_t dist_vect(1);
//			std::vector<double> nn_dist(k);
//#pragma omp parallel for schedule(static)
//			for (int j = 0; j < k; ++j) {
//				nn_dist[j] = std::numeric_limits<double>::infinity();
//			}
//			for (int jj = 0; jj < i; ++jj) {
//				std::vector<int> indj{ jj };
//				distances_funct<T_mat>(i, indj, coords, corr_diag, chol_ip_cross_cov,
//					re_comps_resid_cluster_i, dist_vect, dist_function);
//				if (dist_vect[0] < nn_dist[k - 1]) {
//					nn_dist[k - 1] = dist_vect[0];
//					neighbors_i[k - 1] = jj;
//					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
//				}
//			}
//
//
//			/*std::vector<double> nn_dist(k);
//			dist_vect.resize(i);
//			std::vector<int> coords_ind_j(i);
//			std::iota(coords_ind_j.begin(), coords_ind_j.end(), 0);
//			distances_funct<T_mat>(i, coords_ind_j, coords, corr_diag, chol_ip_cross_cov,
//				re_comps_resid_cluster_i, dist_vect, dist_function);
//			dist_vect_prev.clear();
//			for (int jj = 0; jj < i; ++jj) {
//				dist_vect_prev.push_back(dist_vect[jj]);
//			}
//			std::vector<int> sort_vect(dist_vect_prev.size());
//			SortIndeces<double>(dist_vect_prev, sort_vect);
//#pragma omp parallel for schedule(static)
//			for (int jj = 0; jj < k; jj++) {
//				neighbors_i[jj] = sort_vect[jj];
//				nn_dist[jj] = dist_vect_prev[sort_vect[jj]];
//			}
//			SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);*/
//			//num_smaller_ind = k;
//			if (i == 10010) {
//				Log::REInfo("num_smaller_ind %i",num_smaller_ind);
//				for (int ii = 0; ii < k; ii++) {
//					Log::REInfo("knns %i %i %g %g %g", ii, neighbors_i[ii],
//						coords.coeffRef(neighbors_i[ii], 0),
//						coords.coeffRef(neighbors_i[ii], 1), nn_dist[ii]);
//				}
//			}
//		}
//		if (i == 10011) {
//			end = std::chrono::steady_clock::now();//only for debugging
//			el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
//			Log::REInfo(" time until j = %g ", el_time);
//		}
//	}//end find_kNN_CoverTree

	template<typename T_mat>
	void find_kNN_CoverTree(const int i, 
		const int k,
		const int levels,
		const bool distances_saved,
		const double base,
		const den_mat_t& coords,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& corr_diag,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		std::vector<int>& neighbors_i,
		std::vector<double>& dist_of_neighbors_i,
		std::map<int, std::vector<int>>& cover_tree,
		string_t dist_function) {
		// Initialize distance and help matrices
		den_mat_t coords_j, corr_mat, dist_ij;
		std::vector<den_mat_t> corr_mat_deriv;
		// query point
		den_mat_t coords_i = coords(i, Eigen::all);
		vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(i);
		// Initialize vectors
		int root = cover_tree[-1][0];
		std::vector<int> Q;
		std::vector<double> Q_dist;
		std::vector<int> diff_rev = { root };
		// threshold distance
		double max_dist = 1.;
		double dist_k_Q_cor = max_dist;
		bool early_stop = false;
		int k_scaled = k;
		int Q_before_size = 1;
		for (int ii = 1; ii < levels; ii++) {
			// build set of children
			std::vector<int> diff_rev_interim;
			if (ii == 1) {
				Q.push_back(root);
				diff_rev_interim.push_back(root);
			}
			for (int j : diff_rev) {
				for (int jj : cover_tree[j]) {
					if (jj < i) {
						if (jj != j) {
							Q.push_back(jj);
							diff_rev_interim.push_back(jj);
						}
					}
					else {
						break;
					}
				}
			}
			diff_rev.clear();
			early_stop = diff_rev_interim.size() == 0 || ii == (levels - 1);
			if (diff_rev_interim.size() > 0) {
				vec_t dist_vect_interim(diff_rev_interim.size());
				distances_funct<T_mat>(i, diff_rev_interim, coords, corr_diag, chol_ip_cross_cov,
					re_comps_resid_cluster_i, dist_vect_interim, dist_function, distances_saved);
				int dist_vect_interim_size = dist_vect_interim.size();
				for (int j = 0; j < dist_vect_interim_size; j++) {
					Q_dist.push_back(dist_vect_interim[j]);
				}
			}
			// Find k-th smallest element
			if (ii > 1) {
				if (Q_dist.size() < k_scaled) {
					dist_k_Q_cor = *std::max_element(Q_dist.begin(), Q_dist.end());
				}
				else {
					std::vector<double> Q_dist_interim(Q_dist.begin(), Q_dist.end());
					std::nth_element(Q_dist_interim.begin(), Q_dist_interim.begin() + k_scaled - 1, Q_dist_interim.end());
					dist_k_Q_cor = Q_dist_interim[k_scaled - 1];
				}
				dist_k_Q_cor += 1 / std::pow(base, ii - 1);// TODO? -2
			}
			// TODO erase which are too big in Q and Q_dist 
			int count = 0;
			if (dist_k_Q_cor >= max_dist) {
				if (!early_stop) {
					diff_rev = diff_rev_interim;
					if (ii == 1) {
						diff_rev.erase(diff_rev.begin());
					}
				}
			}
			else {
				std::vector<double> Q_dist_interim;
				std::vector<int> Q_interim;
				auto xi = Q_dist.begin();
				auto yi = Q.begin();
				while (xi != Q_dist.end() && yi != Q.end()) {
					if ((double)*xi <= dist_k_Q_cor) {
						Q_dist_interim.push_back((double)*xi);
						Q_interim.push_back((int)*yi);
						if (count >= Q_before_size) {
							diff_rev.push_back(*yi);
						}
					}
					count += 1;
					++xi;
					++yi;
				}
				Q.clear();
				Q_dist.clear();
				Q = Q_interim;
				Q_dist = Q_dist_interim;
			}
			Q_before_size = (int)Q.size();
			if (early_stop) {
				break;
			}
		}
		std::vector<double> nn_dist(k);
		if (Q_before_size >= k) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < k; ++j) {
				nn_dist[j] = std::numeric_limits<double>::infinity();
			}
			for (int jj = 0; jj < Q_before_size; ++jj) {
				if (Q_dist[jj] < nn_dist[k - 1]) {
					nn_dist[k - 1] = Q_dist[jj];
					neighbors_i[k - 1] = Q[jj];
					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
				}
			}
			/*if (i == 10010) {
				for (int ii = 0; ii < k; ii++) {
					Log::REInfo("knn %i %i %g %g %g %i", ii, neighbors_i[ii],
						coords.coeffRef(neighbors_i[ii]),
						coords.coeffRef(neighbors_i[ii]), nn_dist[ii], Q_before_size);
				}
			}*/
		}
		else {
			vec_t dist_vect(1);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < k; ++j) {
				nn_dist[j] = std::numeric_limits<double>::infinity();
			}
			for (int jj = 0; jj < i; ++jj) {
				std::vector<int> indj{ jj };
				distances_funct<T_mat>(i, indj, coords, corr_diag, chol_ip_cross_cov,
					re_comps_resid_cluster_i, dist_vect, dist_function, distances_saved);
				if (dist_vect[0] < nn_dist[k - 1]) {
					nn_dist[k - 1] = dist_vect[0];
					neighbors_i[k - 1] = jj;
					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
				}
			}
			/*if (i == 10010) {
				Log::REInfo("num_smaller_ind %i", Q_before_size);
				for (int ii = 0; ii < k; ii++) {
					Log::REInfo("knns %i %i %g %g %g", ii, neighbors_i[ii],
						coords.coeffRef(neighbors_i[ii], 0),
						coords.coeffRef(neighbors_i[ii], 1), nn_dist[ii]);
				}
			}*/
		}
		dist_of_neighbors_i = nn_dist;
	}//end find_kNN_CoverTree

	/*!
	* \brief Finds the nearest_neighbors among the previous observations using the fast mean-distance-ordered nn search by Ra and Kim (1993)
	* \param coords Coordinates of observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param re_comps_resid_cluster_i Container that collects the individual component models
	* \param[out] neighbors Vector with indices of neighbors for every observations (length = num_data - start_at)
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation: distances between locations and their neighbors (length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation: distances between all neighbors (length = num_data - start_at)
	* \param start_at Index of first point for which neighbors should be found (useful for prediction, otherwise = 0)
	* \param end_search_at Index of last point which can be a neighbor (useful for prediction when the neighbors are only to be found among the observed data, otherwise = num_data - 1 (if end_search_at < 0, we set end_search_at = num_data - 1)
	* \param[out] check_has_duplicates If true, it is checked whether there are duplicates in coords among the neighbors (result written on output)
	* \param neighbor_selection The way how neighbors are selected
	* \param gen RNG
	* \param save_distances If true, distances are saved in dist_obs_neighbors and dist_between_neighbors
	* \param clusters index of the inducing point each data point is closest to
	*/
	template<typename T_mat>
	void find_nearest_neighbors_Vecchia_FSA_fast(const den_mat_t& coords,
		int num_data,
		int num_neighbors,
		den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		std::vector<std::vector<int>>& neighbors,
		std::vector<den_mat_t>& dist_obs_neighbors,
		std::vector<den_mat_t>& dist_between_neighbors,
		int start_at,
		int end_search_at,
		bool& check_has_duplicates,
		RNG_t& gen,
		bool save_distances,
		double base,
		bool prediction,
		bool cond_on_all,
		const int& num_data_obs) {
		string_t dist_function = "residual_correlation_FSA";
		CHECK((int)neighbors.size() == (num_data - start_at));
		if (save_distances) {
			CHECK((int)dist_obs_neighbors.size() == (num_data - start_at));
			CHECK((int)dist_between_neighbors.size() == (num_data - start_at));
		}
		CHECK((int)coords.rows() == num_data);
		if (end_search_at < 0) {
			end_search_at = num_data - 2;
		}
		if (num_neighbors > end_search_at + 1) {
			Log::REInfo("The number of neighbors (%d) for the Vecchia approximation needs to be smaller than the number of data points (%d). It is set to %d.", num_neighbors, end_search_at + 2, end_search_at + 1);
			num_neighbors = end_search_at + 1;
		}
		int num_nearest_neighbors = num_neighbors;
		bool has_duplicates = false;
		// For correlation matrix
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_cluster_i[0]);
		std::vector<den_mat_t> corr_mat_deriv;//help matrix
		// Variance for residual process
		vec_t corr_diag(num_data);
		den_mat_t dist_ii(1, 1);
		dist_ii(0, 0) = 0.;
		den_mat_t corr_mat_i;
		den_mat_t coords_ii;
		std::vector<int> indii{ 0 };
		coords_ii = coords(indii, Eigen::all);
		//re_comp->GetSubSetCoords(indii, coords_ii);
		re_comps_resid_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(dist_ii, coords_ii, coords_ii,
			corr_mat_i, corr_mat_deriv.data(), false, true, 1., false);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			corr_diag[i] = corr_mat_i.value() - chol_ip_cross_cov.col(i).array().square().sum();
		}
		//Intialize neighbor vectors
		for (int i = start_at; i < num_data; ++i) {
			if (i > 0 && i <= num_neighbors) {
				neighbors[i - start_at].resize(i);
				if (save_distances) {
					dist_obs_neighbors[i - start_at].resize(i, 1);
				}
				for (int j = 0; j < i; ++j) {
					neighbors[i - start_at][j] = j;
					den_mat_t dist_ij(1, 1);
					dist_ij(0, 0) = 0.;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						dist_ij(0, 0) = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_obs_neighbors[i - start_at](j, 0) = dist_ij.value();
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dist_ij.value() < EPSILON_NUMBERS) {
							has_duplicates = true;
						}
					}//end check_has_duplicates
				}
			}
			else if (i > num_neighbors) {
				neighbors[i - start_at].resize(num_neighbors);
			}
		}
		//Find neighbors for those points where the conditioning set (=candidate neighbors) is larger than 'num_neighbors'
		if (num_data > num_neighbors) {
			int first_i = (start_at <= num_neighbors) ? (num_neighbors + 1) : start_at;//The first point (first_i) for which the search is done is the point with index (num_neighbors + 1) or start_at
			// Brute force kNN search until certain number of data points
			int brute_force_threshold = std::min(num_data, std::max(1000, num_neighbors));
			if(prediction){
				brute_force_threshold = std::min(num_data, std::max(first_i + 500, num_neighbors));
			}
			int max_ind_nn = num_data_obs;
			if (cond_on_all) {
				max_ind_nn = num_data;
			}
#pragma omp parallel for schedule(static)
			for (int i = first_i; i < brute_force_threshold; ++i) {
				vec_t dist_vect(1);
				std::vector<double> nn_corr(num_neighbors);
#pragma omp parallel for schedule(static)
				for (int j = 0; j < num_neighbors; ++j) {
					nn_corr[j] = std::numeric_limits<double>::infinity();
				}
				for (int jj = 0; jj < (int)std::min(i, max_ind_nn); ++jj) {
					std::vector<int> indj{ jj };
					distances_funct<T_mat>(i, indj, coords, corr_diag, chol_ip_cross_cov,
						re_comps_resid_cluster_i, dist_vect, dist_function, save_distances);
					if (dist_vect[0] < nn_corr[num_neighbors - 1]) {
						nn_corr[num_neighbors - 1] = dist_vect[0];
						neighbors[i - start_at][num_neighbors - 1] = jj;
						SortVectorsDecreasing<double>(nn_corr.data(), neighbors[i-start_at].data(), num_neighbors);
					}
				}
				//Save distances between points and neighbors
				if (save_distances) {
					dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
				}
				for (int jjj = 0; jjj < num_nearest_neighbors; ++jjj) {
					double dij = (coords(i, Eigen::all) - coords(neighbors[i-start_at][jjj], Eigen::all)).lpNorm<2>();
					if (save_distances) {
						dist_obs_neighbors[i - start_at](jjj, 0) = dij;
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dij < EPSILON_NUMBERS) {
#pragma omp critical
							{
								has_duplicates = true;
							}
						}
					}//end check_has_duplicates
				}
			}
			if (brute_force_threshold < num_data) {
				int level = 0;
				std::chrono::steady_clock::time_point begin, end;//only for debugging
				double el_time;//only for debugging
				begin = std::chrono::steady_clock::now();//only for debugging
				// Build CoverTree
				std::map<int, std::vector<int>> cover_tree;
				std::map<int, std::map<int, std::vector<int>>> cover_trees;
				std::vector<double> dist_dummy;
				// Set number of threads to the maximum available
				int num_threads;
#ifdef _OPENMP
				num_threads = omp_get_max_threads();
#else
				num_threads = 1;
#endif
				std::vector<int> levels_threads(num_threads);
				std::vector<int> segment_start(num_threads);
				std::vector<int> segment_length(num_threads);
				den_mat_t coords_ct;
				if (prediction && !cond_on_all) {
					coords_ct = coords.topRows(num_data_obs);
				}
				else {
					coords_ct = coords;
				}
				for (int i = 0; i < num_threads; ++i) {
					cover_trees[i] = cover_tree;
				}
				if (num_threads != 1) {
					int segment_size = std::ceil((double)coords_ct.rows() / (double)num_threads);
					int last_segment = coords_ct.rows() - (num_threads - 1) * segment_size;
					bool overhead = false;
					if (last_segment != segment_size) {
						num_threads -= 1;
						levels_threads.resize(num_threads);
						segment_start.resize(num_threads);
						segment_length.resize(num_threads);
						overhead = true;
					}
#pragma omp parallel for
					for (int i = 0; i < num_threads; ++i) {
						segment_start[i] = i * segment_size;
						segment_length[i] = segment_size;
						if (i == num_threads - 1 && overhead) {
							segment_length[i] += last_segment;
						}
						//Log::REInfo("Segment size %i %i %i", num_threads, segment_start[i], segment_length[i]);
						CoverTree_kNN<T_mat>(coords_ct.middleRows(segment_start[i], segment_length[i]), chol_ip_cross_cov.middleCols(segment_start[i], segment_length[i]),
							corr_diag.segment(segment_start[i], segment_length[i]), base, segment_start[i], re_comps_resid_cluster_i, cover_trees[i],
							levels_threads[i], save_distances, dist_function);
					}
				}
				else {
					CoverTree_kNN<T_mat>(coords_ct, chol_ip_cross_cov, corr_diag, base, 0, re_comps_resid_cluster_i, cover_trees[0],
						level, save_distances, dist_function);
				}
				end = std::chrono::steady_clock::now();//only for debugging
				el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
				Log::REInfo(" time until CoverTrees for kNN = %g ", el_time);
				Log::REInfo("num threads %i %i %i", num_threads, num_data, levels_threads[0]);
#pragma omp parallel for schedule(static)
				for (int i = brute_force_threshold; i < num_data; ++i) {
					if (num_threads != 1) {
						std::map<int,std::vector<int>> neighbors_per_tree;
						std::map<int,std::vector<double>> dist_of_neighbors_per_tree;
						for (int ii = 0; ii < num_threads; ++ii) {
							if (segment_start[ii] >= i) {
								break;
							}
							neighbors_per_tree[ii] = neighbors[i - start_at];
							dist_of_neighbors_per_tree[ii] = dist_dummy;
						}
#pragma omp parallel for schedule(static)
						for (int ii = 0; ii < neighbors_per_tree.size(); ++ii) {
							if ((segment_start[ii] + num_neighbors) < i) {
								find_kNN_CoverTree<T_mat>(i, num_neighbors, levels_threads[ii], save_distances, base, coords, chol_ip_cross_cov,
									corr_diag, re_comps_resid_cluster_i, neighbors_per_tree[ii], dist_of_neighbors_per_tree[ii], cover_trees[ii], dist_function);
							}
							else if (segment_start[ii] < i) {
								vec_t dist_vect(1);
								int size_smaller_k = std::min(i - segment_start[ii], num_neighbors);
								dist_of_neighbors_per_tree[ii].resize(size_smaller_k);
								neighbors_per_tree[ii].resize(size_smaller_k);
//#pragma omp parallel for schedule(static)
								for (int j = 0; j < size_smaller_k; ++j) {
									dist_of_neighbors_per_tree[ii][j] = std::numeric_limits<double>::infinity();
								}
								for (int jj = segment_start[ii]; jj < i; ++jj) {
									std::vector<int> indj{ jj };
									distances_funct<T_mat>(i, indj, coords, corr_diag, chol_ip_cross_cov,
										re_comps_resid_cluster_i, dist_vect, dist_function, save_distances);
									if (dist_vect[0] < dist_of_neighbors_per_tree[ii][size_smaller_k - 1]) {
										dist_of_neighbors_per_tree[ii][size_smaller_k - 1] = dist_vect[0];
										neighbors_per_tree[ii][size_smaller_k - 1] = jj;
										SortVectorsDecreasing<double>(dist_of_neighbors_per_tree[ii].data(), neighbors_per_tree[ii].data(), size_smaller_k);
									}
								}
							}
						}
						if (neighbors_per_tree.size() == 1) {
							neighbors[i - start_at] = neighbors_per_tree[0];
						} 
						else {
							std::set<std::tuple<double, int, int, int>> set_tuples;
							for (int ii = 0; ii < neighbors_per_tree.size(); ii++) {
								set_tuples.insert({ dist_of_neighbors_per_tree[ii][0], ii, 0, neighbors_per_tree[ii][0] });
							}
							int index_of_vector;
							int index_in_vector;
							for (int ii = 0; ii < num_neighbors; ii++) {
								auto it = set_tuples.begin();
								neighbors[i - start_at][ii] = std::get<3>(*it);
								index_of_vector = std::get<1>(*it);
								index_in_vector = std::get<2>(*it);
								set_tuples.erase(it);
								if (index_in_vector < dist_of_neighbors_per_tree[index_of_vector].size() - 1) {
									set_tuples.insert({ dist_of_neighbors_per_tree[index_of_vector][index_in_vector + 1], index_of_vector, index_in_vector + 1, neighbors_per_tree[index_of_vector][index_in_vector + 1] });
								}
							}
						}
						/*if (i == 1000 || i == 10010) {
							for (int ii = 0; ii < num_neighbors; ii++) {
								Log::REInfo("knns %i %i %g %g", ii, neighbors[i - start_at][ii],
									coords.coeffRef(neighbors[i - start_at][ii], 0),
									coords.coeffRef(neighbors[i - start_at][ii], 1));
							}
						}*/
					}
					else {
						find_kNN_CoverTree<T_mat>(i, num_neighbors, level, save_distances, base,
							coords, chol_ip_cross_cov, corr_diag, re_comps_resid_cluster_i,
							neighbors[i - start_at], dist_dummy, cover_trees[0], dist_function);
					}
					//Log::REInfo("Rep %i %i",i, (int)std::pow(2, multiplicator-1));
					//Save distances between points and neighbors
					if (save_distances) {
						dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
					}
					for (int j = 0; j < num_nearest_neighbors; ++j) {
						double dij = (coords(i, Eigen::all) - coords(neighbors[i - start_at][j], Eigen::all)).lpNorm<2>();
						if (save_distances) {
							dist_obs_neighbors[i - start_at](j, 0) = dij;
						}
						if (check_has_duplicates && !has_duplicates) {
							if (dij < EPSILON_NUMBERS) {
#pragma omp critical
								{
									has_duplicates = true;
								}
							}
						}//end check_has_duplicates
					}
				}
			}
		}
		/*int i = 10010;
		if (true && !prediction && end_search_at > i) {
			std::vector<double> distances_vec;
			vec_t ind_vec(num_data);
			den_mat_t coords_i;
			std::vector<int> indi{ i };
			coords_i = coords(indi, Eigen::all);
			for (int jj = 0; jj < i; ++jj) {
				double dij = 0.;
				den_mat_t dist_ij(1, 1);
				dist_ij(0, 0) = 0.;
				den_mat_t coords_j;
				den_mat_t corr_mat;
				std::vector<int> indj{ jj };
		//		coords_j = coords(indj, Eigen::all);
		//		re_comps_resid_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(dist_ij, coords_i, coords_j,
		//			corr_mat, corr_mat_deriv.data(), false, true, 1., false);
		//		dij = std::sqrt(1. - std::abs((corr_mat.value() -
		//			(chol_ip_cross_cov.col(i).dot(chol_ip_cross_cov.col(jj)))) /
		//			std::sqrt(corr_diag[i] * corr_diag[jj])));
				std::vector<int> indj{ jj };
				vec_t dist_ij(1);
				distances_funct<T_mat>(i, indj, coords, corr_diag, chol_ip_cross_cov,
					re_comps_resid_cluster_i, dist_ij, dist_function, save_distances);
				distances_vec.push_back((double)dist_ij[0]);
				ind_vec[jj] = jj;
			}
			std::vector<int> sort_vect(distances_vec.size());
			SortIndeces<double>(distances_vec, sort_vect);
			int num_k = 0;
			int ind_i = 0;
			while (num_k < num_neighbors) {
				if (sort_vect[ind_i] < i) {
					if (i == 10010) {
						Log::REInfo("knn true %i %i %g %g %g", ind_i, sort_vect[ind_i],
							coords.coeffRef(sort_vect[ind_i], 0),
							coords.coeffRef(sort_vect[ind_i], 1), distances_vec[sort_vect[ind_i]]);
					}
					num_k += 1;
				}
				ind_i += 1;
			}
		}*/
		// Calculate distances among neighbors
		int first_i = (start_at == 0) ? 1 : start_at;
#pragma omp parallel for schedule(static)
		for (int i = first_i; i < num_data; ++i) {
			int nn_i = (int)neighbors[i - start_at].size();
			if (save_distances) {
				dist_between_neighbors[i - start_at].resize(nn_i, nn_i);
			}
			for (int j = 0; j < nn_i; ++j) {
				if (save_distances) {
					dist_between_neighbors[i - start_at](j, j) = 0.;
				}
				den_mat_t coords_i;
				if (!save_distances) {
					std::vector<int> indi{ neighbors[i - start_at][j] };
					coords_i = coords(indi, Eigen::all);
				}
				for (int k = j + 1; k < nn_i; ++k) {
					den_mat_t dist_ij(1, 1);
					dist_ij(0, 0) = 0.;
					den_mat_t coords_j;
					den_mat_t corr_mat;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						if (!save_distances) {
							std::vector<int> indj{ neighbors[i - start_at][k] };
							coords_j = coords(indj, Eigen::all);
						}
						dist_ij(0,0) = (coords(neighbors[i - start_at][j], Eigen::all) - coords(neighbors[i - start_at][k], Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_between_neighbors[i - start_at](j, k) = dist_ij.value();
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dist_ij.value() < EPSILON_NUMBERS) {
#pragma omp critical
							{
								has_duplicates = true;
							}
						}
					}//end check_has_duplicates
				}
			}
			if (save_distances) {
				dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyLower>() = dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyUpper>().transpose();
			}
		}
		if (check_has_duplicates) {
			check_has_duplicates = has_duplicates;
		}
	}//end find_nearest_neighbors_Vecchia_FSA_fast

	/*!
	* \brief Finds the nearest_neighbors among the previous observations
	* \param dist Distance between all observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param[out] nearest_neighbor Vector with indices of nearest neighbors for every observations
	*/
	void find_nearest_neighbors_Vecchia(den_mat_t& dist,
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& neighbors);

	/*!
	* \brief Finds the nearest_neighbors among the previous observations using the fast mean-distance-ordered nn search by Ra and Kim (1993)
	* \param coords Coordinates of observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param[out] neighbors Vector with indices of neighbors for every observations (length = num_data - start_at)
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation: distances between locations and their neighbors (length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation: distances between all neighbors (length = num_data - start_at)
	* \param start_at Index of first point for which neighbors should be found (useful for prediction, otherwise = 0)
	* \param end_search_at Index of last point which can be a neighbor (useful for prediction when the neighbors are only to be found among the observed data, otherwise = num_data - 1 (if end_search_at < 0, we set end_search_at = num_data - 1)
	* \param[out] check_has_duplicates If true, it is checked whether there are duplicates in coords among the neighbors (result written on output)
	* \param neighbor_selection The way how neighbors are selected
	* \param gen RNG
	* \param save_distances If true, distances are saved in dist_obs_neighbors and dist_between_neighbors
	*/
	void find_nearest_neighbors_Vecchia_fast(const den_mat_t& coords,
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& neighbors,
		std::vector<den_mat_t>& dist_obs_neighbors,
		std::vector<den_mat_t>& dist_between_neighbors,
		int start_at,
		int end_search_at,
		bool& check_has_duplicates,
		const string_t& neighbor_selection,
		RNG_t& gen,
		bool save_distances);

	void find_nearest_neighbors_fast_internal(const int i,
		const int num_data,
		const int num_nearest_neighbors,
		const int end_search_at,
		const int dim_coords,
		const den_mat_t& coords,
		const std::vector<int>& sort_sum,
		const std::vector<int>& sort_inv_sum,
		const std::vector<double>& coords_sum,
		std::vector<int>& neighbors_i,
		std::vector<double>& nn_square_dist);

	/*!
	* \brief Initialize individual component models and collect them in a containter when the Vecchia approximation is used
	* \param num_data Number of data points
	* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
	* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
	* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
	* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization
	* \param gp_coords_data Coordinates (features) for Gaussian process
	* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
	* \param[out] re_comps_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param[out] dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param[out] entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param[out] z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] only_one_GP_calculations_on_RE_scale
	* \param[out] has_duplicates_coords If true, there are duplicates in coords among the neighbors (currently only used for the Vecchia approximation for non-Gaussian likelihoods)
	* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param rng Random number generator
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param num_comps_total Total number of random effect components (grouped REs plus other GPs)
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param cov_fct Type of covariance function
	* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
	* \param cov_fct_taper_range Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
	* \param cov_fct_taper_shape Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
	* \param apply_tapering If true, tapering is applied to the covariance function (element-wise multiplication with a compactly supported Wendland correlation function)
	* \param chol_ip_cross_cov sigma_ip^-1/2 * sigma_cross_cov
	* \param re_comps_resid_cluster_i Container that collects the individual component models
	* \param gp_approx Gaussian process approximation
	* \param clusters index of the inducing point each data point is closest to
	*/
	template<typename T_mat>
	void CreateREComponentsVecchia(data_size_t num_data,
		int dim_gp_coords,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
		data_size_t cluster_i,
		std::map<data_size_t, int>& num_data_per_cluster,
		const double* gp_coords_data,
		const double* gp_rand_coef_data,
		std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
		std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
		bool& only_one_GP_calculations_on_RE_scale,
		bool& has_duplicates_coords,
		string_t vecchia_ordering,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		bool check_has_duplicates,
		RNG_t& rng,
		int num_gp_rand_coef,
		int num_gp_total,
		int num_comps_total,
		bool gauss_likelihood,
		string_t cov_fct,
		double cov_fct_shape,
		double cov_fct_taper_range,
		double cov_fct_taper_shape,
		bool apply_tapering,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		string_t& gp_approx,
		const vec_t& clusters,
		double base) {
		int ind_intercept_gp = (int)re_comps_cluster_i.size();
		if ((vecchia_ordering == "random" || vecchia_ordering == "time_random_space") && gp_approx != "full_scale_vecchia") {
			std::shuffle(data_indices_per_cluster[cluster_i].begin(), data_indices_per_cluster[cluster_i].end(), rng);
		}
		std::vector<double> gp_coords;
		for (int j = 0; j < dim_gp_coords; ++j) {
			for (const auto& id : data_indices_per_cluster[cluster_i]) {
				gp_coords.push_back(gp_coords_data[j * num_data + id]);
			}
		}
		den_mat_t gp_coords_mat = Eigen::Map<den_mat_t>(gp_coords.data(), num_data_per_cluster[cluster_i], dim_gp_coords);
		if (vecchia_ordering == "time" || vecchia_ordering == "time_random_space") {
			std::vector<double> coord_time(gp_coords_mat.rows());
#pragma omp for schedule(static)
			for (int i = 0; i < (int)gp_coords_mat.rows(); ++i) {
				coord_time[i] = gp_coords_mat.coeff(i, 0);
			}
			std::vector<int> sort_time;
			SortIndeces<double>(coord_time, sort_time);
			den_mat_t gp_coords_mat_not_sort = gp_coords_mat;
			gp_coords_mat = gp_coords_mat_not_sort(sort_time, Eigen::all);
			gp_coords_mat_not_sort.resize(0, 0);
			std::vector<int> dt_idx_unsorted = data_indices_per_cluster[cluster_i];
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)gp_coords_mat.rows(); ++i) {
				data_indices_per_cluster[cluster_i][i] = dt_idx_unsorted[sort_time[i]];
			}
		}
		only_one_GP_calculations_on_RE_scale = num_gp_total == 1 && num_comps_total == 1 && !gauss_likelihood;
		re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
			gp_coords_mat,
			cov_fct,
			cov_fct_shape,
			cov_fct_taper_range,
			cov_fct_taper_shape,
			apply_tapering,
			false,
			false,
			only_one_GP_calculations_on_RE_scale,
			only_one_GP_calculations_on_RE_scale)));
		std::shared_ptr<RECompGP<T_mat>> re_comp;
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_cluster_i[ind_intercept_gp]);
		}
		else {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		}
		if (re_comp->GetNumUniqueREs() == num_data_per_cluster[cluster_i]) {
			only_one_GP_calculations_on_RE_scale = false;
		}
		bool has_duplicates = check_has_duplicates;
		nearest_neighbors_cluster_i = std::vector<std::vector<int>>(re_comp->GetNumUniqueREs());
		dist_obs_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		dist_between_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		if (vecchia_neighbor_selection != "residual_correlation") {
			find_nearest_neighbors_Vecchia_fast(re_comp->GetCoords(), re_comp->GetNumUniqueREs(), num_neighbors,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, has_duplicates,
				vecchia_neighbor_selection, rng, re_comp->ShouldSaveDistances());
		}
		else {
			has_duplicates = false;
			den_mat_t coords = re_comp->GetCoords();
			bool save_distances = re_comp->ShouldSaveDistances();
			//Intialize neighbor vectors
			for (int i = 0; i < num_data; ++i) {
				if (i > 0 && i <= num_neighbors) {
					nearest_neighbors_cluster_i[i].resize(i);
					if (save_distances) {
						dist_obs_neighbors_cluster_i[i].resize(i, 1);
					}
					for (int j = 0; j < i; ++j) {
						nearest_neighbors_cluster_i[i][j] = j;
						den_mat_t dist_ij(1, 1);
						dist_ij(0, 0) = 0.;
						if (save_distances || (check_has_duplicates && !has_duplicates)) {
							dist_ij(0, 0) = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
						}
						if (save_distances) {
							dist_obs_neighbors_cluster_i[i](j, 0) = dist_ij.value();
						}
						if (check_has_duplicates && !has_duplicates) {
							if (dist_ij.value() < EPSILON_NUMBERS) {
								has_duplicates = true;
							}
						}//end check_has_duplicates
					}
				}
				else if (i > num_neighbors) {
					nearest_neighbors_cluster_i[i].resize(num_neighbors);
				}
			}
		}
		if ((vecchia_ordering == "time" || vecchia_ordering == "time_random_space") && !(re_comp->IsSpaceTimeModel())) {
			Log::REFatal("'vecchia_ordering' is '%s' but the 'cov_function' is not a space-time covariance function ", vecchia_ordering.c_str());
		}
		if (check_has_duplicates) {
			has_duplicates_coords = has_duplicates_coords || has_duplicates;
			if (!gauss_likelihood && has_duplicates_coords) {
				Log::REFatal("Duplicates found in the coordinates for the Gaussian process. "
					"This is currently not supported for the Vecchia approximation for non-Gaussian likelihoods ");
			}
		}
		for (int i = 0; i < re_comp->GetNumUniqueREs(); ++i) {
			for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
				entries_init_B_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
				entries_init_B_grad_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
			}
			entries_init_B_cluster_i.push_back(Triplet_t(i, i, 1.));//Put 1's on the diagonal since B = I - A
		}
		//Random coefficients
		if (num_gp_rand_coef > 0) {
			if (!(re_comp->ShouldSaveDistances())) {
				Log::REFatal("Random coefficient processes are not supported for covariance functions "
					"for which the neighbors are dynamically determined based on correlations");
			}
			z_outer_z_obs_neighbors_cluster_i = std::vector<std::vector<den_mat_t>>(re_comp->GetNumUniqueREs());
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data;
				for (const auto& id : data_indices_per_cluster[cluster_i]) {
					rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
				}
				re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
					rand_coef_data,
					cov_fct,
					cov_fct_shape,
					cov_fct_taper_range,
					cov_fct_taper_shape,
					re_comp->GetTaperMu(),
					apply_tapering,
					false,
					dim_gp_coords)));
				//save random coefficient data in the form ot outer product matrices
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_per_cluster[cluster_i]; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
					int dim_z = (i == 0) ? 1 : ((int)nearest_neighbors_cluster_i[i].size() + 1);
					vec_t coef_vec(dim_z);
					coef_vec(0) = rand_coef_data[i];
					if (i > 0) {
						for (int ii = 1; ii < dim_z; ++ii) {
							coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
						}
					}
					z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
				}
			}
		}// end random coefficients
	}//end CreateREComponentsVecchia

	/*!
	* \brief Update the nearest neighbors based on scaled coordinates
	* \param[out] re_comps_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param[out] entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param rng Random number generator
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param gp_approx Gaussian process approximation
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param re_comps_resid_cluster_i Container that collects the individual component models
	* \param clusters index of the inducing point each data point is closest to
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation: distances between locations and their neighbors (length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation: distances between all neighbors (length = num_data - start_at)
	*/
	template<typename T_mat>
	void UpdateNearestNeighbors(std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		RNG_t& rng,
		int ind_intercept_gp,
		string_t& gp_approx,
		den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		const vec_t& clusters,
		double base,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i) {
		std::shared_ptr<RECompGP<T_mat>> re_comp;
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_cluster_i[ind_intercept_gp]);
		}
		else if (gp_approx == "full_scale_vecchia"){
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		}
		else {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
			CHECK(re_comp->ShouldSaveDistances() == false);
		}
		int num_re = re_comp->GetNumUniqueREs();
		CHECK((int)nearest_neighbors_cluster_i.size() == num_re);
		std::vector<den_mat_t> dist_dummy;
		bool check_has_duplicates = false;
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			find_nearest_neighbors_Vecchia_FSA_fast<T_mat>(re_comp->GetCoords(), num_re, num_neighbors, chol_ip_cross_cov,
				re_comps_resid_cluster_i, nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates, rng, re_comp->ShouldSaveDistances(),
				base, false ,false, num_re);
		}
		else {
			find_nearest_neighbors_Vecchia_fast(re_comp->GetCoords(), num_re, num_neighbors,
				nearest_neighbors_cluster_i, dist_dummy, dist_dummy, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, false);
		}
		int ctr = 0, ctr_grad = 0;
		for (int i = 0; i < std::min(num_re, num_neighbors); ++i) {
			for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
				entries_init_B_cluster_i[ctr] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				entries_init_B_grad_cluster_i[ctr_grad] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				ctr++;
				ctr_grad++;
			}
			entries_init_B_cluster_i[ctr] = Triplet_t(i, i, 1.);//Put 1's on the diagonal since B = I - A
			ctr++;
		}
		if (num_neighbors < num_re) {
#pragma omp parallel for schedule(static)
			for (int i = num_neighbors; i < num_re; ++i) {
				CHECK((int)nearest_neighbors_cluster_i[i].size() == num_neighbors);
				for (int j = 0; j < num_neighbors; ++j) {
					entries_init_B_cluster_i[ctr + (i - num_neighbors) * (num_neighbors + 1) + j] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
					entries_init_B_grad_cluster_i[ctr_grad + (i - num_neighbors) * num_neighbors + j] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				}
				entries_init_B_cluster_i[ctr + (i - num_neighbors) * (num_neighbors + 1) + num_neighbors] = Triplet_t(i, i, 1.);//Put 1's on the diagonal since B = I - A
			}
		}
	}//end UpdateNearestNeighbors

	/*!
	* \brief Calculate matrices A and D_inv as well as their derivatives for the Vecchia approximation for one cluster (independent realization of GP)
	* \param num_re_cluster_i Number of random effects
	* \param calc_gradient If true, the gradient also be calculated (only for Vecchia approximation)
	* \param re_comps_cluster_i Container that collects the individual component models
	* \param nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] B_cluster_i Matrix A = I - B (= Cholesky factor of inverse covariance) for Vecchia approximation
	* \param[out] D_inv_cluster_i Diagonal matrices D^-1 for Vecchia approximation
	* \param[out] B_grad_cluster_i Derivatives of matrices A ( = derivative of matrix -B) for Vecchia approximation
	* \param[out] D_grad_cluster_i Derivatives of matrices D for Vecchia approximation
	* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true
	* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
	* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	*/
	template<typename T_mat>
	void CalcCovFactorVecchia(data_size_t num_re_cluster_i,
		bool calc_gradient,
		const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_ip_cluster_i,
		const den_mat_t& chol_ip_cross_cov_cluster_i,
		const std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
		const std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
		sp_mat_t& B_cluster_i,
		sp_mat_t& D_inv_cluster_i,
		std::vector<sp_mat_t>& B_grad_cluster_i,
		std::vector<sp_mat_t>& D_grad_cluster_i,
		den_mat_t& sigma_ip_inv_cross_cov_T_cluster_i,
		std::vector<den_mat_t>& sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i,
		bool transf_scale,
		double nugget_var,
		bool calc_gradient_nugget,
		int num_gp_total,
		int ind_intercept_gp,
		bool gauss_likelihood,
		string_t& gp_approx,
		const string_t& vecchia_neighbor_selection) {
		Log::REInfo("CalcCovFactorVecchia");
		Log::REInfo("start");//only for debugging
		std::chrono::steady_clock::time_point begin, end;//only for debugging
		double el_time;//only for debugging
		begin = std::chrono::steady_clock::now();//only for debugging
		int num_par_comp = re_comps_cluster_i[ind_intercept_gp]->NumCovPar();
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			num_par_comp = re_comps_ip_cluster_i[ind_intercept_gp]->NumCovPar();
		}
		int num_par_gp = num_par_comp * num_gp_total + calc_gradient_nugget;
		//Initialize matrices B = I - A and D^-1 as well as their derivatives (in order that the code below can be run in parallel)
		B_cluster_i = sp_mat_t(num_re_cluster_i, num_re_cluster_i);//B = I - A
		B_cluster_i.setFromTriplets(entries_init_B_cluster_i.begin(), entries_init_B_cluster_i.end());//Note: 1's are put on the diagonal
		D_inv_cluster_i = sp_mat_t(num_re_cluster_i, num_re_cluster_i);//D^-1. Note: we first calculate D, and then take the inverse below
		D_inv_cluster_i.setIdentity();//Put 1's on the diagonal for nugget effect (entries are not overriden but added below)
		if (!transf_scale && gauss_likelihood) {
			D_inv_cluster_i.diagonal().array() = nugget_var;//nugget effect is not 1 if not on transformed scale
		}
		if (!gauss_likelihood) {
			D_inv_cluster_i.diagonal().array() = 0.;
		}
		bool exclude_marg_var_grad = !gauss_likelihood && (re_comps_cluster_i.size() == 1) && !(gp_approx == "full_scale_vecchia");//gradient is not needed if there is only one GP for non-Gaussian likelihoods
		if (calc_gradient) {
			B_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of B = derviateive of (-A)
			D_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of D
			for (int ipar = 0; ipar < num_par_gp; ++ipar) {
				if (!(exclude_marg_var_grad && ipar == 0)) {
					B_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					B_grad_cluster_i[ipar].setFromTriplets(entries_init_B_grad_cluster_i.begin(), entries_init_B_grad_cluster_i.end());
					D_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					D_grad_cluster_i[ipar].setIdentity();//Put 0 on the diagonal
					D_grad_cluster_i[ipar].diagonal().array() = 0.;
				}
			}
		}//end initialization
		std::shared_ptr<RECompGP<T_mat>> re_comp;
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_cluster_i[ind_intercept_gp]);
		}
		else {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		}
		//std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
		// Components for full scale vecchia
		den_mat_t sigma_cross_cov;
		// Components for gradient of full scale vecchia
		//std::vector<den_mat_t> sigma_cross_cov_grad((int)num_par_comp);
		std::vector<den_mat_t> sigma_cross_cov_gradT((int)num_par_comp);
		std::vector<den_mat_t> sigma_ip_grad((int)num_par_comp);
		std::vector<den_mat_t> sigma_ip_grad_sigma_ip_inv_cross_cov_T((int)num_par_comp);
		//std::vector<den_mat_t> sigma_ip_Ihalf_sigma_cross_cov_gradT((int)num_par_comp);
		//std::vector<den_mat_t> sigma_ip_inv_sigma_ip_grad((int)num_par_comp);
		//std::vector<den_mat_t> chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv((int)num_par_comp);
		//std::vector<den_mat_t> chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv_cross_cov((int)num_par_comp);
		if (gp_approx == "full_scale_vecchia") {
			sigma_cross_cov = *(re_comps_cross_cov_cluster_i[0]->GetZSigmaZt());
			if (calc_gradient) {
				sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i = std::vector<den_mat_t>(num_par_gp);
				sigma_ip_inv_cross_cov_T_cluster_i = chol_fact_sigma_ip_cluster_i.solve(sigma_cross_cov.transpose());
#pragma omp parallel for schedule(static)
				for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
					sigma_ip_grad[ipar] = *(re_comps_ip_cluster_i[0]->GetZSigmaZtGrad(ipar, true, re_comps_ip_cluster_i[0]->CovPars()[0]));
					//sigma_ip_inv_sigma_ip_grad[ipar] = chol_fact_sigma_ip_cluster_i.solve(sigma_ip_grad[ipar]);
					//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
					//	sigma_ip_inv_sigma_ip_grad[ipar].transpose(), chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv[ipar], false);
					//sigma_cross_cov_grad[ipar] = *(re_comps_cross_cov_cluster_i[0]->GetZSigmaZtGrad(ipar, true, re_comps_cross_cov_cluster_i[0]->CovPars()[0]));
					sigma_cross_cov_gradT[ipar] = (*(re_comps_cross_cov_cluster_i[0]->GetZSigmaZtGrad(ipar, true, re_comps_cross_cov_cluster_i[0]->CovPars()[0]))).transpose();
					sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar] = sigma_ip_grad[ipar] * sigma_ip_inv_cross_cov_T_cluster_i;
					//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
					//	(sigma_cross_cov_grad[ipar]).transpose(), sigma_ip_Ihalf_sigma_cross_cov_gradT[ipar], false);
					//chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv_cross_cov[ipar] = chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv[ipar] * sigma_cross_cov.transpose();
				}
			}
		}
		end = std::chrono::steady_clock::now();//only for debugging
		el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
		Log::REInfo(" time until0 = %g ", el_time);
#pragma omp parallel for schedule(static)
		for (data_size_t i = 0; i < num_re_cluster_i; ++i) {
			if (gp_approx == "full_scale_vecchia") {
				D_inv_cluster_i.coeffRef(i, i) -= chol_ip_cross_cov_cluster_i.col(i).array().square().sum();
			}
			std::vector<int> nearest_neighbors_cluster_i_i = nearest_neighbors_cluster_i[i];
			int num_nn = (int)nearest_neighbors_cluster_i_i.size();
			//calculate covariance matrices between observations and neighbors and among neighbors as well as their derivatives
			den_mat_t cov_mat_obs_neighbors;
			den_mat_t cov_mat_between_neighbors;
			std::vector<den_mat_t> cov_grad_mats_obs_neighbors(num_par_gp);//covariance matrix plus derivative wrt to every parameter
			std::vector<den_mat_t> cov_grad_mats_between_neighbors(num_par_gp);
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				for (int j = 0; j < num_gp_total; ++j) {
					int ind_first_par = j * num_par_comp;//index of first parameter (variance) of component j in gradient vectors
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							re_comp->GetSubSetCoords(ind, coords_i);
							re_comp->GetSubSetCoords(nearest_neighbors_cluster_i_i, coords_nn_i);
						}
						if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
							re_comps_resid_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
								cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors.data() + ind_first_par,
								calc_gradient, transf_scale, nugget_var, false);//write on matrices directly for first GP component
							re_comps_resid_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
								cov_mat_between_neighbors, cov_grad_mats_between_neighbors.data() + ind_first_par,
								calc_gradient, transf_scale, nugget_var, true);
						}
						else {
							re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
								cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors.data() + ind_first_par,
								calc_gradient, transf_scale, nugget_var, false);//write on matrices directly for first GP component
							re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
								cov_mat_between_neighbors, cov_grad_mats_between_neighbors.data() + ind_first_par,
								calc_gradient, transf_scale, nugget_var, true);
						}
						// Residual process of full-scale Vecchia approximation
						if (gp_approx == "full_scale_vecchia") {
							//std::vector<int> ind_obs{ i };
							// Cross-covariance neighbors and inducing points
							//den_mat_t sigma_ip_Ihalf_sigma_cross_covT_obs = chol_ip_cross_cov_cluster_i(Eigen::all, ind_obs);
							//den_mat_t sigma_ip_Ihalf_sigma_cross_covT_neighbors = chol_ip_cross_cov_cluster_i(Eigen::all, nearest_neighbors_cluster_i_i);
							vec_t sigma_ip_Ihalf_sigma_cross_covT_obs = chol_ip_cross_cov_cluster_i.col(i);
#pragma omp parallel for schedule(static)
							for (int ii = 0; ii < num_nn; ++ii) {
								cov_mat_obs_neighbors.coeffRef(ii,0) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_ip_Ihalf_sigma_cross_covT_obs);
#pragma omp parallel for schedule(static)
								for (int jj = ii; jj < num_nn; ++jj) {
									if (ii == jj) {
										cov_mat_between_neighbors.coeffRef(ii, jj) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).array().square().sum();
									}
									else {
										cov_mat_between_neighbors.coeffRef(ii, jj) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).dot(chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i_i[jj]));
										cov_mat_between_neighbors.coeffRef(jj, ii) = cov_mat_between_neighbors.coeffRef(ii, jj);
									}
								}
							}
							//cov_mat_obs_neighbors -= sigma_ip_Ihalf_sigma_cross_covT_neighbors.transpose() * sigma_ip_Ihalf_sigma_cross_covT_obs;
							//cov_mat_between_neighbors -= sigma_ip_Ihalf_sigma_cross_covT_neighbors.transpose() * sigma_ip_Ihalf_sigma_cross_covT_neighbors;
							// Gradient
							if (calc_gradient) {
								vec_t sigma_ip_I_sigma_cross_covT_obs = sigma_ip_inv_cross_cov_T_cluster_i.col(i);
#pragma omp parallel for schedule(static)
								for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
									//vec_t sigma_ip_Ihalf_sigma_cross_cov_gradT_obs = (sigma_ip_Ihalf_sigma_cross_cov_gradT[ipar]).col(i);
									vec_t sigma_cross_cov_gradT_obs = sigma_cross_cov_gradT[ipar].col(i);
									vec_t sigma_ip_grad_sigma_ip_inv_cross_cov_T_obs = sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i);
#pragma omp parallel for schedule(static)
									for (int ii = 0; ii < num_nn; ++ii) {
										//cov_grad_mats_obs_neighbors[ind_first_par + ipar].coeffRef(ii, 0) -= (sigma_ip_Ihalf_sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_ip_Ihalf_sigma_cross_covT_obs) +
										//	chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_ip_Ihalf_sigma_cross_cov_gradT_obs - chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv_cross_cov_obs);
										cov_grad_mats_obs_neighbors[ind_first_par + ipar].coeffRef(ii, 0) -= (sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_ip_I_sigma_cross_covT_obs) +
											sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_cross_cov_gradT_obs - sigma_ip_grad_sigma_ip_inv_cross_cov_T_obs);
#pragma omp parallel for schedule(static)
										for (int jj = ii; jj < num_nn; ++jj) {
											cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(ii, jj) -= (sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i_i[ii]).dot(sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i_i[jj])) +
												sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i_i[ii]).dot((sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i_i[jj]) -
													sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(nearest_neighbors_cluster_i_i[jj]));
											if (ii != jj) {
												cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(jj, ii) = cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(ii, jj);
											}
										}
									}
									//den_mat_t sigma_ip_Ihalf_sigma_cross_cov_gradT_neighbors = (sigma_ip_Ihalf_sigma_cross_cov_gradT[ipar])(Eigen::all, nearest_neighbors_cluster_i_i);
									//den_mat_t sigma_ip_Ihalf_sigma_cross_cov_gradT_obs = (sigma_ip_Ihalf_sigma_cross_cov_gradT[ipar])(Eigen::all, ind_obs);
									//cov_grad_mats_obs_neighbors[ind_first_par + ipar] -= sigma_ip_Ihalf_sigma_cross_cov_gradT_neighbors.transpose() * sigma_ip_Ihalf_sigma_cross_covT_obs +
									//	sigma_ip_Ihalf_sigma_cross_covT_neighbors.transpose() * sigma_ip_Ihalf_sigma_cross_cov_gradT_obs - sigma_ip_Ihalf_sigma_cross_covT_neighbors.transpose() * (chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv_cross_cov[ipar](Eigen::all, ind_obs));
									//den_mat_t sigma_grad_n_n = sigma_ip_Ihalf_sigma_cross_cov_gradT_neighbors.transpose() * sigma_ip_Ihalf_sigma_cross_covT_neighbors;
									//cov_grad_mats_between_neighbors[ind_first_par + ipar] -= sigma_grad_n_n + sigma_grad_n_n.transpose() - 
									//	sigma_ip_Ihalf_sigma_cross_covT_neighbors.transpose() * (chol_sigma_ip_inv_sigma_ip_grad_sigma_ip_inv_cross_cov[ipar](Eigen::all, nearest_neighbors_cluster_i_i));
								}
							}
						}
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false);
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();//cov_mat_obs_neighbors_j.cwiseProduct()
						//cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();//cov_mat_obs_neighbors_j.cwiseProduct()//DELETE_FIRST
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
						if (calc_gradient) {
							for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
								cov_grad_mats_obs_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
								cov_grad_mats_between_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
							}
						}
					}
				}//end loop over components j
			}//end if(i>1)
			//Calculate matrices B and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii}) and its derivatives
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps_cluster_i[ind_intercept_gp + j]->CovPars()[0];
				if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
					d_comp_j = re_comps_ip_cluster_i[ind_intercept_gp + j]->CovPars()[0];
				}
				if (!transf_scale && gauss_likelihood) {
					d_comp_j *= nugget_var;
				}
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				D_inv_cluster_i.coeffRef(i, i) += d_comp_j;
				if (calc_gradient) {
					if (!(exclude_marg_var_grad && j == 0)) {
						if (transf_scale) {
							D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = d_comp_j;//derivative of the covariance function wrt the variance. derivative of the covariance function wrt to range is zero on the diagonal
						}
						else {
							if (j == 0) {
								D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = 1.;//1's on the diagonal on the orignal scale
							}
							else {
								D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
							}
						}
					}
				}
			}
			if (calc_gradient && calc_gradient_nugget) {
				D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) = 1.;
			}
			//2. remaining terms
			if (i > 0) {
				if (gauss_likelihood) {
					if (transf_scale) {
						cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
					}
					else {
						cov_mat_between_neighbors.diagonal().array() += nugget_var;
					}
				}
				else {
					cov_mat_between_neighbors.diagonal().array() += EPSILON_ADD_COVARIANCE_STABLE;//Avoid numerical problems when there is no nugget effect
				}
				den_mat_t A_i(1, num_nn);
				den_mat_t A_i_grad_sigma2;
				Eigen::LLT<den_mat_t> chol_fact_between_neighbors = cov_mat_between_neighbors.llt();
				A_i = (chol_fact_between_neighbors.solve(cov_mat_obs_neighbors)).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					B_cluster_i.coeffRef(i, nearest_neighbors_cluster_i_i[inn]) = -A_i(0, inn);
				}
				D_inv_cluster_i.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors)(0, 0);
				if (calc_gradient) {
					if (calc_gradient_nugget) {
						A_i_grad_sigma2 = -(chol_fact_between_neighbors.solve(A_i.transpose())).transpose();
					}
					den_mat_t A_i_grad(1, num_nn);
					for (int j = 0; j < num_gp_total; ++j) {
						int ind_first_par = j * num_par_comp;
#pragma omp parallel for schedule(static)
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							if (!(exclude_marg_var_grad && ipar == 0)) {
								A_i_grad = (chol_fact_between_neighbors.solve(cov_grad_mats_obs_neighbors[ind_first_par + ipar])).transpose() -
									A_i * ((chol_fact_between_neighbors.solve(cov_grad_mats_between_neighbors[ind_first_par + ipar])).transpose());
								for (int inn = 0; inn < num_nn; ++inn) {
									B_grad_cluster_i[ind_first_par + ipar].coeffRef(i, nearest_neighbors_cluster_i_i[inn]) = -A_i_grad(0, inn);
								}
								if (ipar == 0) {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= ((A_i_grad * cov_mat_obs_neighbors)(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar])(0, 0));//add to derivative of diagonal elements for marginal variance 
								}
								else {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) = -((A_i_grad * cov_mat_obs_neighbors)(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar])(0, 0));//don't add to existing values since derivative of diagonal is zero for range
								}
								if (gp_approx == "full_scale_vecchia") {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= sigma_ip_inv_cross_cov_T_cluster_i.col(i).dot(2 * sigma_cross_cov_gradT[ipar].col(i) -
										sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i));
								}
							}
						}
					}
					if (calc_gradient_nugget) {
						for (int inn = 0; inn < num_nn; ++inn) {
							B_grad_cluster_i[num_par_gp - 1].coeffRef(i, nearest_neighbors_cluster_i_i[inn]) = -A_i_grad_sigma2(0, inn);
						}
						D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) -= (A_i_grad_sigma2 * cov_mat_obs_neighbors)(0, 0);
					}
				}//end calc_gradient
			}//end if i > 0;
			if (i == 0 && calc_gradient) {
				if (gp_approx == "full_scale_vecchia") {
					for (int j = 0; j < num_gp_total; ++j) {
						int ind_first_par = j * num_par_comp;
#pragma omp parallel for schedule(static)
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							if (!(exclude_marg_var_grad && ipar == 0)) {
								D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= sigma_ip_inv_cross_cov_T_cluster_i.col(i).dot(2 * sigma_cross_cov_gradT[ipar].col(i) -
									sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i));
							}
						}
					}
				}
			}
			D_inv_cluster_i.coeffRef(i, i) = 1. / D_inv_cluster_i.coeffRef(i, i);
		}//end loop over data i
		Eigen::Index minRow, minCol;
		double min_D_inv = D_inv_cluster_i.diagonal().minCoeff(&minRow, &minCol);
		if (min_D_inv <= 0.) {
			const char* min_D_inv_below_zero_msg = "The matrix D in the Vecchia approximation contains negative or zero values. "
				"This is a serious problem that likely results from numerical instabilities ";
			if (gauss_likelihood) {
				Log::REWarning(min_D_inv_below_zero_msg);
			}
			else {
				Log::REFatal(min_D_inv_below_zero_msg);
			}
		}
		end = std::chrono::steady_clock::now();//only for debugging
		el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;//only for debugging
		Log::REInfo(" time until = %g ", el_time);
	}//end CalcCovFactorVecchia

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable process when observed locations appear first in the ordering
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_pred Total number of prediction locations (over all clusters)
	* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param gp_rand_coef_data_pred Random coefficient data for GPs
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param y_cluster_i Reponse variable data
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean (only for Gaussian likelihoods)
	* \param[out] pred_cov Predictive covariance matrix (only for Gaussian likelihoods)
	* \param[out] pred_var Predictive variances (only for Gaussian likelihoods)
	* \param[out] Bpo Lower left part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the observed locations (only for non-Gaussian likelihoods)
	* \param[out] Bp Lower right part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the prediction locations (only for non-Gaussian likelihoods)
	* \param[out] Dp Diagonal matrix with lower right part of matrix D in joint Vecchia approximation for observed and prediction locations (only for non-Gaussian likelihoods)
	*/
	template<typename T_mat>
	void CalcPredVecchiaObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
		int num_data_pred,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_ip_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_woodbury_cluster_i,
		den_mat_t& cross_cov_pred_ip,
		const sp_mat_rm_t& B_cluster_i,
		const sp_mat_rm_t& Bt_D_inv_cluster_i,
		const vec_t& y_aux_cluster_i,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		const den_mat_t& gp_coords_mat_ip,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<T_mat>>>>& re_comps_resid,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		bool gauss_likelihood,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		sp_mat_t& Bpo,
		sp_mat_t& Bp,
		vec_t& Dp,
		string_t gp_approx,
		const double base) {
		data_size_t num_re_cli = re_comps[cluster_i][ind_intercept_gp]->GetNumUniqueREs();
		std::shared_ptr<RECompGP<T_mat>> re_comp;
		if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid[cluster_i][ind_intercept_gp]);
		}
		else {
			re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		}
		int num_re_pred_cli = (int)gp_coords_mat_pred.rows();
		//Find nearest neighbors
		den_mat_t coords_all(num_re_cli + num_re_pred_cli, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_re_pred_cli);
		bool check_has_duplicates = false;
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		// Components for full scale vecchia
		den_mat_t sigma_cross_cov, chol_ip_cross_cov_pred, chol_ip_cross_cov_obs, chol_ip_cross_cov_obs_pred, sigma_ip_inv_sigma_cross_cov, sigma_ip_inv_sigma_cross_cov_pred;
		// Cross-covariance between predictions and inducing points C_pm
		den_mat_t cov_mat_pred_id, cross_dist;
		std::shared_ptr<RECompGP<den_mat_t>> re_comp_cross_cov_cluster_i_pred_ip;
		// Components for gradient of full scale vecchia
		if (gp_approx == "full_scale_vecchia") {// TODO: change
			re_comp_cross_cov_cluster_i_pred_ip = std::dynamic_pointer_cast<RECompGP<den_mat_t>>(re_comps_cross_cov_cluster_i[0]);
			re_comp_cross_cov_cluster_i_pred_ip->AddPredCovMatrices(gp_coords_mat_ip, gp_coords_mat_pred, cross_cov_pred_ip,
				cov_mat_pred_id, true, false, true, nullptr, false, cross_dist);
			sigma_cross_cov = *(re_comps_cross_cov_cluster_i[0]->GetZSigmaZt());
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
				sigma_cross_cov.transpose(), chol_ip_cross_cov_obs, false);
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
				cross_cov_pred_ip.transpose(), chol_ip_cross_cov_pred, false);
			sigma_ip_inv_sigma_cross_cov = chol_fact_sigma_ip_cluster_i.solve(sigma_cross_cov.transpose());//TODO
			sigma_ip_inv_sigma_cross_cov_pred = chol_fact_sigma_ip_cluster_i.solve(cross_cov_pred_ip.transpose());//TODO
			if (vecchia_neighbor_selection == "residual_correlation") {
				chol_ip_cross_cov_obs_pred.resize(chol_ip_cross_cov_obs.rows(), chol_ip_cross_cov_obs.cols() + chol_ip_cross_cov_pred.cols());
				chol_ip_cross_cov_obs_pred.leftCols(chol_ip_cross_cov_obs.cols()) = chol_ip_cross_cov_obs;
				chol_ip_cross_cov_obs_pred.rightCols(chol_ip_cross_cov_pred.cols()) = chol_ip_cross_cov_pred;
			}
		}
		if (!distances_saved) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
		}
		if (CondObsOnly) {
			if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
				find_nearest_neighbors_Vecchia_FSA_fast<T_mat>(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred, chol_ip_cross_cov_obs_pred,
					re_comps_resid[cluster_i], nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli,
					num_re_cli - 1, check_has_duplicates, rng, distances_saved, base, true, false, (int)num_re_cli);
			}
			else {
				if (distances_saved) {
					find_nearest_neighbors_Vecchia_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred,
						nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, num_re_cli - 1, check_has_duplicates,
						vecchia_neighbor_selection, rng, distances_saved);
				}
				else {
					find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re_cli + num_re_pred_cli, num_neighbors_pred,
						nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, num_re_cli - 1, check_has_duplicates,
						vecchia_neighbor_selection, rng, distances_saved);
				}
			}
		}
		else {//find neighbors among both the observed and prediction locations
			if (!gauss_likelihood) {
				check_has_duplicates = true;
			}
			if (gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") {
				find_nearest_neighbors_Vecchia_FSA_fast<T_mat>(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred, chol_ip_cross_cov_obs_pred,
					re_comps_resid[cluster_i], nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli,
					-1, check_has_duplicates, rng, distances_saved, base, true, true, (int)num_re_cli);
			}
			else {
				if (distances_saved) {
					find_nearest_neighbors_Vecchia_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred,
						nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, -1, check_has_duplicates,
						vecchia_neighbor_selection, rng, distances_saved);
				}
				else {
					find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re_cli + num_re_pred_cli, num_neighbors_pred,
						nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, -1, check_has_duplicates,
						vecchia_neighbor_selection, rng, distances_saved);
				}
			}
			if (check_has_duplicates) {
				Log::REFatal("Duplicates found among training and test coordinates. "
					"This is not supported for predictions with a Vecchia approximation for non-Gaussian likelihoods "
					"when neighbors are selected among both training and test points ('_cond_all') ");
			}
		}
		//Random coefficients
		std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_re_pred_cli);
		if (num_gp_rand_coef > 0) {
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data = re_comps[cluster_i][ind_intercept_gp + j + 1]->RandCoefData();//First entries are the observed data, then the predicted data
				for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {//TODO: maybe do the following in parallel? (see CalcPredVecchiaPredictedFirstOrder)
					rand_coef_data.push_back(gp_rand_coef_data_pred[j * num_data_pred + id]);
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_re_pred_cli; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
					int dim_z = (int)nearest_neighbors_cluster_i[i].size() + 1;
					vec_t coef_vec(dim_z);
					coef_vec(0) = rand_coef_data[num_re_cli + i];
					if ((num_re_cli + i) > 0) {
						for (int ii = 1; ii < dim_z; ++ii) {
							coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
						}
					}
					z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
				}
			}
		}
		// Determine Triplet for initializing Bpo and Bp
		std::vector<Triplet_t> entries_init_Bpo, entries_init_Bp;
		for (int i = 0; i < num_re_pred_cli; ++i) {
			entries_init_Bp.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
			for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
				if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
					entries_init_Bpo.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
				}
				else {//nearest neighbor belongs to predicted data
					entries_init_Bp.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn] - num_re_cli, 0.));
				}
			}
		}
		Bpo = sp_mat_t(num_re_pred_cli, num_re_cli);
		Bp = sp_mat_t(num_re_pred_cli, num_re_pred_cli);
		Dp = vec_t(num_re_pred_cli);
		Bpo.setFromTriplets(entries_init_Bpo.begin(), entries_init_Bpo.end());//initialize matrices (in order that the code below can be run in parallel)
		Bp.setFromTriplets(entries_init_Bp.begin(), entries_init_Bp.end());
		if (gauss_likelihood) {
			Dp.setOnes();//Put 1 on the diagonal (for nugget effect if gauss_likelihood, see comment below on why we add the nugget effect variance irrespective of 'predict_response')
		}
		else {
			Dp.setZero();
		}
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_re_pred_cli; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			for (int j = 0; j < num_gp_total; ++j) {
				if (j == 0) {
					if (!distances_saved) {
						std::vector<int> ind{ num_re_cli + i };
						coords_i = coords_all(ind, Eigen::all);
						coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
					}
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
					// Residual process of full-scale Vecchia approximation
					if (gp_approx == "full_scale_vecchia") {
						std::vector<int> ind_pred{ i };
						// Cross-covariance neighbors and inducing points
						den_mat_t sigma_ip_inv_cross_cov_neighbors(chol_ip_cross_cov_obs.rows(), num_nn);
						for (int inn = 0; inn < num_nn; ++inn) {
							if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
								sigma_ip_inv_cross_cov_neighbors.col(inn) = chol_ip_cross_cov_obs.col(nearest_neighbors_cluster_i[i][inn]);
							}
							else {
								sigma_ip_inv_cross_cov_neighbors.col(inn) = chol_ip_cross_cov_pred.col(nearest_neighbors_cluster_i[i][inn] - num_re_cli);
							}
						}
						cov_mat_obs_neighbors -= sigma_ip_inv_cross_cov_neighbors.transpose() * chol_ip_cross_cov_pred(Eigen::all, ind_pred);
						cov_mat_between_neighbors -= sigma_ip_inv_cross_cov_neighbors.transpose() * sigma_ip_inv_cross_cov_neighbors;
					}
				}
				else {//random coefficient GPs
					den_mat_t cov_mat_obs_neighbors_j;
					den_mat_t cov_mat_between_neighbors_j;
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true);
					//multiply by coefficient matrix
					cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
					cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
					cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
					cov_mat_between_neighbors += cov_mat_between_neighbors_j;
				}
			}//end loop over components j
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps[cluster_i][ind_intercept_gp + j]->CovPars()[0];
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				Dp[i] += d_comp_j;
			}
			if (gp_approx == "full_scale_vecchia") {
				Dp[i] -= chol_ip_cross_cov_pred.col(i).array().square().sum();
			}
			//2. remaining terms
			if (gauss_likelihood) {
				cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
				//Note: we add the nugget effect variance irrespective of 'predict_response' since (i) this is numerically more stable and 
				//	(ii) otherwise we would have to add it only for the neighbors in the observed training data if predict_response == false
				//	If predict_response == false, the nugget variance is simply subtracted from the predictive covariance matrix later again.
			}
			den_mat_t A_i(1, num_nn);//dim = 1 x nn
			A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
			for (int inn = 0; inn < num_nn; ++inn) {
				if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
					Bpo.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
				}
				else {
					Bp.coeffRef(i, nearest_neighbors_cluster_i[i][inn] - num_re_cli) -= A_i(0, inn);
				}
			}
			Dp[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
		}//end loop over data i
		// row-major
		sp_mat_rm_t Bpo_rm = sp_mat_rm_t(Bpo);
		sp_mat_rm_t Bp_rm = sp_mat_rm_t(Bp);
		if (gauss_likelihood) {
			if (gp_approx == "full_scale_vecchia") {
				pred_mean = -Bpo_rm * (y_cluster_i - sigma_cross_cov * chol_fact_sigma_woodbury_cluster_i.solve(sigma_cross_cov.transpose() * (Bt_D_inv_cluster_i * (B_cluster_i * y_cluster_i))));
				if (!CondObsOnly) {
					sp_L_solve(Bp.valuePtr(), Bp.innerIndexPtr(), Bp.outerIndexPtr(), num_re_pred_cli, pred_mean.data());
				}
				pred_mean += cross_cov_pred_ip * chol_fact_sigma_ip_cluster_i.solve(sigma_cross_cov.transpose() * y_aux_cluster_i);
				if (calc_pred_cov || calc_pred_var) {
					den_mat_t Vecchia_cross_cov(sigma_cross_cov.rows(), sigma_cross_cov.cols());
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < sigma_cross_cov.cols(); ++i) {
						Vecchia_cross_cov.col(i) = Bt_D_inv_cluster_i * (B_cluster_i * sigma_cross_cov.col(i));
					}
					den_mat_t cross_cov_PP_Vecchia = chol_ip_cross_cov_pred.transpose() * (chol_ip_cross_cov_obs * Vecchia_cross_cov);
					den_mat_t cross_cov_pred_obs_pred_inv;
					den_mat_t B_po_cross_cov(num_re_pred_cli, sigma_cross_cov.cols());
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < sigma_cross_cov.cols(); ++i) {
						B_po_cross_cov.col(i) = Bpo_rm * sigma_cross_cov.col(i);
					}
					den_mat_t cross_cov_PP_Vecchia_woodbury = chol_fact_sigma_woodbury_cluster_i.solve(cross_cov_PP_Vecchia.transpose());
					sp_mat_t Bp_inv_Dp;
					sp_mat_t Bp_inv(num_re_pred_cli, num_re_pred_cli);
					if (CondObsOnly) {
						if (calc_pred_cov) {
							pred_cov = Dp.asDiagonal();
						}
						if (calc_pred_var) {
							pred_var = Dp;
						}
						cross_cov_pred_obs_pred_inv = B_po_cross_cov;
					}
					else {
						TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(Bp, B_po_cross_cov, cross_cov_pred_obs_pred_inv, false);
						Bp_inv.setIdentity();
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
						Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
						if (calc_pred_cov) {
							pred_cov = T_mat(Bp_inv_Dp * Bp_inv.transpose());
						}
						if (calc_pred_var) {
							pred_var.resize(num_re_pred_cli);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_re_pred_cli; ++i) {
								pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i));
							}
						}
					}
					den_mat_t cross_cov_pred_obs_pred_inv_woodbury = chol_fact_sigma_woodbury_cluster_i.solve(cross_cov_pred_obs_pred_inv.transpose());
					if (calc_pred_cov) {
						if (num_re_pred_cli > 10000) {
							Log::REInfo("The computational complexity and the storage of the predictive covariance martix heavily depend on the number of prediction location. "
								"Therefore, if this number is large we recommend only computing the predictive variances ");
						}
						T_mat PP_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_ip * sigma_ip_inv_sigma_cross_cov_pred, PP_Part);
						T_mat PP_V_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia * sigma_ip_inv_sigma_cross_cov_pred, PP_V_Part);
						T_mat V_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * sigma_ip_inv_sigma_cross_cov_pred, V_Part);
						T_mat V_Part_t;
						ConvertTo_T_mat_FromDense<T_mat>(sigma_ip_inv_sigma_cross_cov_pred.transpose() * cross_cov_pred_obs_pred_inv.transpose(), V_Part_t);
						T_mat PP_V_PP_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * cross_cov_PP_Vecchia_woodbury, PP_V_PP_Part);
						T_mat PP_V_PP_Part_t;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia_woodbury.transpose() * cross_cov_pred_obs_pred_inv.transpose(), PP_V_PP_Part_t);
						T_mat PP_V_V_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia * cross_cov_PP_Vecchia_woodbury, PP_V_V_Part);
						T_mat V_V_Part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * cross_cov_pred_obs_pred_inv_woodbury, V_V_Part);
						pred_cov += PP_Part - PP_V_Part + V_Part + V_Part_t - PP_V_PP_Part + PP_V_V_Part - PP_V_PP_Part_t + V_V_Part;
					}
					if (calc_pred_var) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_re_pred_cli; ++i) {
							pred_var[i] += (cross_cov_pred_ip.row(i) - cross_cov_PP_Vecchia.row(i) +
								2 * cross_cov_pred_obs_pred_inv.row(i)).dot(sigma_ip_inv_sigma_cross_cov_pred.col(i)) +
								(cross_cov_PP_Vecchia.row(i) - 2 * cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_PP_Vecchia_woodbury.col(i)) +
								(cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_pred_obs_pred_inv_woodbury.col(i));
						}
					}
				}
			}
			else {
				pred_mean = -Bpo * y_cluster_i;
				if (!CondObsOnly) {
					sp_L_solve(Bp.valuePtr(), Bp.innerIndexPtr(), Bp.outerIndexPtr(), num_re_pred_cli, pred_mean.data());
				}
				if (calc_pred_cov || calc_pred_var) {
					if (calc_pred_var) {
						pred_var = vec_t(num_re_pred_cli);
					}
					if (CondObsOnly) {
						if (calc_pred_cov) {
							pred_cov = Dp.asDiagonal();
						}
						if (calc_pred_var) {
							pred_var = Dp;
						}
					}
					else {
						sp_mat_t Bp_inv(num_re_pred_cli, num_re_pred_cli);
						Bp_inv.setIdentity();
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
						sp_mat_t Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
						if (calc_pred_cov) {
							pred_cov = T_mat(Bp_inv_Dp * Bp_inv.transpose());
						}
						if (calc_pred_var) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_re_pred_cli; ++i) {
								pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i));
							}
						}
					}
				}//end calc_pred_cov || calc_pred_var
			}
			//release matrices that are not needed anymore
			Bpo.resize(0, 0);
			Bp.resize(0, 0);
			Dp.resize(0);
		}//end if gauss_likelihood
	}//end CalcPredVecchiaObservedFirstOrder

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable proces when prediction locations appear first in the ordering
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_pred Total number of prediction locations (over all clusters)
	* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param gp_rand_coef_data_pred Random coefficient data for GPs
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	*/
	template<typename T_mat>
	void CalcPredVecchiaPredictedFirstOrder(data_size_t cluster_i,
		int num_data_pred,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var) {
		int num_data_cli = (int)gp_coords_mat_obs.rows();
		int num_data_pred_cli = (int)gp_coords_mat_pred.rows();
		int num_data_tot = num_data_cli + num_data_pred_cli;
		//Find nearest neighbors
		den_mat_t coords_all(num_data_tot, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_pred, gp_coords_mat_obs;
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_tot);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_tot);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_tot);
		bool check_has_duplicates = false;
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		if (distances_saved) {
			find_nearest_neighbors_Vecchia_fast(coords_all, num_data_tot, num_neighbors_pred,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, distances_saved);
		}
		else {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
			find_nearest_neighbors_Vecchia_fast(coords_scaled, num_data_tot, num_neighbors_pred,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, distances_saved);
		}

		//Prepare data for random coefficients
		std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_tot);
		if (num_gp_rand_coef > 0) {
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data(num_data_tot);//First entries are the predicted data, then the observed data
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					rand_coef_data[i] = gp_rand_coef_data_pred[j * num_data_pred + data_indices_per_cluster_pred[cluster_i][i]];
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_cli; ++i) {
					rand_coef_data[num_data_pred_cli + i] = re_comps[cluster_i][ind_intercept_gp + j + 1]->RandCoefData()[i];
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_tot; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
					int dim_z = (int)nearest_neighbors_cluster_i[i].size() + 1;
					vec_t coef_vec(dim_z);
					coef_vec(0) = rand_coef_data[i];
					if (i > 0) {
						for (int ii = 1; ii < dim_z; ++ii) {
							coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
						}
					}
					z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
				}
			}
		}
		// Determine Triplet for initializing Bo, Bop, and Bp
		std::vector<Triplet_t> entries_init_Bo, entries_init_Bop, entries_init_Bp;
		for (int i = 0; i < num_data_pred_cli; ++i) {
			entries_init_Bp.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
			for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
				entries_init_Bp.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
			}
		}
		for (int i = 0; i < num_data_cli; ++i) {
			entries_init_Bo.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
			for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i + num_data_pred_cli].size(); ++inn) {
				if (nearest_neighbors_cluster_i[i + num_data_pred_cli][inn] < num_data_pred_cli) {//nearest neighbor belongs to predicted data
					entries_init_Bop.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i + num_data_pred_cli][inn], 0.));
				}
				else {//nearest neighbor belongs to predicted data
					entries_init_Bo.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i + num_data_pred_cli][inn] - num_data_pred_cli, 0.));
				}
			}
		}
		sp_mat_t Bo(num_data_cli, num_data_cli);
		sp_mat_t Bop(num_data_cli, num_data_pred_cli);
		sp_mat_t Bp(num_data_pred_cli, num_data_pred_cli);
		Bo.setFromTriplets(entries_init_Bo.begin(), entries_init_Bo.end());//initialize matrices (in order that the code below can be run in parallel)
		Bop.setFromTriplets(entries_init_Bop.begin(), entries_init_Bop.end());
		Bp.setFromTriplets(entries_init_Bp.begin(), entries_init_Bp.end());
		vec_t Do_inv(num_data_cli);
		vec_t Dp_inv(num_data_pred_cli);
		Do_inv.setOnes();//Put 1 on the diagonal (for nugget effect)
		Dp_inv.setOnes();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data_tot; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			//define covariance and gradient matrices
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				for (int j = 0; j < num_gp_total; ++j) {
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							coords_i = coords_all(ind, Eigen::all);
							coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
						}
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
					}
				}//end loop over components j
			}
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps[cluster_i][ind_intercept_gp + j]->CovPars()[0];
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				if (i < num_data_pred_cli) {
					Dp_inv[i] += d_comp_j;
				}
				else {
					Do_inv[i - num_data_pred_cli] += d_comp_j;
				}
			}
			//2. remaining terms
			if (i > 0) {
				cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					if (i < num_data_pred_cli) {
						Bp.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
					}
					else {
						if (nearest_neighbors_cluster_i[i][inn] < num_data_pred_cli) {//nearest neighbor belongs to predicted data
							Bop.coeffRef(i - num_data_pred_cli, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
						}
						else {
							Bo.coeffRef(i - num_data_pred_cli, nearest_neighbors_cluster_i[i][inn] - num_data_pred_cli) -= A_i(0, inn);
						}
					}
				}
				if (i < num_data_pred_cli) {
					Dp_inv[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
				}
				else {
					Do_inv[i - num_data_pred_cli] -= (A_i * cov_mat_obs_neighbors)(0, 0);
				}
			}
			if (i < num_data_pred_cli) {
				Dp_inv[i] = 1 / Dp_inv[i];
			}
			else {
				Do_inv[i - num_data_pred_cli] = 1 / Do_inv[i - num_data_pred_cli];
			}
		}//end loop over data i
		sp_mat_t cond_prec = Bp.transpose() * Dp_inv.asDiagonal() * Bp + Bop.transpose() * Do_inv.asDiagonal() * Bop;
		chol_sp_mat_t CholFact;
		CholFact.compute(cond_prec);
		vec_t y_aux = Bop.transpose() * (Do_inv.asDiagonal() * (Bo * y_cluster_i));
		pred_mean = -CholFact.solve(y_aux);
		if (calc_pred_cov || calc_pred_var) {
			sp_mat_t cond_prec_chol_inv(num_data_pred_cli, num_data_pred_cli);
			cond_prec_chol_inv.setIdentity();
			TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(CholFact.CholFactMatrix(), cond_prec_chol_inv, cond_prec_chol_inv, false);
			if (calc_pred_cov) {
				pred_cov = T_mat(cond_prec_chol_inv.transpose() * cond_prec_chol_inv);
			}
			if (calc_pred_var) {
				pred_var = vec_t(num_data_pred_cli);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					pred_var[i] = (cond_prec_chol_inv.col(i)).dot(cond_prec_chol_inv.col(i));
				}
			}
		}//end calc_pred_cov || calc_pred_var
	}//end CalcPredVecchiaPredictedFirstOrder

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the latent process when observed locations appear first in the ordering (only for Gaussian likelihoods)
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects (only has an effect on pred_cov and pred_var)
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	 */
	template<typename T_mat>
	void CalcPredVecchiaLatentObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		int ind_intercept_gp,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool predict_response,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var) {
		int num_data_cli = (int)gp_coords_mat_obs.rows();
		CHECK(num_data_cli == re_comps[cluster_i][ind_intercept_gp]->GetNumUniqueREs());
		int num_data_pred_cli = (int)gp_coords_mat_pred.rows();
		int num_data_tot = num_data_cli + num_data_pred_cli;
		//Find nearest neighbors
		den_mat_t coords_all(num_data_cli + num_data_pred_cli, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
		//Determine number of unique observartion locations
		std::vector<int> uniques;//unique points
		std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
		DetermineUniqueDuplicateCoordsFast(gp_coords_mat_obs, num_data_cli, uniques, unique_idx);
		int num_coord_unique_obs = (int)uniques.size();
		//Determine unique locations (observed and predicted)
		DetermineUniqueDuplicateCoordsFast(coords_all, num_data_tot, uniques, unique_idx);
		int num_coord_unique = (int)uniques.size();
		den_mat_t coords_all_unique;
		if ((int)uniques.size() == num_data_tot) {//no multiple observations at the same locations -> no incidence matrix needed
			coords_all_unique = coords_all;
		}
		else {
			coords_all_unique = coords_all(uniques, Eigen::all);
		}
		//Determine incidence matrices
		sp_mat_t Z_o = sp_mat_t(num_data_cli, uniques.size());
		sp_mat_t Z_p = sp_mat_t(num_data_pred_cli, uniques.size());
		std::vector<Triplet_t> entries_Z_o, entries_Z_p;
		for (int i = 0; i < num_data_tot; ++i) {
			if (i < num_data_cli) {
				entries_Z_o.push_back(Triplet_t(i, unique_idx[i], 1.));
			}
			else {
				entries_Z_p.push_back(Triplet_t(i - num_data_cli, unique_idx[i], 1.));
			}
		}
		Z_o.setFromTriplets(entries_Z_o.begin(), entries_Z_o.end());
		Z_p.setFromTriplets(entries_Z_p.begin(), entries_Z_p.end());
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_coord_unique);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_coord_unique);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_coord_unique);
		bool check_has_duplicates = true;
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		if (!distances_saved) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all_unique, coords_scaled);
		}
		if (CondObsOnly) {//find neighbors among both the observed locations only
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
		}
		else {//find neighbors among both the observed and prediction locations
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
		}
		if (check_has_duplicates) {
			Log::REFatal("Duplicates found among training and test coordinates. "
				"This is not supported for predictions with a Vecchia approximation for the latent process ('latent_') ");
		}
		// Determine Triplet for initializing Bpo and Bp
		std::vector<Triplet_t> entries_init_B;
		for (int i = 0; i < num_coord_unique; ++i) {
			entries_init_B.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
			for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
				entries_init_B.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
			}
		}
		sp_mat_t B(num_coord_unique, num_coord_unique);
		B.setFromTriplets(entries_init_B.begin(), entries_init_B.end());//initialize matrices (in order that the code below can be run in parallel)
		vec_t D(num_coord_unique);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_coord_unique; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			//define covariance and gradient matrices
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				if (!distances_saved) {
					std::vector<int> ind{ i };
					coords_i = coords_all_unique(ind, Eigen::all);
					coords_nn_i = coords_all_unique(nearest_neighbors_cluster_i[i], Eigen::all);
				}
				re_comps[cluster_i][ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
					cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
				re_comps[cluster_i][ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
					cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
			}
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			D[i] = re_comps[cluster_i][ind_intercept_gp]->CovPars()[0];
			//2. remaining terms
			if (i > 0) {
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					B.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
				}
				D[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
			}
		}//end loop over data i
		//Calculate D_inv and B_inv in order to calcualte Sigma and Sigma^-1
		vec_t D_inv = D.cwiseInverse();
		sp_mat_t B_inv(num_coord_unique, num_coord_unique);
		B_inv.setIdentity();
		TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(B, B_inv, B_inv, false);
		//Calculate inverse of covariance matrix for observed data using the Woodbury identity
		sp_mat_t M_aux_Woodbury = B.transpose() * D_inv.asDiagonal() * B + Z_o.transpose() * Z_o;
		chol_sp_mat_t CholFac_M_aux_Woodbury;
		CholFac_M_aux_Woodbury.compute(M_aux_Woodbury);
		if (calc_pred_cov || calc_pred_var) {
			sp_mat_t Identity_obs(num_data_cli, num_data_cli);
			Identity_obs.setIdentity();
			sp_mat_t MInvSqrtX_Z_o_T;
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(CholFac_M_aux_Woodbury, Z_o.transpose(), MInvSqrtX_Z_o_T, false);
			sp_mat_t ZoSigmaZoT_plusI_Inv = -MInvSqrtX_Z_o_T.transpose() * MInvSqrtX_Z_o_T + Identity_obs;
			sp_mat_t Z_p_B_inv = Z_p * B_inv;
			sp_mat_t Z_p_B_inv_D = Z_p_B_inv * D.asDiagonal();
			sp_mat_t ZpSigmaZoT = Z_p_B_inv_D * (B_inv.transpose() * Z_o.transpose());
			sp_mat_t M_aux = ZpSigmaZoT * ZoSigmaZoT_plusI_Inv;
			pred_mean = M_aux * y_cluster_i;
			if (calc_pred_cov) {
				pred_cov = T_mat(Z_p_B_inv_D * Z_p_B_inv.transpose() - M_aux * ZpSigmaZoT.transpose());
				if (predict_response) {
					pred_cov.diagonal().array() += 1.;
				}
			}
			if (calc_pred_var) {
				pred_var = vec_t(num_data_pred_cli);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					pred_var[i] = (Z_p_B_inv_D.row(i)).dot(Z_p_B_inv.row(i)) - (M_aux.row(i)).dot(ZpSigmaZoT.row(i));
				}
				if (predict_response) {
					pred_var.array() += 1.;
				}
			}
		}//end calc_pred_cov || calc_pred_var
		else {
			vec_t resp_aux = Z_o.transpose() * y_cluster_i;
			vec_t resp_aux2 = CholFac_M_aux_Woodbury.solve(resp_aux);
			resp_aux = y_cluster_i - Z_o * resp_aux2;
			pred_mean = Z_p * (B_inv * (D.asDiagonal() * (B_inv.transpose() * (Z_o.transpose() * resp_aux))));
		}
	}//end CalcPredVecchiaLatentObservedFirstOrder

}  // namespace GPBoost

#endif   // GPB_VECCHIA_H_
