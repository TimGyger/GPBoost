% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/GPModel.R
\name{predict.GPModel}
\alias{predict.GPModel}
\title{Make predictions for a \code{GPModel}}
\usage{
\method{predict}{GPModel}(object, predict_response = TRUE,
  predict_var = FALSE, predict_cov_mat = FALSE, y = NULL,
  cov_pars = NULL, group_data_pred = NULL,
  group_rand_coef_data_pred = NULL, gp_coords_pred = NULL,
  gp_rand_coef_data_pred = NULL, cluster_ids_pred = NULL, X_pred = NULL,
  use_saved_data = FALSE, offset = NULL, offset_pred = NULL,
  fixed_effects = NULL, fixed_effects_pred = NULL,
  vecchia_pred_type = NULL, num_neighbors_pred = NULL, ...)
}
\arguments{
\item{object}{a \code{GPModel}}

\item{predict_response}{A \code{boolean}. If TRUE, the response variable (label) 
is predicted, otherwise the latent random effects}

\item{predict_var}{A \code{boolean}. If TRUE, the (posterior) 
predictive variances are calculated}

\item{predict_cov_mat}{A \code{boolean}. If TRUE, the (posterior) 
predictive covariance is calculated in addition to the (posterior) predictive mean}

\item{y}{Observed data (can be NULL, e.g. when the model has been estimated 
already and the same data is used for making predictions)}

\item{cov_pars}{A \code{vector} containing covariance parameters which are used if the 
\code{GPModel} has not been trained or if predictions should be made for other 
parameters than the trained ones}

\item{group_data_pred}{A \code{vector} or \code{matrix} with elements being group levels 
for which predictions are made (if there are grouped random effects in the \code{GPModel})}

\item{group_rand_coef_data_pred}{A \code{vector} or \code{matrix} with covariate data 
for grouped random coefficients (if there are some in the \code{GPModel})}

\item{gp_coords_pred}{A \code{matrix} with prediction coordinates (=features) for 
Gaussian process (if there is a GP in the \code{GPModel})}

\item{gp_rand_coef_data_pred}{A \code{vector} or \code{matrix} with covariate data for 
Gaussian process random coefficients (if there are some in the \code{GPModel})}

\item{cluster_ids_pred}{A \code{vector} with elements indicating the realizations of 
random effects / Gaussian processes for which predictions are made 
(set to NULL if you have not specified this when creating the \code{GPModel})}

\item{X_pred}{A \code{matrix} with prediction covariate data for the 
fixed effects linear regression term (if there is one in the \code{GPModel})}

\item{use_saved_data}{A \code{boolean}. If TRUE, predictions are done using 
a priory set data via the function '$set_prediction_data' (this option is not used by users directly)}

\item{offset}{A \code{numeric} \code{vector} with 
additional fixed effects contributions that are added to the linear predictor (= offset). 
The length of this vector needs to equal the number of training data points.}

\item{offset_pred}{A \code{numeric} \code{vector} with 
additional fixed effects contributions that are added to the linear predictor for the prediction points (= offset). 
The length of this vector needs to equal the number of prediction points.}

\item{fixed_effects}{This is discontinued. Use the renamed equivalent argument \code{offset} instead}

\item{fixed_effects_pred}{This is discontinued. Use the renamed equivalent argument \code{offset_pred} instead}

\item{vecchia_pred_type}{A \code{string} specifying the type of Vecchia approximation used for making predictions.
This is discontinued here. Use the function 'set_prediction_data' to specify this}

\item{num_neighbors_pred}{an \code{integer} specifying the number of neighbors for making predictions.
This is discontinued here. Use the function 'set_prediction_data' to specify this}

\item{...}{(not used, ignore this, simply here that there is no CRAN warning)}
}
\value{
Predictions from a \code{GPModel}. A list with three entries is returned:
\itemize{
\item{ "mu" (first entry): predictive (=posterior) mean. For (generalized) linear mixed
effects models, i.e., models with a linear regression term, this consists of the sum of 
fixed effects and random effects predictions }
\item{ "cov" (second entry): predictive (=posterior) covariance matrix. 
This is NULL if 'predict_cov_mat=FALSE'  }
\item{ "var" (third entry) : predictive (=posterior) variances. 
This is NULL if 'predict_var=FALSE'  }
}
}
\description{
Make predictions for a \code{GPModel}
}
\examples{
# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

\donttest{
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
                       likelihood="gaussian", params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance


#--------------------Gaussian process model----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       likelihood="gaussian", y = y, X = X1, params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted (posterior) mean of GP
pred$cov # Predicted (posterior) covariance matrix of GP
}

}
\author{
Fabio Sigrist
}
