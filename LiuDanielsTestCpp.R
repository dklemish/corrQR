library(copula)
library(mvtnorm)
library(MCMCpack)
library(ggplot2)
library(inline)
library(RcppArmadillo)

set.seed(6)

Q      <- 3
n.obs  <- 1000
n.samp <- 10000
R <- matrix(c(1, 0.8, -0.3, 0.8, 1, -0.5, -0.3, -0.5, 1), nrow=Q)

R.samp      <- array(0, dim=c(Q,Q,n.samp))
R.samp[,,1] <- 0.5*diag(Q)

### Barnard sampling method
incl.text.Barnard <- '
  #include <R.h>
  #include <Rmath.h>

  using namespace arma;

  mat rInvWish(const double nu, const mat &S){
    int i, j;
    int q = S.n_rows;

    mat L = chol(inv(S));
    mat A = mat(q, q, arma::fill::zeros);

    for(i = 0; i < q; i++){
      A.at(i,i) = sqrt(as<double>(rchisq(1, nu-i)));
    }

    // Memory access is faster going in column order
    for(j = 0; j < q; j++){
      for(i = j+1; i < q; i++){
        A.at(i, j) = as<double>(rnorm(1, 0 ,1));
      }
    }

    return inv(L * A * (L*A).t());
  }
'

src.code.Barnard <- '
  Rcpp::NumericMatrix Rmat(R_);
  int n = Rcpp::as<int>(n_);
  int q = Rcpp::as<int>(q_);
  Rcpp::NumericMatrix Zmat(Z_);

  arma::mat Rcorr(Rmat.begin(), q, q, false);
  arma::mat Z(Zmat.begin(), n, q, false);
  
  Z.rows(0,9).print("Z");

  arma::mat R_inv(q, q, arma::fill::zeros);
  arma::vec alpha_samp(q, arma::fill::zeros);
  arma::mat D(q,q, arma::fill::zeros);
  arma::mat D_inv(q,q, arma::fill::zeros);
  arma::mat eps_star(q, n, arma::fill::zeros);
  arma::mat S_samp(q, q, arma::fill::zeros);
  arma::mat Sigma(q, q, arma::fill::zeros);
  
  Rcorr.print("Rcorr");
  R_inv      = arma::inv_sympd(Rcorr);
  R_inv.print("R_inv");
  //alpha_samp = as<arma::vec>(rgamma(q, ((double)q+1)/2, 1.0));
  alpha_samp << 3.0 << 2.0 << 1.0;
  alpha_samp.print("alpha_samp");
  D          = diagmat(sqrt(R_inv.diag() / (2*alpha_samp)));
  D.print("D");
  D_inv      = arma::inv(D);
  D_inv.print("D_inv");
  eps_star   = D * Z.t();
  eps_star.print("eps_star");
  S_samp     = eps_star * eps_star.t();
  S_samp.print("S_samp");
  Sigma      = rInvWish(n + q + 1, S_samp);
  Sigma.print("Sigma");
  D_inv      = diagmat(pow(Sigma.diag(), -0.5));
  D_inv.print("D_inv");
  Rcorr      = D_inv * Sigma * D_inv;
  Rcorr.print("Rcorr");
  
  return(Rcpp::wrap(Rcorr));
'

sample.Barnard <- 
  cxxfunction(signature(n_="numeric", q_="numeric", R_="numeric", Z_="numeric"),
              includes=incl.text.Barnard,
              body=src.code.Barnard,
              plugin = "RcppArmadillo")

# for(i in 2:n.samp){
#   Z <- rmvnorm(n.obs, sigma=R)
#   R.samp[,,i] <- sample.Barnard(n.obs, Q, R.samp[,,i-1], Z)
# }
# 
# apply(R.samp[,,101:n.samp],c(1,2),summary)
# 
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[1,2,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[1,3,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[2,3,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
# 
# 
# 
# 
# 
# 
# 
# 
# # Liu & Daniels paper
# R.samp <- array(0, dim=c(Q,Q,n.samp))
# accept <- rep(0, n.samp)
# R.samp[,,1] <- diag(Q)
# 
# log.curr.det <- log(det(R.samp[,,1]))
# expo <- -(Q+1)/2
# uniform.accept.draws <- runif(n.samp)
# 
# # Target prior \propto 1
# # Candidiate prior \propto |R|^((Q+1)/2)
# for(i in 2:n.samp){
#   Z <- rmvnorm(n.obs, sigma=R)
#   #Zt <- t(Z)
#   
#   # Calculate current D since \sum Z_{ij}^2 = d_{jj}^{-2} \forall j
#   D <- diag(apply(Z^2, 2, sum)^(-0.5))
#   
#   # Calculate parameter expansion & draw new sigma matrix
#   epsilon <- Z %*% D
#   S       <- matrix(apply(apply(epsilon, 1, function(x) x %*% t(x)), 1, sum),
#                     nrow=Q)
#   Sigma   <- riwish(n.obs, S)
#   
#   # Drawing Sigma implicitly draws new D & R, so calculate new D & R
#   D.new   <- diag(diag(Sigma)^-0.5)
#   R.prop  <- D.new %*% Sigma %*% D.new
#   
#   # MH acceptance step
#   log.prop.det <- log(det(R.prop))
#   accept.prob  <- min(1, exp(expo*(log.prop.det - log.curr.det)))
#   
#   if(uniform.accept.draws[i] < accept.prob){
#     R.samp[,,i]  <- R.prop
#     log.curr.det <- log.prop.det
#     accept[i] <- 1
#   } else{
#     R.samp[,,i]  <- R.samp[,,i-1]
#   }
# }
# apply(R.samp[,,101:n.samp],c(1,2),summary)
# 
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[1,2,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[1,3,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
# ggplot() +
#   geom_histogram(data=data.frame(x=R.samp[2,3,101:n.samp]),
#                  aes(x, ..density..),
#                  binwidth=0.0025) +
#   theme_bw()
