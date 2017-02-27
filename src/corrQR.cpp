#define sqrt5 2.236067977499789696
#define u_min_val 1.0e-15
#define u_max_val (1.0 - 1.0e-15)

#include <R.h>
#include <Rmath.h>
#include <RcppArmadillo.h>
#include <iomanip>      // std::setprecision
// RcppArmadillo.h also includes Rcpp.h

// For use of lambda functions
// [[Rcpp::plugins(cpp11)]]

// Declare dependency on RcppArmadillo so Rcpp knows to link libraries
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

/***** Function headers *****/
double sigFn(double);
double sigFn_inv(double);
double nuFn(double);
double nuFn_inv(double);

vec unitFn(const vec &);
vec q0(const vec &, double);
vec Q0(const vec &, double);
vec F0(const vec &, double);

vec q0tail(const vec &);
double Q0tail(double);
double lf0tail(double);

double ppFn0(const vec &, int);
double ppFn(const vec &, int, int);
double logsum(const vec &);
double logmean(const vec &);
void   trape(double *x, double *h, int length, double *integral);
void   trapeReverse(double *x, double *h, int length, double *integral);
double logPosterior(const vec &, bool llonly);

double log_dGaussCopula(const mat &, const mat &);

void MCMC(void);
mat rInvWish(const double, const mat &);

SEXP corr_qr_fit(SEXP par_,
                 SEXP x_,
                 SEXP y_,
                 SEXP hyper,
                 SEXP dim_,
                 SEXP A_,
                 SEXP R_,
                 SEXP LOGDET_,
                 SEXP LPGRID_,
                 SEXP tauG_,
                 SEXP muV_,
                 SEXP SV_,
                 SEXP blocks_,
                 SEXP blockSizes_,
                 SEXP dmcmcpar_,
                 SEXP imcmcpar_);

SEXP devianceCalc(SEXP pars_,
                  SEXP x_,
                  SEXP y_,
                  SEXP hyper,
                  SEXP dim_,
                  SEXP A_,
                  SEXP R_,
                  SEXP LOGDET_,
                  SEXP LPGRID_,
                  SEXP tauG_);

void Init_Prior_Param(int L, int m, int G, int nblocks, int nkap, NumericVector hyp,
                      List Ag, List Rg, List muV, List SV,
                      IntegerVector blocks_, IntegerVector bSize);

/**** Global variables ****/

// int par_position = (p+1)*m, gam_pos_offset = p - 1;
int par_position;
int gam_pos_offset;
double zeta0tot;
double Q0_val;
double Q_L; // = -std::numeric_limits<double>::infinity();
double Q_U; // =  std::numeric_limits<double>::infinity();
double w0max;
double alp;
double y_i;
double lp;
int lower_ind, upper_ind;
double t_l, t_u, w_l, w_u;

const double shrinkFactor = 1.0;

// Data
mat x;        // covariates (n x p)
mat y;        // response (n x 1)
vec taugrid;  // quantiles of interest (L x 1)

// Dimension parameters
int n;     // # obs
int p;     // # predictors
int q;     // # response variables
int L;     // # tau's (quantiles) at which betas are estimated
int mid;   // location in taugrid closest to median (0.50)
int m;     // dim of low rank approx to w functions
int G;     // dim G of approx to pi(lambda)
int nkap;  // # of mixture components for prior on kappa_j
int ncorr; // # of correlation parameters for Gaussian copula

// MCMC parameters
int niter;    // MCMC iterations
int thin;     // thinning factor
int nsamp;    // # of MCMC samples to keep
int npar;     // # parameters to track per response variable
int nrespar;  // # of response parameters = q * npar
int totpar;   // # of total parameters
int nblocks;  // # of MH blocks
bool verbose; // flag to print intermediate calc's
bool fix_corr;// flag for whether correlation parameters should be fixed (true) or learned via MCMC (false)
int ticker;   // how often to update screen

// Adaptive MCMC
int refresh;  // adaption rate of adaptive MH
double decay;
ivec refresh_counter; //
vec acpt_target;      //
vec lm;               //

// Prior parameters
cube Agrid;   // storage for A_g = C_0*(g) C**^{-1}(g) (L x m x G)
cube Rgrid;   // storage for R_g = chol(C**(g)) (m x m x G) (upper triangular Cholesky decomposition)
vec ldRgrid;  // log trace(R_g), normalizing constant
vec lpgrid;   // log prior (lambda_j)

// vector of vector / matrices allows for jagged sizes
std::vector<vec> mu; // init MH block means
std::vector<mat> S;  // initial MH block covariances
std::vector<uvec> blocks;
ivec blockSizes;

// Hyperparameters
vec akap;
vec bkap;
vec lpkap;   // hyperparms for kappa

// Model parameters
double gam0;  // curr value of gamma_0
double sigma; // curr value of sigma
double nu;    // curr value of nu

vec gam;      // current value of vector gamma (p x 1) for given response
vec zeta0;    // current values of zeta(tau_l) (L x 1) for given response
vec zeta0dot; // current values of zeta.dot(tau_l) (L x 1) for given response

mat w0;       // current values of w_0(tau_l) (L x q) for all responses
cube wMat;    // each col corresponds to current values of W_j (L x p x q) for all responses
cube vMat;    // each col corresponds to W_j(zeta(tau)) (L x p x q) for all responses

mat Rcorr;    // correlation matrix for Gaussian copula

// Intermediate calculations
mat tau_y_x;  // matrix of quantiles for y_{ij} | x_i (n x q)
mat wgrid;    // temp to store A_g * low rank w knots for each g
mat a;        // = (x*vMat) (n x L)
vec aX;       // = (L x 1)
mat aTilde;   // = (n x L)
mat vTilde;   // = (L x p)
vec b0dot;    // (L x 1)
mat bdot;     // = x^T %*% beta.dot(tau) (L x p)
vec Q0_med;   // (n x 1)
vec vNormSq;
vec wknot;    // (m x 1)
vec zknot;    // (m x 1)
mat llmat;    // log-likelihood by obs (n x (q+1)) - first q columns are
//  contribution of q-th response for ith obs; last column is
//  contribution of copula for ith obs
vec llgrid;   // MV t-dist contrib to loglikelihood (G x 1)

vec lps0;     // (q x 1)
vec resLin;   // (n x 1)
vec Q0Pos;    // (L x 1)
vec Q0Neg;    // (L x 1)
mat bPos;     // (L x p)
mat bNeg;     // (L x p)
double QPos, QPosold, QNeg, QNegold;

cube pgvec;   // contains p_g(W_j*) (G x (p+1) x q)
vec lb;       // used for shrinkage prior on gamma vector (10 x 1)
vec lw;       // used for interpolation of marginal GP (nkap x 1)

vec parSample; // current parameter draw

// Outputs
vec lpSample;     // stored log-likelihood
mat acceptSample; // stored MH acceptance history
mat parStore;     // stored posterior draws npar x nsamp

// [[Rcpp::export]]
SEXP corr_qr_fit(SEXP par_,
                 SEXP x_,
                 SEXP y_,
                 SEXP hyper_,
                 SEXP dim_,
                 SEXP A_,
                 SEXP R_,
                 SEXP LOGDET_,
                 SEXP LPGRID_,
                 SEXP tauG_,
                 SEXP muV_,
                 SEXP SV_,
                 SEXP blocks_,
                 SEXP blockSizes_,
                 SEXP dmcmcpar_,
                 SEXP imcmcpar_){
  /***** Initialization *****/
  // Input data
  // Convert SEXP objects to Rcpp objects
  NumericVector PAR    = as<NumericVector>(par_);
  NumericMatrix X      = as<NumericMatrix>(x_);
  NumericMatrix Y      = as<NumericMatrix>(y_);
  NumericVector HYP    = as<NumericVector>(hyper_);
  IntegerVector DIM    = as<IntegerVector>(dim_);
  List Ag              = as<List>(A_);
  List Rg              = as<List>(R_);
  NumericVector LOGDET = as<NumericVector>(LOGDET_);
  NumericVector LPGRID = as<NumericVector>(LPGRID_);
  NumericVector TAU_G  = as<NumericVector>(tauG_);
  List MU_V            = as<List>(muV_);
  List SV              = as<List>(SV_);
  IntegerVector BLOCKS = as<IntegerVector>(blocks_);
  IntegerVector BLOCKS_SIZE = as<IntegerVector>(blockSizes_);
  NumericVector DMCMCPAR = as<NumericVector>(dmcmcpar_);
  IntegerVector IMCMCPAR = as<IntegerVector>(imcmcpar_);

  x       = mat(X.begin(), X.nrow(), X.ncol(), TRUE);
  y       = mat(Y.begin(), Y.nrow(), Y.ncol(), TRUE);
  taugrid = vec(TAU_G.begin(), TAU_G.size(), TRUE);
  ldRgrid = vec(LOGDET.begin(), LOGDET.size(), TRUE);
  lpgrid  = vec(LPGRID.begin(), LPGRID.size(), TRUE);

  // Dimensions
  n     = DIM[0];
  p     = DIM[1];
  q     = DIM[2];
  L     = DIM[3];
  mid   = DIM[4];
  m     = DIM[5];
  G     = DIM[6];
  nkap  = DIM[7];
  ncorr = DIM[11];

  par_position   = (p+1)*m;
  gam_pos_offset = p - 1;

  // MCMC parameters
  niter   = DIM[8];
  thin    = DIM[9];
  nsamp   = DIM[10];
  npar    = (m+1) * (p+1) + 2;
  nrespar = q*npar;
  totpar  = nrespar + ncorr;
  nblocks = IMCMCPAR[0];
  refresh = IMCMCPAR[1];
  verbose = (bool) IMCMCPAR[2];
  fix_corr= (bool) IMCMCPAR[4 + nblocks];
  ticker  = IMCMCPAR[3];
  decay   = DMCMCPAR[0];

  refresh_counter = ivec(IMCMCPAR.begin() + 4, nblocks, TRUE);
  acpt_target     =  vec(DMCMCPAR.begin() + 1, nblocks, TRUE);
  lm              =  vec(DMCMCPAR.begin() + 1 + nblocks, nblocks, TRUE);

  // Prior parameters
  Init_Prior_Param(L, m, G, nblocks, nkap, HYP, Ag, Rg, MU_V, SV, BLOCKS, BLOCKS_SIZE);

  // Parameters
  gam0     = 0;
  sigma    = 1;
  nu       = 1;
  gam      = vec(p, fill::zeros);
  zeta0    = vec(L, fill::zeros);
  zeta0dot = vec(L, fill::zeros);

  w0       = mat(L, q, fill::zeros);
  wMat     = cube(L, p, q, fill::zeros);
  vMat     = cube(L, p, q, fill::zeros);

  Rcorr    = mat(q, q, fill::eye);

  // Intermediate calculations
  tau_y_x  = mat(n, q, fill::zeros);
  wgrid    = mat(L, G, fill::zeros);
  a        = mat(n, L, fill::zeros);
  aX       = vec(L, fill::zeros);
  aTilde   = mat(n, L, fill::zeros);
  vTilde   = mat(L, p, fill::zeros);
  b0dot    = vec(L, fill::zeros);
  bdot     = mat(L, p, fill::zeros);
  Q0_med   = vec(n, fill::zeros);
  vNormSq  = vec(L, fill::zeros);
  wknot    = vec(m, fill::zeros);
  zknot    = vec(m, fill::zeros);
  llmat    = mat(n, q+1, fill::zeros);
  llgrid   = vec(G, fill::zeros);

  lps0     = vec(q, fill::zeros);
  resLin   = vec(n, fill::zeros);
  Q0Pos    = vec(L, fill::zeros);
  Q0Neg    = vec(L, fill::zeros);
  bPos     = mat(L, p, fill::zeros);
  bNeg     = mat(L, p, fill::zeros);

  pgvec    = cube(G, p+1, q, fill::zeros);
  lb       = vec(10, fill::zeros);
  lw       = vec(nkap, fill::zeros);

  parSample = vec(PAR.begin(), PAR.size(), TRUE);

  // Output
  lpSample     = vec(nsamp, fill::zeros);           // stored log-likelihood
  acceptSample = mat(nsamp, nblocks, fill::zeros);  // stored MH acceptance history
  parStore     = mat(totpar, nsamp, fill::zeros);     // stored posterior draws npar x nsamp

  // Convert portion of parameter vector corresponding to correlation matrix
  // to actual matrix
  int reach = 0;
  for(int i = 0; i < (q-1); i++){
    for(int j = i+1; j < q; j++){
      Rcorr.at(i,j) = parSample(q*npar + reach);
      reach++;
    }
  }
  Rcorr = symmatu(Rcorr);

  MCMC();

  return Rcpp::List::create(Rcpp::Named("parsamp") = parStore.t(),
                            Rcpp::Named("lpsamp") = lpSample,
                            Rcpp::Named("acptsamp") = acceptSample,
                            Rcpp::Named("blockMu") = mu,
                            Rcpp::Named("blockCov") = S,
                            Rcpp::Named("tau_y_x") = tau_y_x);
}

void Init_Prior_Param(int L, int m, int G, int nblocks, int nkap, NumericVector hyp,
                      List A, List R, List muV, List SV,
                      IntegerVector blocks_, IntegerVector bSize){

  int reach, i, b, block_point;
  NumericVector tempMu;
  NumericMatrix tempS;
  NumericMatrix tempA;
  NumericMatrix tempR;

  Agrid   = cube(L, m, G);
  Rgrid   = cube(m, m, G);

  mu.resize(nblocks);
  S.resize(nblocks);
  blocks.resize(nblocks);

  blockSizes = ivec(bSize.begin(), bSize.size(), TRUE);

  akap  = vec(nkap);
  bkap  = vec(nkap);
  lpkap = vec(nkap);

  // Initialize mixture components for prior on kappa_j
  for(reach = 2, i = 0; i < nkap; i++){
    akap[i]  = hyp[reach++];
    bkap[i]  = hyp[reach++];
    lpkap[i] = hyp[reach++];
  }

  for(i = 0; i < G; i++){
    tempA = as<NumericMatrix>(A[i]);
    Agrid.slice(i) = mat(tempA.begin(), tempA.nrow(), tempA.ncol(), TRUE);

    tempR = as<NumericMatrix>(R[i]);
    Rgrid.slice(i) = mat(tempR.begin(), tempR.nrow(), tempR.ncol(), TRUE);
  }

  // Initialize user provided block means, covariances
  for(b=0, block_point=0; b < nblocks; b++){

    tempMu = as<NumericVector>(muV[b]);
    tempS  = as<NumericMatrix>(SV[b]);

    mu[b]  = vec(tempMu.begin(), tempMu.size(), TRUE);
    S[b]   = mat(tempS.begin(), tempS.nrow(), tempS.ncol(), TRUE);

    blocks[b] = arma::conv_to<uvec>::from(ivec(blocks_.begin()+block_point, blockSizes[b], TRUE));
    block_point += blockSizes[b];
  }

  return;
}

// double log_dGaussCopula(const mat &U, const mat &Rc){
//   // Note no checks that U contains values in [0,1]
//   int i, j;
//   double d = det(Rc);
//   double result = 0;
//
//   vec phi_inv_u = vec(q);
//
//   for(i = 0; i < n; i++){
//     for(j = 0; j < q; j++){
//       phi_inv_u[j] = R::qnorm(U.at(i,j), 0, 1, 1, 0);
//     }
//     result += -0.5 * (log(d) + as_scalar(phi_inv_u.t() * (Rc.i() - eye(q,q)) * phi_inv_u));
//   }

double log_dGaussCopula(const vec &U, const mat &Rc){
  // Note no checks that U contains values in [0,1]
  int j;
  double d = det(Rc);
  double result = 0;

  vec phi_inv_u = vec(q);

  for(j = 0; j < q; j++){
    phi_inv_u[j] = R::qnorm(U[j], 0, 1, 1, 0);
  }
  result = -0.5 * (log(d) + as_scalar(phi_inv_u.t() * (Rc.i() - eye(q,q)) * phi_inv_u));

  return result;
}

double ppFn0(const vec &par, int resp){
  // Calculate interpolated w0 function at all quantiles based on values of
  // function at m knots for response variable # resp.
  int i;
  int offset = resp*npar;
  double zss, lps = 0;

  //par[0:(m-1)] contains w0 values at knots
  wknot = par.subvec(offset, offset + m-1);

  for(i = 0; i < G; i++){
    wgrid.col(i) = Agrid.slice(i) * wknot;
    zknot = arma::solve(arma::trimatl(Rgrid.slice(i).t()), wknot);

    // zss = w_*j^T * C_**^-1 * W_j*
    zss = arma::dot(zknot, zknot);

    // Calculate p(W_{j*}|lambda_g)
    llgrid[i] = -(0.1 + 0.5 * (double)m) * log1p(0.5 * zss / 0.1);
  }

  // pgvec contains p_g(W_j*)
  pgvec.slice(resp).col(0) = llgrid - ldRgrid + lpgrid;
  lps = logsum(pgvec.slice(resp).col(0));

  // Normalize probability
  for(i = 0; i < G; i++)
    pgvec.at(i,0,resp) = exp(pgvec.at(i,0,resp) - lps);

  w0.col(resp) = wgrid * pgvec.slice(resp).col(0);

  return lps;
}

double ppFn(const vec &par, int p, int resp){
  int i, j;
  int offset = resp*npar;
  double akapm, zss, lps;

  //par[p*m + 1:m] contains wJ values at knots
  wknot = par.subvec(offset + p * m, offset + (p+1) * m - 1);

  for(i = 0; i < G; i++){
    wgrid.col(i) = Agrid.slice(i) * wknot;
    zknot = arma::solve(arma::trimatl(Rgrid.slice(i).t()), wknot);

    // zss = w_*j^T * C_**^-1 * W_j*
    zss = arma::dot(zknot, zknot);

    // Calculate p(W_{j*}|lambda_g)
    for(j = 0; j < nkap; j++){
      akapm = akap[j] + 0.5 * (double)m;
      lw[j] = -akapm * log1p(0.5 * zss/ bkap[j]) + lgamma(akapm) - lgamma(akap[j]) -
        0.5 * (double)m * log(bkap[j]) + lpkap[j];
    }

    llgrid[i] = logsum(lw);
  }

  pgvec.slice(resp).col(p) = llgrid - ldRgrid + lpgrid;

  lps = logsum(pgvec.slice(resp).col(p));

  for(i = 0; i < G; i++)
    pgvec.at(i,p,resp) = exp(pgvec.at(i,p,resp) - lps);

  wMat.slice(resp).col(p-1) = wgrid * pgvec.slice(resp).col(p);

  return lps;
}

double logsum(const vec& L){
  double lxmax = L.max();
  double a = 0.0;

  for(unsigned int i = 0; i < L.size(); i++) a += exp(L[i] - lxmax);

  return lxmax + log(a);
}

double logmean(const vec& L){
  return logsum(L) - log(L.size());
}

void trape(double *x, double *h, int length, double *integral){
  // Calculates integral of function x over domain h via trapezoidal rule
  integral[0] = 0.0;

  for(int i = 1; i < length; i++){
    integral[i] = integral[i-1] + 0.5 * (h[i]-h[i-1]) * (x[i]+x[i-1]);
  }
  return;
}

void trapeReverse(double *x, double *h, int length, double *integral){
  // Calculates integral of function x over domain h via trapezoidal rule
  int j = 0;
  integral[0] = 0.0;

  for(int i = 1; i < length; i++){
    integral[i] = integral[i-1] + 0.5 * (h[j] - h[j-1]) * (x[j] + x[j-1]);
    j--;
  }

  return;
}


double sigFn(double z) {
  return exp(z/2.0);
}

double sigFn_inv(double s) {
  return 2.0 * log(s);
}

double nuFn(double z) {
  return 0.5 + 5.5*exp(z/2.0);
}

double nuFn_inv(double nu) {
  return 2.0*log((nu - 0.5)/5.5);
}

double logPosterior(const vec &par, bool llonly){
  int i, j, k, l;

  // Initialize log-likelihood matrix
  llmat.fill(0);

  // Loop over response variables
  for(k = 0; k < q; k++){
    // Read in current values of parameters
    gam0 = par[k*npar + par_position];
    gam  = par.subvec(k*npar + par_position + 1,
                      k*npar + par_position + gam_pos_offset + 1);
    sigma = sigFn(par[k*npar + par_position + gam_pos_offset + 2]);
    nu    = nuFn(par[k*npar + par_position + gam_pos_offset + 3]);

    // Calculate basic quantities

    //** 1. Determine vector w_0 based on interpolation at specified knots
    lps0[k] = ppFn0(par, k);

    //** 2. zeta & zeta.dot
    w0max    = w0.col(k).max();
    zeta0dot = exp(shrinkFactor * (w0.col(k) - w0max));

    // zeta is integral of zeta.dot
    trape(zeta0dot.memptr() + 1, taugrid.memptr() + 1, L-1, zeta0.memptr() + 1);

    // zeta0[0] = 0, zeta0[1] = 1 by definition
    // rescale zeta0 so that it lies in [0,1] for tau != 0, 1
    zeta0tot = zeta0[L-2];

    zeta0[0] = 0.0; zeta0[L-1] = 1.0;
    zeta0.subvec(1,L-2) = taugrid[1] + (taugrid[L-2] - taugrid[1]) * zeta0.subvec(1,L-2) / zeta0tot;

    zeta0dot[0] = 0.0; zeta0dot[L-1] = 0.0;
    zeta0dot.subvec(1,L-2) = (taugrid[L-2] - taugrid[1]) * zeta0dot.subvec(1,L-2) / zeta0tot;

    //** 3. beta0.dot for each l (% is element-wise multiplication for Armadillo vectors)
    b0dot = sigma * q0(zeta0, nu) % zeta0dot;

    for(j = 1; j <= p; j++){
      lps0[k] += ppFn(par, j, k);
    }
    lps0[k] += R::dlogis(par[k*npar + par_position + gam_pos_offset + 3], 0.0, 1.0, 1);

    //** 4. Calculate matrix of w_j functions at each l based on interpolating between knots.
    //       Then find v matrix = w(zeta(tau)) for each l, via linear interpolation
    // Uses arma as_scalar & find functions
    for(j = 0; j < p; j++){
      for(l = 1; l < L-1; l++){
        lower_ind = as_scalar(find(zeta0[l] >= taugrid, 1, "last"));
        upper_ind = lower_ind + 1;

        t_l = as_scalar(taugrid[lower_ind]);
        t_u = as_scalar(taugrid[upper_ind]);
        w_l = as_scalar(wMat.slice(k).at(lower_ind, j));
        w_u = as_scalar(wMat.slice(k).at(upper_ind, j));

        vMat.at(l,j,k) = w_l + (zeta0[l]-t_l)*(w_u - w_l)/(t_u - t_l);
      }
      vMat.at(0,j,k)   = wMat.at(0,j,k);
      vMat.at(L-1,j,k) = wMat.at(L-1,j,k);
    }

    //** 5. ||v_l||^2 for each quantile l (1:L)
    for(l = 0; l < L; l++)
      vNormSq[l] = pow(norm(vMat.slice(k).row(l)),2);

    if(vNormSq.min() > 0.0){
      //** 5. a_i for each observation i & quantile l
      a = x * vMat.slice(k).t();  // a is n x L

      //** 6. a_chi for each l & tilde{a}_il for each obs i & quant l
      for(l = 0; l < L; l++){
        aX[l] = -a.col(l).min() / sqrt(vNormSq[l]);
        for(i = 0; i < p; i++){
          vTilde.at(l,i) = vMat.at(l,i,k) / (aX[l]*sqrt(1+vNormSq[l]));
          aTilde.at(i,l) = a.at(i,l) / (aX[l]*sqrt(1+vNormSq[l]));
        }
      }

      for(i = 0; i < p; i++){
        bdot.col(i) = b0dot % vTilde.col(i);
      }

      trape(b0dot.memptr() + mid, taugrid.memptr() + mid, L - mid, Q0Pos.memptr());
      Q0Pos[L-mid] = std::numeric_limits<double>::infinity();
      trapeReverse(b0dot.memptr() + mid, taugrid.memptr() + mid, mid + 1, Q0Neg.memptr());
      Q0Neg[mid+1] = std::numeric_limits<double>::infinity();

      for(i = 0; i < p; i++){
        trape(bdot.memptr() + L*i + mid, taugrid.memptr() + mid, L - mid, bPos.memptr() + L*i);
        trapeReverse(bdot.memptr() + L*i + mid, taugrid.memptr() + mid, mid + 1, bNeg.memptr() + L*i);
      }

      Q0_med = vec(n).fill(gam0) + x*gam;

      resLin = y.col(k) - Q0_med;

      // Compute modeled median (specifically the quantile of Y at tau_0 = F_0(0),
      // but our prior guess for F is a t-distribution, so F_0(0) = 0.5) of Y | X
      // for each observation i
      //** 7. Calculate log-likelihood by sequencing through observations
      for(i = 0; i < n; i++){
        if(resLin[i] == 0.0){
          llmat.at(i,k) = -log(b0dot[mid] + as_scalar(x.row(i) * bdot.row(mid).t()));

          tau_y_x.at(i,k) = taugrid[mid];

        } else if(resLin[i] > 0.0){
          l = 0;
          QPosold = 0.0;
          QPos = Q0Pos[l] + as_scalar(x.row(i) * bPos.row(l).t());

          QPos = Q0Pos[l] + as_scalar(x.row(i) * bPos.row(l).t());
          while(resLin[i] > QPos && l < L-mid-1){
            QPosold = QPos;
            l++;
            QPos = Q0Pos[l] + as_scalar(x.row(i) * bPos.row(l).t());
          }

          tau_y_x.at(i,k) = taugrid[mid + l];

          if(l == L - mid - 1)
            llmat.at(i,k) = lf0tail(Q0tail(taugrid[L-2]) + (resLin[i] - QPos)/sigma) - log(sigma);
          else
            llmat.at(i,k) = log(taugrid[mid+l] - taugrid[mid+l-1]) - log(QPos - QPosold);

        } else {
          l = 0;
          QNegold = 0.0;
          QNeg = Q0Neg[l] + as_scalar(x.row(i) * bNeg.row(l).t());

          while(resLin[i] < -QNeg && l < mid){
            QNegold = QNeg;
            l++;
            QNeg = Q0Neg[l] + as_scalar(x.row(i) * bNeg.row(l).t());
          }

          tau_y_x.at(i,k) = taugrid[mid - l];

          if(l == mid)
            llmat.at(i,k) = lf0tail(Q0tail(taugrid[1]) + (resLin[i] + QNeg)/sigma) - log(sigma);
          else
            llmat.at(i,k) = log(taugrid[mid-l+1]-taugrid[mid-l]) - log(QNeg - QNegold);
        }
        //if(ll[i] == qt(1.0, 1.0, 1, 0)) Rprintf("i = %d, ll[i] = %g, resLin[i] = %g, l = %d\n", i, ll[i], resLin[i], l);
      }
    }
  }

  // Calculate contribution to loglikelihood from copula distribution
  for(i = 0; i < n; i++){
    llmat.at(i, q) = log_dGaussCopula(conv_to< vec >::from(tau_y_x.row(i)), Rcorr);
  }

  // Calculate total loglikelihood
  lp = arma::accu(llmat);
  if(std::isnan(lp)) lp = -std::numeric_limits<double>::infinity();

  // lp += log_dGaussCopula(tau_y_x, Rcorr);

  if(!llonly){
    lp += accu(lps0);
  }

  return lp;
}

vec unitFn(const vec &u) {
  vec z(u);

  // u_min_val, u_max_val defined as #define preprocesser commands
  for(unsigned int i = 0; i < z.size(); i++){
    if(z[i] < u_min_val)
      z[i] = u_min_val;
    else if(z[i] > u_max_val)
      z[i] = u_max_val;
  }

  return z;
}

vec q0(const vec &u, double nu) {
  vec q(u.size());
  vec u_cap(unitFn(u));

  double adjFac = R::qt(0.9, nu, 1, 0);

  for(unsigned int i = 0; i < q.size(); i++){
    q[i] = 1/(R::dt(R::qt(u_cap[i], nu, 1, 0), nu, 0) * adjFac);
  }

  return q;
}

vec Q0(const vec &u, double nu) {
  vec Q(u.size());
  vec u_cap = unitFn(u);

  double adjFac = R::qt(0.9, nu, 1, 0);

  for(unsigned int i = 0; i < Q.size(); i++){
    Q[i] = R::qt(u_cap[i], nu, 1, 0) / adjFac;
  }

  return Q;
}

vec F0(const vec &x, double nu) {
  vec F(x.size());

  double adjFac = R::qt(0.9, nu, 1, 0);

  for(unsigned int i = 0; i < F.size(); i++){
    F[i] = R::pt(x[i]*adjFac, nu, 1, 0);
  }

  return F;
}

vec q0tail(const vec &u) {
  vec q(u.size());

  for(unsigned int i = 0; i < q.size(); i++){
    q[i] = 1/R::dt(R::qt(u[i], 0.1, 1, 0), 0.1, 0);
  }

  return q;
}

double Q0tail(double u) {
  return R::qt(u, 0.1, 1, 0);
}

double lf0tail(double x){
  return R::dt(x, 0.1, 1);
}

void MCMC(void){
  int b, i, j;
  int u = 0;
  int g = 0;
  int iter;
  int store_lp=0;
  int store_par=0;
  int store_acpt=0;
  int currBlockSize=0;

  double lpval = 0;
  double lpvalnew = 0;
  double lp_diff;
  double chs;
  double lambda;

  ivec blocks_rank(nblocks);
  ivec run_counter(nblocks, arma::fill::zeros);
  vec  alpha(nblocks);
  vec  alpha_run(nblocks, arma::fill::zeros);

  vec  blocks_d(npar);
  vec  parOld(npar);
  vec  logUniform(nblocks*niter);

  ivec chunk_size(nblocks, arma::fill::zeros);
  vec  acpt_chunk(nblocks, arma::fill::zeros);
  vec  frac(nblocks);
  vec  gammaAdj(nblocks*niter);

  // Ragged arrays
  std::vector<ivec> blocks_pivot(nblocks);
  std::vector<vec>  parbar_chunk(nblocks);
  std::vector<vec>  zsamp(nblocks);
  std::vector<vec>  par_incr(nblocks);
  std::vector<mat>  R(nblocks); // Chol factor of MCMC block proposal covariances

  // Data structures for sampling correlation matrices
  mat D(q, q, arma::fill::zeros);
  mat S_samp(q, q, arma::fill::zeros);
  mat Sigma(q, q, arma::fill::zeros);
  mat D_inv(q, q, arma::fill::zeros);

  // Liu & Daniels
  mat epsilon(n, q);
  mat R_prop(q,q);
  double log_det_current;
  double log_det_proposed;
  double accept_prob;

  // Barnard et. al
  // mat R_inv(q, q, arma::fill::zeros);
  // vec alpha_samp(q, arma::fill::zeros);
  // mat eps_star(q, n, arma::fill::zeros);
  // mat Z(n, q);
  // double z;

  // Initialize variables
  for(b=0; b < nblocks; b++){
    // Dynamically allocate ragged arrays
    currBlockSize   = blockSizes[b];

    R[b]            = mat(currBlockSize, currBlockSize);
    blocks_pivot[b] = ivec(currBlockSize);
    parbar_chunk[b] = vec(currBlockSize, arma::fill::zeros);
    zsamp[b]        = vec(currBlockSize, arma::fill::zeros);
    par_incr[b]     = vec(currBlockSize, arma::fill::zeros);

    chunk_size[b] = 0;
    acpt_chunk[b] = 0.0;

    R[b] = arma::chol(S[b], "upper");
    frac[b] = sqrt(1.0 / ((double) refresh_counter[b] + 1.0));
  }

  GetRNGstate();

  // Use Rcpp sugar random number generation functions
  // Pre-allocate random numbers used in MCMC to speed execution
  //  at cost of increased memory
  logUniform = arma::log(as<arma::vec>(runif(logUniform.size())));
  gammaAdj   = as<arma::vec>(rgamma(gammaAdj.size(), 3.0, 1.0));

  parStore.col(0) = parSample;
  lpval = logPosterior(parSample, FALSE);
  if(verbose) Rcout << "Initial lp = " << lpval << std::endl;

  // log_det_current = log(det(Rcorr));

  // Adaptive Metropolis MCMC
  for(iter = 0; iter < niter; iter++){
    // Model parameters other than copula parameters
    for(b = 0; b < nblocks; b++){
      // Sample new parameters for variables in block b
      currBlockSize = blockSizes[b];
      chunk_size[b]++;

      zsamp[b]      = as<arma::vec>(rnorm(currBlockSize)); // Use Rcpp sugar rnorm
      par_incr[b]   = trimatu(R[b]) * zsamp[b];
      lambda        = lm[b] * sqrt(3.0 / gammaAdj[g++]);

      parOld = parSample;
      parSample.elem(blocks[b]) += lambda * par_incr[b];

      // Evaluate loglikelihood at proposed parameters
      lpvalnew = logPosterior(parSample, FALSE);
      lp_diff  = lpvalnew - lpval;

      alpha[b] = exp(lp_diff);
      if(alpha[b] > 1.0)  alpha[b] = 1.0;
      if(isnan(alpha[b])) alpha[b] = 0.0;

      // Check for acceptance
      if(logUniform[u++] < lp_diff){
        // Accept
        lpval = lpvalnew;
      }
      else{
        // Reject
        parSample = parOld;
      }

      alpha_run[b] = ((double)run_counter[b] * alpha_run[b] + alpha[b]) / ((double)run_counter[b] + 1.0);
      run_counter[b]++;
    }

    if(!fix_corr){
      // Sample new correlation matrix

      // Liu & Daniels
      for(i=0; i < q; i++){
        for(j=0; j < n; j++){
          D.at(i,i) += pow(tau_y_x.at(j,i),2);
        }
      }
      epsilon = tau_y_x * D;

      for(j=0; j < n; j++){
        S_samp += epsilon.row(j).t() * epsilon.row(j);
      }
      Sigma  = rInvWish(n, S_samp);
      D_inv  = diagmat(pow(Sigma.diag(), -0.5));
      R_prop = D_inv * Sigma * D_inv;
      log_det_proposed = log(det(R_prop));
      accept_prob = std::min(1.0, exp(-((q+1)/2) * (log_det_proposed - log_det_current)));
      if(as<double>(runif(1)) < accept_prob){
        Rcorr = R_prop;
        Rcout << "Corr = " << Rcorr.at(0,1) << std::endl;
        log_det_current = log_det_proposed;
        lpval = logPosterior(parSample, FALSE);
      }

      // Barnard et al.
      // for(i = 0; i < n; i++){
      //   for(j = 0; j < q; j++){
      //     z = R::qnorm(tau_y_x.at(i,j), 0.0, 1.0, 1, 0);
      //     if(z == std::numeric_limits<double>::infinity()){
      //       z = 10;
      //     }
      //     else if(z == -std::numeric_limits<double>::infinity()){
      //       z = -10;
      //     }
      //     Z.at(i,j) = z;
      //   }
      // }
      //
      // R_inv      = arma::inv_sympd(Rcorr);
      // alpha_samp = as<arma::vec>(rgamma(q, (double)(q+1)/2, 1.0));
      // D          = diagmat(sqrt(R_inv.diag() / (2*alpha_samp)));
      // D_inv      = arma::inv(D);
      // eps_star   = D * Z.t();
      // S_samp     = eps_star * eps_star.t();
      // Sigma      = rInvWish(n + q + 1, S_samp);
      // D_inv      = diagmat(pow(Sigma.diag(), -0.5));
      // Rcorr      = D_inv * Sigma * D_inv;
      //
      // lpval = logPosterior(parSample, FALSE);
    }

    // Store results at appropriate iterations
    if((iter+1) % thin == 0){
      for(i=0; i < q-1; i++){
        for(j=i+1; j < q; j++){
          parSample.at(nrespar + i*q + j - 1) = Rcorr.at(i,j);
        }
      }
      lpSample[store_lp++]           = lpval;
      parStore.col(store_par++)      = parSample;
      acceptSample.row(store_acpt++) = alpha.t();
    }

    // Output status updates for users
    if(verbose){
      if(niter < ticker || (iter+1) % (niter / ticker) == 0){
        Rcout << "iter = " << iter + 1 << ", lp = " << lpval << std::endl;
        alpha_run.t().print("alpha_run");

        alpha_run.zeros();  // reinit to 0 for each block
        run_counter.zeros();
      }
    }

    // Adapt covariance matrices for proposal distributions for each block
    for(b=0; b < nblocks; b++){
      chs = std::max((double) chunk_size[b], 1.0);

      acpt_chunk[b]   = acpt_chunk[b] + (alpha[b] - acpt_chunk[b]) / chs;
      parbar_chunk[b] = parbar_chunk[b] + (parSample.elem(blocks[b]) - parbar_chunk[b]) / chs;

      if(chunk_size[b] == refresh * blockSizes[b]){
        refresh_counter[b]++;
        frac[b] = sqrt(1.0 / ((double) refresh_counter[b] + 1.0));

        for(i = 0; i < blockSizes[b]; i++){
          for(j = 0; j < i; j++){
            S[b].at(i,j) = (1.0 - frac[b]) * S[b].at(i,j) + frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][j] - mu[b][j]);
            S[b].at(j,i) = S[b].at(i,j);
          }
          S[b].at(i,i) = (1.0 - frac[b]) * S[b].at(i,i) + frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][i] - mu[b][i]);
        }
        R[b]  = arma::chol(S[b], "upper");

        mu[b] = mu[b] + frac[b] * (parbar_chunk[b] - mu[b]);
        lm[b] = lm[b] * exp(frac[b] * (acpt_chunk[b] - acpt_target[b]));

        acpt_chunk[b] = 0;
        parbar_chunk[b].zeros();
        chunk_size[b] = 0;
      }
    }
  }
  PutRNGstate();
}

mat rInvWish(const double nu, const mat &S){
  int i, j;

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


// [[Rcpp::export]]
SEXP devianceCalc(SEXP pars_,
                  SEXP x_,
                  SEXP y_,
                  SEXP hyper_,
                  SEXP dim_,
                  SEXP A_,
                  SEXP R_,
                  SEXP LOGDET_,
                  SEXP LPGRID_,
                  SEXP tauG_){
  /***** Initialization *****/
  // Temp variables
  int reach, i;
  NumericMatrix tempA;
  NumericMatrix tempR;
  mat posteriorSamples;

  // Input data
  // Convert SEXP objects to Rcpp objects
  NumericMatrix PARS   = as<NumericMatrix>(pars_);
  NumericMatrix X      = as<NumericMatrix>(x_);
  NumericMatrix Y      = as<NumericMatrix>(y_);
  NumericVector HYP    = as<NumericVector>(hyper_);
  IntegerVector DIM    = as<IntegerVector>(dim_);
  List Ag              = as<List>(A_);
  List Rg              = as<List>(R_);
  NumericVector LOGDET = as<NumericVector>(LOGDET_);
  NumericVector LPGRID = as<NumericVector>(LPGRID_);
  NumericVector TAU_G  = as<NumericVector>(tauG_);

  x       = mat(X.begin(), X.nrow(), X.ncol(), TRUE);
  y       = mat(Y.begin(), Y.nrow(), Y.ncol(), TRUE);
  posteriorSamples = mat(PARS.begin(), PARS.nrow(), PARS.ncol(), TRUE);
  taugrid = vec(TAU_G.begin(), TAU_G.size(), TRUE);
  ldRgrid = vec(LOGDET.begin(), LOGDET.size(), TRUE);
  lpgrid  = vec(LPGRID.begin(), LPGRID.size(), TRUE);

  // Dimensions
  n     = DIM[0];
  p     = DIM[1];
  q     = DIM[2];
  L     = DIM[3];
  mid   = DIM[4];
  m     = DIM[5];
  G     = DIM[6];
  nkap  = DIM[7];
  ncorr = DIM[11];

  par_position   = (p+1)*m;
  gam_pos_offset = p - 1;

  // MCMC parameters
  niter   = DIM[8];
  thin    = DIM[9];
  nsamp   = DIM[10];
  npar    = (m+1) * (p+1) + 2;
  nrespar = q*npar;
  totpar  = nrespar + ncorr;

  // Prior parameters
  Agrid   = cube(L, m, G);
  Rgrid   = cube(m, m, G);

  akap  = vec(nkap);
  bkap  = vec(nkap);
  lpkap = vec(nkap);

  // Initialize mixture components for prior on kappa_j
  for(reach = 2, i = 0; i < nkap; i++){
    akap[i]  = HYP[reach++];
    bkap[i]  = HYP[reach++];
    lpkap[i] = HYP[reach++];
  }

  for(i = 0; i < G; i++){
    tempA = as<NumericMatrix>(Ag[i]);
    Agrid.slice(i) = mat(tempA.begin(), tempA.nrow(), tempA.ncol(), TRUE);

    tempR = as<NumericMatrix>(Rg[i]);
    Rgrid.slice(i) = mat(tempR.begin(), tempR.nrow(), tempR.ncol(), TRUE);
  }

  // Parameters
  gam0     = 0;
  sigma    = 1;
  nu       = 1;
  gam      = vec(p, fill::zeros);
  zeta0    = vec(L, fill::zeros);
  zeta0dot = vec(L, fill::zeros);

  w0       = mat(L, q, fill::zeros);
  wMat     = cube(L, p, q, fill::zeros);
  vMat     = cube(L, p, q, fill::zeros);

  Rcorr    = mat(q, q, fill::eye);

  // Intermediate calculations
  tau_y_x  = mat(n, q, fill::zeros);
  wgrid    = mat(L, G, fill::zeros);
  a        = mat(n, L, fill::zeros);
  aX       = vec(L, fill::zeros);
  aTilde   = mat(n, L, fill::zeros);
  vTilde   = mat(L, p, fill::zeros);
  b0dot    = vec(L, fill::zeros);
  bdot     = mat(L, p, fill::zeros);
  Q0_med   = vec(n, fill::zeros);
  vNormSq  = vec(L, fill::zeros);
  wknot    = vec(m, fill::zeros);
  zknot    = vec(m, fill::zeros);
  llmat    = mat(n, q+1, fill::zeros);
  llgrid   = vec(G, fill::zeros);

  lps0     = vec(q, fill::zeros);
  resLin   = vec(n, fill::zeros);
  Q0Pos    = vec(L, fill::zeros);
  Q0Neg    = vec(L, fill::zeros);
  bPos     = mat(L, p, fill::zeros);
  bNeg     = mat(L, p, fill::zeros);

  pgvec    = cube(G, p+1, q, fill::zeros);
  lb       = vec(10, fill::zeros);
  lw       = vec(nkap, fill::zeros);

  // Output
  vec devSample = vec(niter, fill::zeros);    // deviance
  mat llSample  = mat(n, niter, fill::zeros); // observation log-likelihood
  field<cube> pgSample(niter);                // posterior lambda values

  for(int iter = 0; iter < niter; iter++){
    // Read in draw from posterior
    parSample = conv_to< vec >::from(posteriorSamples.row(iter));

    // Convert portion of parameter vector corresponding to correlation matrix
    // to actual matrix
    reach = 0;
    for(int i = 0; i < (q-1); i++){
      for(int j = i+1; j < q; j++){
        Rcorr.at(i,j) = parSample[q*npar + reach];
        reach++;
      }
    }
    Rcorr = symmatu(Rcorr);

    // Store deviance
    devSample[iter] = -2.0 * logPosterior(parSample, true);

    // Calculate log-likelihood contribution from each observation from llmat
    llSample.col(iter) = sum(llmat, 1);

    // Store posterior weights on the lambda grid for each GP function for
    // each response
    pgSample(iter) = pgvec;
  }

  return Rcpp::List::create(Rcpp::Named("devsamp") = devSample,
                            Rcpp::Named("llsamp") = llSample,
                            Rcpp::Named("pgsamp") = pgSample);
}
