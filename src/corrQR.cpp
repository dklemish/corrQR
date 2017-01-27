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
double logPosterior(const vec &, bool llonly);

double log_dGaussCopula(const mat &, const mat &);

void MCMC(void);
void adMCMC(void);

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

void Init_Prior_Param(int L, int m, int G, int nblocks, int nkap, NumericVector hyp,
                      List Ag, List Rg, List muV, List SV,
                      IntegerVector blocks_, IntegerVector bSize);

/**** Global variables ****/

// int par_position = (p+1)*m, gam_pos_offset = p - 1;
int par_position;
int gam_pos_offset;
double zeta0tot, lps0 = 0;
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
int totpar;   // # of total parameters
int nblocks;  // # of MH blocks
bool verbose; // flag to print intermediate calc's
bool adapt;   // flag for whether adaptive MCMC should be used
bool fix_corr;// flag for whether correlation parameters should be
//  fixed (true) or learned via MCMC (false)
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
// mat aTilde;   // = (n x L)
mat vTilde;   // = (p x L)
vec b0dot;    // (L x 1)
mat bdot;     // = x^T %*% beta.dot(tau) (p x L)
vec vNormSq;
vec wknot;    // (m x 1)
vec zknot;    // (m x 1)
mat llmat;    // log-likelihood by obs (n x q)
vec llgrid;   // MV t-dist contrib to loglikelihood (G x 1)
ivec zeta0_tick; // (L x 1)
vec  zeta0_dist; // (L x 1)

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
  totpar  = q*npar + ncorr;
  nblocks = IMCMCPAR[0];
  refresh = IMCMCPAR[1];
  verbose = (bool) IMCMCPAR[2];
  adapt   = (bool) IMCMCPAR[4 + nblocks];
  fix_corr= (bool) IMCMCPAR[5 + nblocks];
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
  // aTilde   = mat(n, L, fill::zeros);
  vTilde   = mat(p, L, fill::zeros);
  b0dot    = vec(L, fill::zeros);
  bdot     = mat(p, L, fill::zeros);
  vNormSq  = vec(L, fill::zeros);
  wknot    = vec(m, fill::zeros);
  zknot    = vec(m, fill::zeros);
  llmat    = mat(n, q, fill::zeros);
  llgrid   = vec(G, fill::zeros);

  zeta0_tick = ivec(L, fill::zeros);
  zeta0_dist = vec(L, fill::zeros);

  pgvec    = cube(G, p+1, q, fill::zeros);
  lb       = vec(10, fill::zeros);
  lw       = vec(nkap, fill::zeros);

  parSample = vec(PAR.begin(), PAR.size(), TRUE);

  // Output
  lpSample     = vec(nsamp, fill::zeros);           // stored log-likelihood
  acceptSample = mat(nsamp, nblocks, fill::zeros);  // stored MH acceptance history
  parStore     = mat(totpar, nsamp, fill::zeros);     // stored posterior draws npar x nsamp

  int reach = 0;
  for(int i = 0; i < (q-1); i++){
    for(int j = i+1; j < q; j++){
      Rcorr.at(i,j) = parSample(q*npar + reach);
      reach++;
    }
  }
  Rcorr = symmatu(Rcorr);

  /********* TEST CODE *******/
  Rcorr.print("R correlation matrix");

  //parSample.t().print("parSample");

  //double junk = ppFn(parSample, 1, 0);

  double junk = logPosterior(parSample, false);
  llmat.print("llmat");
  tau_y_x.print("tau_y_x");
  Rcout << "log copula pdf = " << log_dGaussCopula(tau_y_x, Rcorr) << std::endl;
  Rcout << "initial logPosterior = " << junk << std::endl;

  //
  // vec U = vec(2); U[0] = 0.8; U[1] = 0.9;
  //
  // U.print("U");
  //
  // Rcout << "dGaussCopula(U) = " << dGaussCopula(U, Rcorr, false) << std::endl;
  // Rcout << "log dGaussCopula(U) = " << dGaussCopula(U, Rcorr, true) << std::endl;
  //
  // parSample.subvec(0, m-1).print("w0 parms");
  // Rcout << std::endl;
  // double junk = ppFn0(parSample, 0);
  // Rcout << "junk = " << junk << std::endl;
  // w0.print("w0");
  // Rcout << std::endl;
  //
  // parSample.subvec(30, 35).print("w0 parms response 1");
  // Rcout << std::endl;
  // junk += ppFn0(parSample, 1);
  // Rcout << "junk = " << junk << std::endl;
  // w0.print("w0");
  // Rcout << std::endl;
  //
  // vec test = vec(L, fill::zeros);
  // trape(w0.memptr(), taugrid.memptr(), L, test.memptr());
  // Rcout << "Length of test = " << test.size() << std::endl;
  // test.print("Integral test:");
  //
  // mat test = mat(2,2);
  // test(0,0)=1; test(1,0)=0.5; test(0,1)=0.5; test(1,1)=1;
  // mat test2 = mat(2,2, fill::zeros);
  //
  // rCorrMat(2, 100, test, test2);
  // test2.print("test2 after function call");
  //
  // rCorrMat(2, 100, test2, test2);
  // test2.print("test2 after function call");
  //
  // rCorrMat(2, 100, test2, test2);
  // test2.print("test2 after function call");
  //
  // for(int i = 0; i < 100; i++){
  //   rCorrMat(2, 100, test2, test2);
  // }
  //
  // test2.print("test2 after 100 function call");
  //
  if(adapt == false){
    //MCMC();
  }
  else{
    //adMCMC();
  }

  //PutRNGstate();

  return Rcpp::List::create(Rcpp::Named("parsamp") = parStore.t(),
                            Rcpp::Named("lpsamp") = lpSample,
                            Rcpp::Named("acpt") = acceptSample);
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

double log_dGaussCopula(const mat &U, const mat &Rc){
  // Note no checks that U contains values in [0,1]
  int i, j;
  double d = det(Rc);
  double result = 0;

  vec phi_inv_u = vec(q);

  for(i = 0; i < n; i++){
    for(j = 0; j < q; j++){
      phi_inv_u[j] = R::qnorm(U.at(i,j), 0, 1, 1, 0);
    }
    result += -0.5 * (log(d) + as_scalar(phi_inv_u.t() * (Rc.i() - eye(q,q)) * phi_inv_u));
  }

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
    lps0 = ppFn0(par, k);

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
      lps0 += ppFn(par, j, k);
    }

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

    //wMat.slice(k).print("wMat");
    //zeta0.print("zeta0");
    vMat.slice(k).print("vMat");

    //** 5. ||v_l||^2 for each quantile l (1:L)
    for(l = 0; l < L; l++)
      vNormSq[l] = pow(norm(vMat.slice(k).row(l)),2);

    vNormSq.print("vNormSq");

    if(vNormSq.min() > 0.0){
      //** 5. a_i for each observation i & quantile l
      a = x * vMat.slice(k).t();  // a is n x L

      //** 6. a_chi for each l & tilde{a}_il for each obs i & quant l
      for(l = 0; l < L; l++){
        aX[l] = -a.col(l).min() / sqrt(vNormSq[l]);
        for(i = 0; i < p; i++){
          vTilde.at(i,l) = vMat.at(i,l,k) / (aX[l]*sqrt(1+vNormSq[l]));
        }
      }

      a.print("a");
      aX.print("aX");
      vTilde.t().print("vTilde");

      /*
       // Compute modeled median (specifically the quantile of Y at tau_0 = F_0(0),
       // but our prior guess for F is a t-distribution, so F_0(0) = 0.5) of Y | X
       // for each observation i
       //** 7. Calculate log-likelihood by sequencing through observations
        for(i = 0; i < n; i++){
        Q0_val = gam0 + dot(x.row(i), gam);

        y_i = y.at(i,k);

        if(y_i == Q0_val){
        // Y_i exactly equals modeled median, conditional on X_i
        llmat.at(i,k) = -log(b0dot[mid] + dot(x.row(i), bdot.col(mid)));
        tau_y_x.at(i, k) = taugrid[mid];
        }
        else if(y_i > Q0_val){
        // Y_i > median
        Q_U = Q0_val;
        l = mid;

        while(y_i > Q_U && l < L-1){
        Q_L = Q_U;
        Q_U = Q_L + 0.5*(taugrid[l]-taugrid[l-1]) *
        (b0dot[l]*(1+aTilde.at(i,l)) + b0dot[l-1]*(1+aTilde.at(i,l-1)));
        l++;
        }
        if(l == L){
        Q_U = std::numeric_limits<double>::infinity();
        }
        tau_y_x.at(i, k) = taugrid[l];
        }
        else {
        l = mid + 1;
        Q_L = Q0_val;

        while(y_i < Q_L && l > 0){
        Q_U = Q_L;
        Q_L = Q_U - 0.5*(taugrid[l]-taugrid[l-1]) *
        (b0dot[l]*(1+aTilde.at(i,l)) + b0dot[l-1]*(1+aTilde.at(i,l-1)));
        l--;
        }
        if(l == 0){
        Q_L = -std::numeric_limits<double>::infinity();
        }
        tau_y_x.at(i, k) = taugrid[l];
        }

        if(Q_L == -std::numeric_limits<double>::infinity() ||
        Q_U == std::numeric_limits<double>::infinity()){
        llmat.at(i,k) = -std::numeric_limits<double>::infinity();
        }
        else{
        alp = (y_i - Q_L) / (Q_U - Q_L);
        llmat.at(i,k) += -1*log((1-alp)*b0dot[l-1]*(1+aTilde.at(i,l-1)) +
        alp*b0dot[l] * (1+aTilde.at(i,l)));
        }
        }
        */
    }
  }

  // Calculate contribution to loglikelihood from marginal distributions for
  // each response
  lp = arma::accu(llmat);
  if(std::isnan(lp)) lp = -std::numeric_limits<double>::infinity();

  // Calculate contribution to loglikelihood from copula distribution
  lp += log_dGaussCopula(tau_y_x, Rcorr);

  if(!llonly){
    lp += lps0 + R::dlogis(nu, 0.0, 1.0, 1);
  }

  return lp;

  //return 0;
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
  int b;

  Rcout << "Start of MCMC" << std::endl;
  GetRNGstate();

  int u = 0;
  int iter;
  int store_lp=0;
  int store_par=0;
  int store_acpt=0;
  int currBlockSize=0;

  double lpval = 0;
  double lpvalnew = 0;
  double lp_diff;

  ivec blocks_rank(nblocks);
  ivec run_counter(nblocks, arma::fill::zeros);
  vec  alpha(nblocks, arma::fill::zeros);
  vec  alpha_run(nblocks, arma::fill::zeros);

  vec blocks_d(npar);
  vec parOld(npar);
  vec logUniform(nblocks*niter);

  // Ragged arrays
  std::vector<ivec> blocks_pivot(nblocks);
  std::vector<vec>  parbar_chunk(nblocks);
  std::vector<vec>  zsamp(nblocks);
  std::vector<vec>  par_incr(nblocks);
  std::vector<mat>  R(nblocks); // Chol factor of MCMC block proposal covariances

  // Initialize variables
  for(b=0; b < nblocks; b++){
    currBlockSize   = blockSizes[b];
    blocks_pivot[b] = ivec(currBlockSize);
    parbar_chunk[b] = vec(currBlockSize, arma::fill::zeros);
    zsamp[b]        = vec(currBlockSize, arma::fill::zeros);
    par_incr[b]     = vec(currBlockSize, arma::fill::zeros);

    R[b] = mat(blockSizes[b], currBlockSize); // allocate memory for arma matrix
    R[b] = arma::chol(S[b], "upper");
  }

  // Use Rcpp sugar random number generation functions
  // Pre-allocate random numbers used in MCMC to speed execution
  //  at cost of increased memory
  logUniform = arma::log(as<arma::vec>(runif(logUniform.size())));

  parStore.col(0) = parSample;
  lpval = logPosterior(parSample, FALSE);
  if(verbose) Rcout << "Initial lp = " << lpval << std::endl;

  // Metropolis MCMC
  for(iter = 0; iter < niter; iter++){
    // Model parameters other than copula parameters
    for(b = 0; b < nblocks; b++){
      // Sample new parameters for variables in block b
      currBlockSize = blockSizes[b];
      zsamp[b]      = as<arma::vec>(rnorm(currBlockSize)); // Use Rcpp sugar rnorm
      par_incr[b]   = trimatu(R[b]) * zsamp[b];

      parOld = parSample;
      parSample.elem(blocks[b]) += par_incr[b];

      // Evaluate loglikelihood at proposed parameters
      lpvalnew = logPosterior(parSample, FALSE);

      lp_diff = lpvalnew - lpval;

      // Acceptance probability
      alpha[b] = exp(lp_diff);
      if(alpha[b] > 1.0)
        alpha[b] = 1.0;

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

    if(fix_corr == false){
      // Input code to sample correlation matrix and accept via MH
    }
    // Store results at appropriate iterations
    if((iter+1) % thin == 0){
      lpSample[store_lp++] = lpval;
      parStore.col(store_par++) = parSample;
      acceptSample.row(store_acpt++) = alpha.t();
    }

    // Output status updates for users
    if(verbose){
      if(niter < ticker || (iter+1) % (niter / ticker) == 0){
        Rcout << "iter = " << iter + 1 << ", lp = " << lpval << ", ";
        alpha_run.t().print("acpt = ");
        for(b = 0; b < nblocks; b++){
          alpha_run[b] = 0.0;
          run_counter[b] = 0;
        }
      }
    }
  }
  PutRNGstate();
}


void adMCMC(void){
  int b, i, j;

  Rcout << "Start of adMCMC" << std::endl;

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
  vec alpha(nblocks);
  vec alpha_run(nblocks, arma::fill::zeros);

  vec blocks_d(npar);
  vec parOld(npar);
  vec logUniform(nblocks*niter);

  ivec chunk_size(nblocks, arma::fill::zeros);
  vec acpt_chunk(nblocks, arma::fill::zeros);
  vec frac(nblocks);
  vec gammaAdj(nblocks*niter);

  // Rcout << "1" << std::endl;

  // Ragged arrays
  std::vector<ivec> blocks_pivot(nblocks);
  std::vector<vec>  parbar_chunk(nblocks);
  std::vector<vec>  zsamp(nblocks);
  std::vector<vec>  par_incr(nblocks);
  std::vector<mat>  R(nblocks); // Chol factor of MCMC block proposal covariances

  // Initialize variables
  for(b=0; b < nblocks; b++){
    currBlockSize = blockSizes[b];
    blocks_pivot[b] = ivec(currBlockSize);
    parbar_chunk[b] = vec(currBlockSize, arma::fill::zeros);
    zsamp[b]        = vec(currBlockSize, arma::fill::zeros);
    par_incr[b]     = vec(currBlockSize, arma::fill::zeros);

    R[b] = mat(blockSizes[b], currBlockSize);
    R[b] = arma::chol(S[b], "upper");
    frac[b] = sqrt(1.0 / ((double) refresh_counter[b] + 1.0));
  }

  Rcout << "Block sizes = " << blockSizes << std::endl;

  // Use Rcpp sugar random number generation functions
  // Pre-allocate random numbers used in MCMC to speed execution
  //  at cost of increased memory
  logUniform = arma::log(as<arma::vec>(runif(logUniform.size())));
  gammaAdj   = as<arma::vec>(rgamma(gammaAdj.size(), 3.0, 1.0));

  Rcout << "2" << std::endl;

  parStore.col(0) = parSample;
  lpval = logPosterior(parSample, FALSE);
  if(verbose) Rcout << "Initial lp = " << lpval << std::endl;

  Rcout << "3" << std::endl;

  // Adaptive Metropolis MCMC
  for(iter = 0; iter < niter; iter++){
    // Rcout << "Iteration: " << iter << ". lp = " << lpval << std::endl;
    // lm.t().print("lm");

    // Model parameters other than copula parameters
    for(b = 0; b < nblocks; b++){
      // Sample new parameters for variables in block b
      currBlockSize = blockSizes[b];
      zsamp[b]      = as<arma::vec>(rnorm(currBlockSize)); // Use Rcpp sugar rnorm
      par_incr[b]   = trimatu(R[b]) * zsamp[b];
      lambda        = lm[b] * sqrt(3.0 / gammaAdj[g++]);

      parOld = parSample;
      parSample.elem(blocks[b]) += lambda * par_incr[b];

      chunk_size[b]++;

      // Evaluate loglikelihood at proposed parameters
      lpvalnew = logPosterior(parSample, FALSE);

      lp_diff = lpvalnew - lpval;

      alpha[b] = exp(lp_diff);
      if(alpha[b] > 1.0)
        alpha[b] = 1.0;

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

    // Store results at appropriate iterations
    if((iter+1) % thin == 0){
      lpSample[store_lp++] = lpval;
      parStore.col(store_par++) = parSample;
      acceptSample.row(store_acpt++) = alpha;
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
        // Rcout << "Adjusting covariance matrices!" << std::endl;
        refresh_counter[b]++;
        frac[b] = sqrt(1.0 / ((double) refresh_counter[b] + 1.0));

        for(i = 0; i < blockSizes[b]; i++){
          for(j = 0; j < i; j++){
            S[b].at(i,j) = (1.0 - frac[b]) * S[b].at(i,j) +
              frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][j] - mu[b][j]);
            S[b].at(j,i) = S[b].at(i,j);
          }
          S[b].at(i,i) = (1.0 - frac[b]) * S[b].at(i,i) +
            frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][i] - mu[b][i]);
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
}
