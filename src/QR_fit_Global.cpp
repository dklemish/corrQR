#define sqrt5 2.236067977499789696

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
void Init_Prior_Param(int, int, int, int, int, NumericVector, NumericMatrix,
                      NumericVector, NumericVector, IntegerVector, 
                      IntegerVector, IntegerVector, bool);
bool Stopping_Rule(double, double , double);
double sigFn(double);
double sigFn_inv(double);
double nuFn(double);
double nuFn_inv(double);
vec unitFn(vec);
vec Q0(vec, double);
vec q0(vec, double);
vec F0(vec, double);
vec q0tail(vec);
double Q0tail(double);
double lf0tail(double);
double ppFn0(vec &);
double ppFn(vec &, int);
double logsum(vec);
double logmean(vec);
void adMCMC(void);
double dbase_joint_scl(double, vec &);
double dbasejoint(vec &);
void trape(double *x, double *h, int length, double *integral);
/*
double lpFn(vec &, double temp);
double lpFn1(vec &);
double lpFn2(double sigma);
double logpostFn(vec &, double temp, bool llonly);
*/
double lpFn(vec, double temp);
double lpFn1(vec);
double lpFn2(double sigma);
double logpostFn(vec, double temp, bool llonly);
void Adjust_Sigma(double (*f)(double), double &a, double &b, double tolerance);
void qr_fit_cpp(NumericVector, NumericMatrix, NumericVector, IntegerVector, 
                int, NumericVector, IntegerVector, NumericMatrix,
                NumericVector, NumericVector, NumericVector,
                NumericVector, IntegerVector, IntegerVector, 
                NumericVector, IntegerVector);

/**** Global variables ****/
// Data
mat x;        // covariates (n x p)
vec y;        // response (n x 1)
vec taugrid;  // quantiles of interest (L x 1)

// Dimension parameters
int n;    // # obs
int p;    // # predictors
int L;    // # tau's (quantiles) at which betas are estimated
int mid;  // location in taugrid of median (0.50)
int m;    // dim of low rank approx to w functions
int G;    // dim G of approx to pi(lambda)
int nkap; // # of mixture components for prior on kappa_j

// MCMC parameters
int niter;    // MCMC iterations
int thin;     // thinning factor
int nsamp;    // # of MCMC samples to keep
int npar;     // # parameters to track
int nblocks;  // # of MH blocks
int refresh;  // adaption rate of adaptive MH
bool verbose; // flag to print intermediate calc's
int ticker;   // how often to update screen
double temp;
double decay;
ivec refresh_counter; //
vec acpt_target;    // 
vec lm;             //

// Prior parameters
cube Agrid;   // storage for A_g = C_0*(g) C**^{-1}(g) (L x m x G)
cube Rgrid;   // storage for R_g = chol(C**(g)) (m x m x G)
vec ldRgrid;  // log trace(R_g), normalizing constant
vec lpgrid;   // log prior (lambda_j)

// vector of vector / matrices allows for jagged sizes
std::vector<vec> mu; // init MH block means
std::vector<mat> S;  // initial MH block covariances
std::vector<uvec> blocks;

ivec blockSizes;
ivec cens;
bool shrink;

// Hyperparameters
double asig;
double bsig; // hyperparms for sigma
vec akap;
vec bkap;
vec lpkap;   // hyperparms for kappa

// Model parameters
double gam0;  // curr value of gamma_0
double sigma; // curr value of sigma
double nu;    // curr value of nu

vec gam;      // current value of vector gamma (p x 1)
vec w0;       // current values of w_0(tau_l) (L x 1)
vec zeta0;    // current values of zeta(tau_l) (L x 1)
vec zeta0dot; // current values of zeta.dot(tau_l) (L x 1)
mat wMat;     // each col corresponds to current values of W_j (p x L)
mat vMat;     // each col corresponds to W_j(zeta(tau)) (p x L)  

// Intermediate calculations
mat wgrid;    // temp to store A_g * low rank w knots for each g
mat a;        // = (x*vMat) (n x L)
mat aTilde;   // = (n x L)
mat vTilde;   // = (p x L)
vec b0dot;    // (L x 1)
mat bdot;     // = x^T %*% beta.dot(tau) (p x L)
mat bPos;     // (mid+1 x L)
mat bNeg;     // (mid+1 X L)
vec vNormSq;
vec zknot;    // (m x 1)
vec Q0vec;    // corresponds to Q0_i (n x 1)
vec resLin;   // (n x 1)
vec llvec;    // log-likelihood by obs (n x 1)
vec llgrid;   // MV t-dist contrib to loglikelihood (G x 1)

mat pgvec;    // contains p_g(W_j*) (G x (p+1))
vec lb;       // used for shrinkage prior on gamma vector (10 x 1)
vec lw;       // used for interpolation of marginal GP (nkap x 1)

vec par0;      // used in initial adjustment of sigma
vec parSample; // current parameter draw

// Outputs
vec lpSample;     // stored log-likelihood
mat acceptSample; // stored MH acceptance history
mat parStore;     // stored posterior draws npar x nsamp

// [[Rcpp::export]] 
void qr_fit_cpp(NumericVector par_,
                NumericMatrix x_, 
                NumericVector y_, 
                IntegerVector cens_, 
                int shrink_,
                NumericVector hyper,
                IntegerVector dim_,
                NumericMatrix gridpars,
                NumericVector tauG,
                NumericVector siglim,
                NumericVector muV,
                NumericVector SV,
                IntegerVector blocks_,
                IntegerVector blockSizes_,
                NumericVector dmcmcpar,
                IntegerVector imcmcpar){
  /***** Initialization *****/
  // Input data
  x = mat(x_.begin(), x_.nrow(), x_.ncol(), TRUE);
  y = vec(y_.begin(), y_.size(), TRUE);
  taugrid = vec(tauG.begin(), tauG.size(), TRUE);
  
  // Dimensions
  n = dim_[0];
  p = dim_[1];
  L = dim_[2];
  mid = dim_[3];
  m = dim_[4];
  G = dim_[5];
  nkap = dim_[6];
  
  // MCMC parameters
  niter = dim_[7];
  thin = dim_[8];
  nsamp = dim_[9];
  npar = (m+1) * (p+1) + 2;
  nblocks = imcmcpar[0];
  refresh = imcmcpar[1];
  verbose = (bool) imcmcpar[2];
  ticker = imcmcpar[3];
  temp = dmcmcpar[0];
  decay = dmcmcpar[1];
  
  refresh_counter = ivec(imcmcpar.begin() + 4, nblocks, TRUE);
  acpt_target = vec(dmcmcpar.begin() + 2, nblocks, TRUE);
  lm = vec(dmcmcpar.begin() + 2 + nblocks, nblocks, TRUE);
  
  // Prior parameters
  Init_Prior_Param(L, m, G, nblocks, nkap, hyper, gridpars, muV, SV, 
                   blocks_, blockSizes_, cens_, (bool)shrink_);
  
  // Hyperparameters
  double a_siglim = siglim[0];
  double b_siglim = siglim[1];  
  
  // Parameters
  gam0 = 0;
  sigma = 1;
  nu = 1;
  gam = vec(p);
  w0  = vec(L);
  zeta0 = vec(L);
  zeta0dot = vec(L);
  wMat = mat(p, L);
  vMat = mat(p, L);
  
  // Intermediate calculations
  wgrid = mat(L, G, fill::zeros);
  a = mat(n, L, fill::zeros);
  aTilde = mat(n, L, fill::zeros);
  vTilde = mat(p, L, fill::zeros);
  b0dot = vec(L, fill::zeros);
  bdot = mat(p, L, fill::zeros);
  bPos = mat(p, mid+1, fill::zeros);
  bNeg = mat(p, mid+1, fill::zeros);
  vNormSq = vec(L, fill::zeros);
  zknot = vec(m, fill::zeros);
  Q0vec = vec(n, fill::zeros);
  resLin = vec(n,fill::zeros);
  llvec = vec(n, fill::zeros);
  llgrid = vec(G, fill::zeros);
  
  pgvec = mat(G, p+1, fill::zeros);
  lb = vec(10, fill::zeros);
  lw = vec(nkap, fill::zeros);
  
  par0      = vec(par_.begin(), par_.size(), TRUE);
  parSample = vec(par_.begin(), par_.size(), TRUE);
  
  // Options
  // const double shrinkFactor = 1; 
  
  // Output
  lpSample     = vec(nsamp, fill::zeros);           // stored log-likelihood
  acceptSample = mat(nsamp, nblocks, fill::zeros);  // stored MH acceptance history
  parStore     = mat(npar, nsamp, fill::zeros);     // stored posterior draws npar x nsamp
  
  //shrinkFactor = shrinkFn((double)p);
  // Adjust starting value of sigma

   //Rcout << "sigma before adjust: " << parSample[(m+1)*(p+1)] << std::endl;
   //parSample.t().print("parSample before adjust:");
   Adjust_Sigma(lpFn2, a_siglim, b_siglim, 1.0e-5);
   
   parSample[(m+1)*(p+1)] = (a_siglim + b_siglim)/2;
   
   //Rcout << "sigma after adjust: " << parSample[(m+1)*(p+1)] << std::endl;
   //parSample.t().print("parSample after adjust:");
   
   adMCMC();
   //parSample.t().print("last Sample");
}

void Init_Prior_Param(int L, int m, int G, int nblocks, int nkap, NumericVector hyp,
                      NumericMatrix gridpars, NumericVector muV, NumericVector SV,
                      IntegerVector blocks_, IntegerVector bSize, IntegerVector cen,
                      bool shrink_){
  
  int reach, i, l, k, b, mu_point, S_point, block_point;
  
  Agrid   = cube(L, m, G);
  Rgrid   = cube(m, m, G);
  ldRgrid = vec(G);
  lpgrid  = vec(G);
  
  std::vector<vec> mu_init(nblocks);
  std::vector<mat> S_init(nblocks);
  std::vector<uvec> blocks_init(nblocks);
  
  mu = mu_init;
  S  = S_init;
  blocks = blocks_init;
  
  blockSizes = ivec(bSize.begin(), bSize.size(), TRUE);
  cens = ivec(cen.begin(), cen.size(), TRUE);
  
  asig = hyp[0];
  bsig = hyp[1];    // hyperparms for sigma
  
  akap  = vec(nkap);
  bkap  = vec(nkap);
  lpkap = vec(nkap);
  
  // Initialize mixture components for prior on kappa_j
  for(reach = 2, i = 0; i < nkap; i++){
    akap[i]  = hyp[reach++];
    bkap[i]  = hyp[reach++];
    lpkap[i] = hyp[reach++];
  }
  
  // Initialize A_g & R_g matrices, ldRgrid & lpgrid    
  for(reach = 0, i = 0; i < G; i++){
    for(l = 0; l < L; l++) 
      for(k = 0; k < m; k++) 
        Agrid(l,k,i) = gridpars[reach++];
    
    for(k = 0; k < m; k++) 
      for(l = 0; l < m; l++) 
        Rgrid(l,k,i) = gridpars[reach++];
    
    ldRgrid[i] = gridpars[reach++];
    lpgrid[i] = gridpars[reach++];
  }
  
  // Initialize user provided block means, covariances
  for(b=0, mu_point=0, S_point=0, block_point=0; b < nblocks; b++){
    mu[b] = vec(muV.begin()+mu_point, blockSizes[b], TRUE);
    mu_point += blockSizes[b];
    
    S[b] = mat(SV.begin()+S_point, blockSizes[b], blockSizes[b]);
    S_point += pow(blockSizes[b],2);
    
    blocks[b] = arma::conv_to<uvec>::from(ivec(blocks_.begin()+block_point, blockSizes[b], TRUE));
    block_point += blockSizes[b];
  }  
  
  shrink = shrink_;
  
  return;
}

bool Stopping_Rule(double x0, double x1, double tolerance){
  double xm = 0.5 * fabs( x1 + x0 );
  
  if ( xm <= 1.0 ) return ( fabs( x1 - x0 ) < tolerance ) ? TRUE : FALSE;
  return ( fabs( x1 - x0 ) < tolerance * xm ) ? TRUE : FALSE;
}

void Adjust_Sigma(double (*f)(double), double &a, double &b, double tolerance){
  // Replaces combination of Max_Search_Golden_Section & lpFn2
  const double lam = 0.5 * (sqrt5 - 1.0);
  const double mu = 0.5 * (3.0 - sqrt5);     // = 1 - lam
  
  Rcout << "Siglim[0] = " << a << std::endl;
  Rcout << "Siglim[1] = " << b << std::endl;
  
  if (tolerance <= 0.0) tolerance = 1.0e-5 * (b - a);
  
  double x1  = b - lam * (b - a);
  double x2  = a + lam * (b - a);
  double fx1 = f(x1);
  double fx2 = f(x2);
  
  while ( ! Stopping_Rule( a, b, tolerance) ) {
    if (fx1 < fx2) {
      a = x1;
      //fa = fx1;
      if ( Stopping_Rule( a, b, tolerance) ) break;
      x1 = x2;
      fx1 = fx2;
      x2 = b - mu * (b - a);
      fx2 = f(x2);
    } else {
      b = x2;
      //fb = fx2;
      if ( Stopping_Rule( a, b, tolerance) ) break;
      x2 = x1;
      fx2 = fx1;
      x1 = a + mu * (b - a);
      fx1 = f(x1);
    }
  }
  Rcout << "a = " << a << "; b = " << b << std::endl;
  
  return;
}

double ppFn0(vec &par){
  // Calculate interpolated w0 function at all quantiles based on values of 
  // function at m knots
  int i;
  double zss, lps;
  
  //par[0:(m-1)] contains w0 values at knots
  vec wknot = par.subvec(0, m-1);
  
  for(i = 0; i < G; i++){
    wgrid.col(i) = Agrid.slice(i) * wknot; 
    zknot = arma::solve(arma::trimatu(Rgrid.slice(i)), wknot);
    // zss = w_*j^T * C_**^-1 * W_j*
    zss = arma::dot(zknot, zknot); 
    
    // Calculate p(W_{j*}|lambda_g)
    // llgrid = part of multivariate t-distribution
    // ldRgrid = log-determinant of C_** ^ -0.5
    // lpgrid = log prior of0.1 + 0.5 * (double)m
    llgrid[i] = -(0.1 + 0.5 * (double)m) * log1p(0.5 * zss / 0.1);
  }
  // pgvec contains p_g(W_j*)
  pgvec.col(0) = llgrid - ldRgrid + lpgrid;
  lps = logsum(pgvec.col(0));
  
  // Normalize probability    
  for(i = 0; i < G; i++) 
    pgvec.at(i,0) = exp(pgvec.at(i,0) - lps);
  
  w0 = wgrid * pgvec.col(0);
  
  return lps;
}	

double ppFn(vec &par, int p){
  int i, j;
  double akapm, zss, lps;
  
  //par[p*m + 1:m] contains wJ values at knots
  vec wknot = par.subvec(p * m, (p+1) * m - 1);
  
  for(i = 0; i < G; i++){
    wgrid.col(i) = Agrid.slice(i) * wknot; 
    zknot = arma::solve(arma::trimatu(Rgrid.slice(i)), wknot);
    zss = arma::dot(zknot, zknot);
    
    for(j = 0; j < nkap; j++){
      akapm = akap[j] + 0.5 * (double)m;
      lw[j] = -akapm * log1p(0.5 * zss/ bkap[j]) + lgamma(akapm) - lgamma(akap[j]) - 
        0.5 * (double)m * log(bkap[j]) + lpkap[j];
    }
    
    llgrid[i] = logsum(lw);
  }
  
  pgvec.col(p) = llgrid - ldRgrid + lpgrid;
  lps = logsum(pgvec.col(p));
  
  for(i = 0; i < G; i++) 
    pgvec.at(i,p) = exp(pgvec.at(i,p) - lps);
  
  //wMat.col(p) = wgrid * pgvec.col(p);
  wMat.row(p-1) = (wgrid * pgvec.col(p)).t();
  
  return lps;
}	

double logsum(vec L){
  NumericVector lx(L.begin(), L.end());
  double lxmax = *std::max_element(lx.begin(), lx.end());
  double a = std::accumulate(lx.begin(), lx.end(), 0.0, 
                             [lxmax](double x, double y){return x + exp(y-lxmax);});
  
  return lxmax + log(a);
}

double logmean(vec L){
  return logsum(L) - log(L.size());
}

double lpFn(vec par, double temp){
//double lpFn(vec &par, double temp){
  return logpostFn(par, temp, FALSE);
}

double lpFn1(vec par){
//  double lpFn1(vec &par){
  return logpostFn(par, 1.0, TRUE);
}

double lpFn2(double sigma){
  par0[(m+1)*(p+1)] = sigma;
  return logpostFn(par0, 1.0, FALSE);
}

double logpostFn(vec par, double temp, bool llonly){
//double logpostFn(vec &par, double temp, bool llonly){
  int i, j, l;
  const int par_position = (p+1)*m, gam_pos_offset = p - 1;
  double zeta0tot, lps0;
  double Q_L = -std::numeric_limits<double>::infinity();
  double Q_U =  std::numeric_limits<double>::infinity();
  double bdot_adj;
  double w0max;
  double alpha;
  double y_i;
  double q0_i;
  double lp;
  
  int lower_ind, upper_ind;
  double t_l, t_u, w_l, w_u;
  
  const double shrinkFactor = 1;
  
  gam0 = par[par_position];
  gam  = par.subvec(par_position+1, par_position + gam_pos_offset + 1);
  sigma = sigFn(par[par_position + gam_pos_offset + 2]);
  nu    = nuFn(par[par_position + gam_pos_offset + 3]);
  
  // Set vector w_j (j=0,...,p) to current interpolation of functions at specified knots
  lps0 = ppFn0(par);
  for(j = 1; j <= p; j++) lps0 += ppFn(par, j);
  
  if(temp > 0.0){
    // Initialize log-likelihood vector
    llvec.fill(-std::numeric_limits<double>::infinity());
    
    // Calculate zeta, zetadot
    w0max = w0.max(); // Armadillo max function
    
    zeta0dot = arma::exp(shrinkFactor * (w0-w0max));
    trape(zeta0dot.memptr() + 1, taugrid.memptr() + 1, L-1, zeta0.memptr() + 1);
    
    // zeta0[0] = 0, zeta0[1] = 1 by definition
    // rescale zeta0 so that it lies in [0,1] for tau != 0, 1
    zeta0tot = zeta0[L-2];
    
    zeta0[0] = 0.0; zeta0[L-1] = 1.0;
    zeta0.subvec(1,L-2) = taugrid[1] + (taugrid[L-2] - taugrid[1]) * zeta0.subvec(1,L-2) / zeta0tot;
    
    zeta0dot[0] = 0.0; zeta0dot[L-1] = 0.0;
    zeta0dot.subvec(1,L-2) = (taugrid[L-2] - taugrid[1]) * zeta0dot.subvec(1,L-2) / zeta0tot;
    
    // Approximate v = w(zeta(tau))
    // Linearly interpolate value of w function at tau's transformed by zeta
    /*
    for(j = 0; j < p; j++){
      for(l = 1; l < L-1; l++){
        //Rcout << "12a: "; arma::find(zeta0[l] >= taugrid, 1, "last").t().print();
        //Rcout << "12a: " << arma::as_scalar(arma::find(zeta0[l] >= taugrid, 1, "last")) << std::endl;
        lower_ind = arma::as_scalar(arma::find(zeta0[l] >= taugrid, 1, "last"));
        upper_ind = lower_ind + 1;
        //Rcout << "12b: " << upper_ind << std::endl;
        //Rcout << "12c: t_l = " << taugrid[lower_ind] << std::endl;
        t_l = arma::as_scalar(taugrid[lower_ind]);
        //Rcout << "12d: t_u = " << taugrid[upper_ind] << std::endl;
        t_u = arma::as_scalar(taugrid[upper_ind]);
        //Rcout << "12e: w_l = " << wMat.at(lower_ind, j) << std::endl;
        w_l = arma::as_scalar(wMat.at(lower_ind, j));
        //Rcout << "12f: w_u =" << wMat.at(upper_ind, j) << std::endl;
        w_u = arma::as_scalar(wMat.at(upper_ind, j));
        
        vMat.at(l,j) = w_l + (zeta0[l]-t_l)*(w_u - w_l)/(t_u - t_l);
      }
      vMat.at(0,j) = wMat.at(0,j);
      vMat.at(L-1,j) = wMat.at(L-1,j);
    }
    //vMat.print("vMat");
    */
    vMat = wMat;
    
    // Compute ||v_l||^2
    for(l = 0; l < L; l++) vNormSq[l] = arma::dot(vMat.col(l), vMat.col(l));
    
    if(vNormSq.min() > 0.0){
      a = x * vMat;  // a is n x L
      
      for(l = 0; l < L; l++){
        //aX[l] = ;
        bdot_adj = (-1 * a.col(l).min() / sqrt(vNormSq[l])) * sqrt(1.0 + vNormSq[l]);
        aTilde.col(l) = a.col(l) / bdot_adj;
        vTilde.col(l) = vMat.col(l) / bdot_adj;
      }
      
      // Compute modeled median (specifically the quantile of Y at tau_0 = F_0(0), 
      // but our prior guess for F is a t-distribution, so F_0(0) = 0.5) of Y | X 
      // for each observation i
      for(i = 0; i < n; i++)
        Q0vec[i] = gam0 + arma::dot(x.row(i), gam);
      
      //resLin = y - Q0vec;
      
      // % is element-wise multiplication for Armadillo vectors
      b0dot = sigma * q0(zeta0, nu) % zeta0dot;			
      
      //for(j = 0; j < p; j++) 
      //bdot.row(j) = b0dot.t() % vTilde.row(j);
      
      //b0dot.t().print("b0dot"); Rcout << std::endl;
      //bdot.print("bdot"); Rcout << std::endl;
      //Rcout << "b0dot" << std::endl; for(i=0; i < b0dot.n_elem; i++) Rcout << b0dot[i] << "\t"; Rcout << std::endl;
      //Rcout << "bdot" << std::endl; for(i=0; i < bdot.n_cols; i++) Rcout << bdot.at(0,i) << "\t"; Rcout << std::endl;
      
      //arma::vec b0dotrev = arma::flipud(b0dot);
      //arma::mat bdotrev  = arma::fliplr(bdot);
      
      //Rcout << "b0dot reverse" << std::endl; for(i=0; i < b0dotrev.n_elem; i++) Rcout << b0dotrev[i] << "\t"; Rcout << std::endl;
      //Rcout << "bdot reverse"  << std::endl; for(i=0; i < bdotrev.n_cols; i++) Rcout << bdotrev.at(0,i) << "\t"; Rcout << std::endl;
      
      //vTilde.print("vTilde"); Rcout << std::endl;
      
      // //Q0Pos.subvec(0, mid) = trape(b0dot.subvec(mid, L-1), taugrid.subvec(mid,L-1));
      // Q0Pos.subvec(0, mid) = trape(b0dot.subvec(mid, L-1), taugrid.subvec(mid,L-1));
      // Q0Pos[L-mid] = std::numeric_limits<double>::infinity();
      // 
      // //Q0Neg.subvec(0,mid) = trape(arma::flipud(b0dot.subvec(0, mid)), -1*arma::flipud(taugrid.subvec(0,mid)));
      // Q0Neg.subvec()
      // Q0Neg[mid+1] = std::numeric_limits<double>::infinity();
      // 
      // for(j = 0; j < p; j++){
      //   bPos.row(j) = trape(bdot.row(j).subvec(mid, L-1).t(), taugrid.subvec(mid, L-1)).t();
      //   bNeg.row(j) = trape(arma::fliplr(bdot.row(j).subvec(0,mid)).t(),
      //                    -1*arma::flipud(taugrid.subvec(0, mid))).t();
      // }
      
      //Q0Pos.t().print("Q0Pos");
      //Q0Neg.t().print("Q0Neg");
      //bPos.print("bPos");
      //bNeg.print("bNeg");
      //Rcout << std::endl;
      
      //sigmat1 = sigmat2 = sigma;
      
      for(i = 0; i < n; i++){
        //Rcout <<  i << " resLin[i]=" << std::setprecision(4) << resLin[i] << "\t";
        //if(resLin[i] == 0.0){
        y_i = y[i];
        q0_i = Q0vec[i];
        
        //Rcout << "a ";
        
        if(y_i == q0_i){
          // Y_i exactly equals modeled median, conditional on X_i
          llvec[i] = -log(b0dot[mid] + arma::dot(x.row(i), bdot.col(mid)));
        } 
        //else if(resLin[i] > 0.0){
        else if(y_i > q0_i){
          // Y_i > median
          Q_U = q0_i;
          l = mid;
          
          while(y_i > Q_U && l < L-1){
            l++;
            Q_L = Q_U;
            Q_U = Q_L + 0.5*(taugrid[l]-taugrid[l-1])*(b0dot[l]*(1+aTilde.at(i,l)) + b0dot[l-1]*(1+aTilde.at(i,l-1)));
            
            //QPosold = QPos;
            //QPos = Q0Pos[l] + arma::dot(x.row(i), bPos.col(l));
            //for(QPos = Q0Pos[l], j = 0; j < p; j++) QPos += x.at(i,j) * bPos.at(j,l);
            //Rcout << std::setprecision(4) << QPos << "\t";
          }
          if(l == L){
            Q_U = std::numeric_limits<double>::infinity();
          }
          
          // if(prior.cens[i]){
          //   params.llvec[i] = log(1.0 - dat.taugrid[d.mid + l]);
          // } else {
          //   if(l == d.L - d.mid - 1)
          //     params.llvec[i] = lf0tail(Q0tail(dat.taugrid[d.L-2]) + (params.resLin[i] - params.QPos)/params.sigma) - log(params.sigma);
          //   else
          //     params.llvec[i] = log(dat.taugrid[d.mid+l] - dat.taugrid[d.mid+l-1]) - log(params.QPos - params.QPosold);
          //   
          //   //Rcout << "ll==" << std::setprecision(4) << llvec[i] << std::endl;
          // }
          //Rcout << "y_" << i << "; l= " << l << "; Q_U= " << Q_U << std::endl;
        } 
        else {
          l = mid + 1;
          Q_L = q0_i;
          //QNegold = 0.0;
          //QNeg = Q0Neg[l] + arma::dot(x.row(i), bNeg.col(l));
          //for(QNeg = Q0Neg[l], j = 0; j < p; j++) QNeg += x.at(i,j) * bNeg.at(j,l);
          //Rcout << "QNeg" << "\t" << QNeg << "\t";
          
          //while(resLin[i] < -QNeg && l < mid){
          while(y_i < Q_L && l > 0){
            l--;
            Q_U = Q_L;
            Q_L = Q_U - 0.5*(taugrid[l]-taugrid[l-1])*(b0dot[l]*(1+aTilde.at(i,l)) + b0dot[l-1]*(1+aTilde.at(i,l-1)));
            //QNegold = QNeg;
            //QNeg = Q0Neg[l] + arma::dot(x.row(i), bNeg.col(l));
            //for(QNeg = Q0Neg[l], j = 0; j < p; j++) QNeg += x.at(i,j) * bNeg.at(j,l);
            //Rcout << std::setprecision(4) << QNeg << "\t";
          }
          if(l == 0){
            Q_L = -std::numeric_limits<double>::infinity();
          }
          
          // if(prior.cens[i]){
          //   params.llvec[i] = log(1.0 - dat.taugrid[d.mid - l]);
          // } else {
          //   if(l == d.mid)
          //     params.llvec[i] = lf0tail(Q0tail(dat.taugrid[1]) + (params.resLin[i] + params.QNeg)/params.sigma) - log(params.sigma);
          //   else
          //     params.llvec[i] = log(dat.taugrid[d.mid-l+1]-dat.taugrid[d.mid-l]) - log(params.QNeg - params.QNegold);
          // }
          //Rcout << "ll=" << std::setprecision(4) << llvec[i] << std::endl;
          //Rcout << "y_" << i << "; l= " << l << "; Q_L= " << Q_L << std::endl;
        }
        //Rcout << "b " << std::endl;
        if(Q_L == -std::numeric_limits<double>::infinity() || Q_U == std::numeric_limits<double>::infinity()){
          llvec[i] = -std::numeric_limits<double>::infinity();
        }
        else{
          alpha = (y_i - Q_L) / (Q_U - Q_L);
          llvec[i] = -1*log((1-alpha)*b0dot[l-1]*(1+aTilde.at(i,l-1)) + alpha*b0dot[l]*(1+aTilde.at(i,l)));
        }
        //if(llvec[i] == -std::numeric_limits<double>::infinity()) 
        //Rprintf("i = %d, ll[i] = %g, resLin[i] = %g, l = %d\n", i, llvec[i], resLin[i], l);
        
      }
      //Rcout << "sigma = " << sigma << std::endl;
    }
  } else{
    for(i = 0; i < n; i++) llvec[i] = 0.0;
  }
  
  lp = temp * arma::accu(llvec);
  
  lp = 0;
  for(i = 0; i < n; i++){
    lp += llvec[i];
  }
  
  if(std::isnan(lp)) lp = -std::numeric_limits<double>::infinity();
  
  if(!llonly){
    lp += lps0 + R::dlogis(nu, 0.0, 1.0, 1);
    if(shrink) lp += dbasejoint(gam);
  }

  return lp;
}

void trape(double *x, double *h, int length, double *integral){
  // Calculates integral of function x over domain h via trapezoidal rule
  integral[0] = 0.0;
  
  for(int i = 1; i < length; i++){
    integral[i] = integral[i-1] + 0.5 * (h[i]-h[i-1]) * (x[i]+x[i-1]);
  }
  return;
}

double dbase_joint_scl(double b, vec &gam){
  double a = 0.0;
  int j;
  for(j = 0; j < p; j++){
    a += R::dt(gam[j] / b, 1.0, 1) - log(b);
  }
  return a;
}

double dbasejoint(vec &gam){
  int i;
  double pp = 0.525;
  for(i = 0; i < 10; i++){
    lb[i] = dbase_joint_scl(R::qt(pp, 1.0, 1, 0), gam);
    pp += 0.05;
  }
  return logmean(lb);
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

vec unitFn(vec u) {
  vec z(u);
  
  z.transform([](double u){
    if(u < 1.0e-15)
      u = 1.0e-15; 
    else if(u > 1.0 - 1.0e-15)
      u = 1.0 - 1.0e-15;
    return u;});
  
  return z;
}

vec Q0(vec u, double nu) {
  vec u_cap = unitFn(u);
  NumericVector Q = qt(NumericVector(u_cap.begin(), u_cap.end()), nu);
  
  return vec(Q.begin(), Q.size(), FALSE);
}

vec q0(vec u, double nu) {
  vec u_cap(unitFn(u));
  NumericVector q = 1 / dt(qt(NumericVector(u_cap.begin(), u_cap.end()), nu), nu);
  
  return vec(q.begin(), q.size(), FALSE);
}

vec F0(vec x, double nu) {
  return pt(NumericVector(x.begin(), x.end()), nu);  
}

vec q0tail(vec u) {
  return 1.0/dt(qt(NumericVector(u.begin(), u.end()), 0.1), 0.1);
}

double Q0tail(double u) {
  return R::qt(u, 0.1, 1, 0);	
}

double lf0tail(double x){
  return R::dt(x, 0.1, 1);
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
  ivec chunk_size(nblocks, arma::fill::zeros);
  
  vec blocks_d(npar);
  vec acpt_chunk(nblocks, arma::fill::zeros);
  rowvec alpha(nblocks);
  vec frac(nblocks);
  vec par_incr(npar);
  vec alpha_run(nblocks, arma::fill::zeros);
  vec zsamp(npar);
  vec parOld(npar);
  vec logUniform(nblocks*niter);
  vec gammaAdj(nblocks*niter);
  
  Rcout << "1" << std::endl;
  
  // Ragged arrays
  std::vector<ivec> blocks_pivot(nblocks);
  std::vector<vec> parbar_chunk(nblocks);
  std::vector<mat> R(nblocks); // Chol factor of MCMC block proposal covariances
  
  // Initialize variables
  for(b=0; b < nblocks; b++){
    currBlockSize = blockSizes[b];
    R[b] = mat(blockSizes[b], currBlockSize);
    blocks_pivot[b] = ivec(currBlockSize);
    parbar_chunk[b] = vec(currBlockSize, arma::fill::zeros);
    
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
  lpval = logpostFn(parSample, temp, FALSE);
  if(verbose) Rcout << "Initial lp = " << lpval << std::endl;
  
  Rcout << "3" << std::endl;

  // Adaptive Metropolis MCMC
  for(iter = 0; iter < niter; iter++){
    
    Rcout << "Iteration: " << iter << ". lp = " << lpval << std::endl;
    lm.t().print("lm");
    
    for(b = 0; b < nblocks; b++){
      // Sample new parameters for variables in block b
      
      currBlockSize = blockSizes[b];
      chunk_size[b]++;
      zsamp.subvec(0, currBlockSize-1) = as<arma::vec>(rnorm(currBlockSize)); // Use Rcpp sugar rnorm
      par_incr.subvec(0,currBlockSize-1) = trimatu(R[b]) * zsamp.subvec(0, currBlockSize-1);
      //lambda = lm[b] * sqrt(3.0 / R::rgamma(3.0, 1.0));
      lambda = lm[b] * sqrt(3.0 / gammaAdj[g++]);
      parOld = parSample;
      
      Rcout << "Block " << b << "; " << "lambda = " << lambda << std::endl;
      //par_incr.t().print("par_incr");
      
      parSample.elem(blocks[b]) += lambda * par_incr.subvec(0, currBlockSize-1);
      parSample.t().print("parSample");
      
      // Evaluate loglikelihood at proposed parameters
      lpvalnew = logpostFn(parSample, temp, FALSE);
      /*
      lp_diff = lpvalnew - lpval;
      
      alpha[b] = exp(lp_diff);
      if(alpha[b] > 1.0) alpha[b] = 1.0;
      
      // Check for acceptance
      if(logUniform[u++] < lp_diff){
        // Accept
        Rcout << "Block " << b << " Accept! lpdiff = " << lp_diff << " accept = " << logUniform[u-1] << std::endl;
        Rcout << "lpval = " << lpval << "; lpvalnew = " << lpvalnew << std::endl;
        //lpval = lpvalnew;
        //Rcout << "Accept! lpvalnew = " << lpvalnew << std::endl;
      }
      else{
        // Reject
        Rcout << "Block " << b << " Reject! lpdiff = " << lp_diff << " accept = " << logUniform[u-1] << std::endl;
        Rcout << "lpval = " << lpval << "; lpvalnew = " << lpvalnew << std::endl;
        parSample = parOld;
      }
      parSample.t().print("parSample");
      
      alpha_run[b] = ((double)run_counter[b] * alpha_run[b] + alpha[b]) / ((double)run_counter[b] + 1.0);
      run_counter[b]++;
      
      */
    }
    /*
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
      
      acpt_chunk[b] = acpt_chunk[b] + (alpha[b] - acpt_chunk[b]) / chs;
      parbar_chunk[b] = parbar_chunk[b] + (parSample.elem(blocks[b]) - parbar_chunk[b]) / chs;
      
      if(chunk_size[b] == refresh * blockSizes[b]){
        Rcout << "Adjusting covariance matrices!" << std::endl;
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
        //if(iter >= 45){S[b].print("S_" + std::to_string(b));}
        
        R[b]  = arma::chol(S[b], "upper");
        mu[b] = mu[b] + frac[b] * (parbar_chunk[b] - mu[b]);
        lm[b] = lm[b] * exp(frac[b] * (acpt_chunk[b] - acpt_target[b]));
        
        acpt_chunk[b] = 0;
        parbar_chunk[b].zeros();
        chunk_size[b] = 0;
      }
    }
    */
  }
}


/*
 // [[Rcpp::export]]
 NumericVector ppFn0_Cpp(NumericVector wknot_,
 NumericVector gridpars,
 int L, 
 int nknots,
 int G){
 
 int i;
 double zss, lps;
 
 arma::mat wgrid(L, G);
 
 arma::colvec zknot;
 arma::colvec wknot(wknot_.begin(), wknot_.size(), FALSE);
 
 arma::Cube<double> Agrid(L, nknots, G);
 arma::Cube<double> Rgrid(nknots, nknots, G);
 
 arma::colvec pgvec(G);
 arma::colvec llgrid(G);
 arma::colvec ldRgrid(G);
 arma::colvec lpgrid(G);
 
 int reach = 0;
 
 for(int i = 0; i < G; i++){
 for(int l = 0; l < L; l++) 
 for(int k = 0; k < nknots; k++) 
 Agrid(l,k,i) = gridpars[reach++];
 
 for(int k = 0; k < nknots; k++) 
 for(int l = 0; l < nknots; l++) 
 Rgrid(l,k,i) = gridpars[reach++];
 
 ldRgrid[i] = gridpars[reach++];
 lpgrid[i] = gridpars[reach++];
 }
 
 //par[1:m] contains w0 values at knots
 //arma::Col<double> wknot = par.subvec(0,m-1);
 
 for(i = 0; i < G; i++){
 wgrid.col(i) = Agrid.slice(i) * wknot; 
 
 //Rgrid.slice(i).print();
 zknot = arma::solve(arma::trimatu(Rgrid.slice(i)), wknot);
 // zss = w_*j^T * C_**^-1 * W_j*
 zss = arma::as_scalar(zknot.t() * zknot); 
 
 // Calculate p(W_{j*}|lambda_g)
 // llgrid = part of multivariate t-distribution
 // ldRgrid = log-determinant of C_** ^ -0.5
 // lpgrid = log prior of0.1 + 0.5 * (double)m
 
 llgrid[i] = -(1.5 + 0.5 * (double)nknots) * log1p(0.5 * zss / 1.5);
 }
 pgvec = llgrid - ldRgrid + lpgrid;
 lps = logsum_Cpp(wrap(pgvec));
 
 for(i = 0; i < G; i++) 
 pgvec.at(i) = exp(pgvec.at(i) - lps);
 
 arma::colvec w0 = wgrid * pgvec;
 
 return wrap(w0);
 }	
 */
/*
 // [[Rcpp::export]]
 void qrJointClassTest(NumericVector par_,
 NumericMatrix x_,
 NumericVector y_,
 IntegerVector cens_,
 int shrink_,
 NumericVector hyper_,
 IntegerVector dimpars_,
 NumericMatrix gridmats_,
 NumericVector tauG_,
 NumericVector siglim_,
 NumericVector muV_,
 NumericVector SV_,
 IntegerVector blocks_,
 IntegerVector blockSizes_,
 NumericVector dmcmcpar,
 IntegerVector imcmcpar){
 qrJoint Q(par_, x_, y_, cens_, shrink_, hyper_, dimpars_, gridmats_, tauG_, siglim_, 
 muV_, SV_, blocks_, blockSizes_, dmcmcpar, imcmcpar);
 
 Rcout << "Constructor finished!" << std::endl;
 
 Q.adMCMC();
 
 //return wrap(Q.parStore.t());
 return;
 }	
 
 // [[Rcpp::export]]
 void qrLikelihoodTest(NumericVector par_,
 NumericMatrix x_,
 NumericVector y_,
 IntegerVector cens_,
 int shrink_,
 NumericVector hyper_,
 IntegerVector dimpars_,
 NumericMatrix gridmats_,
 NumericVector tauG_,
 NumericVector siglim_,
 NumericVector muV_,
 NumericVector SV_,
 IntegerVector blocks_,
 IntegerVector blockSizes_,
 NumericVector dmcmcpar,
 IntegerVector imcmcpar){
 
 qrJoint Q(par_, x_, y_, cens_, shrink_, hyper_, dimpars_, gridmats_, tauG_, siglim_, 
 muV_, SV_, blocks_, blockSizes_, dmcmcpar, imcmcpar);
 
 Rcout << "Constructor finished!" << std::endl;
 
 // Tau grid
 Q.taugrid.t().print("Grid of tau's"); Rcout << std::endl;
 
 // GP interpolation
 //Q.ppFn0();
 Q.parSample.subvec(0, Q.m - 1).t().print("W0 at knots"); Rcout << std::endl;
 Q.w0.t().print("W0 approximation"); Rcout << std::endl;
 Rcout << std::endl;
 
 //Q.ppFn(1);
 Q.parSample.subvec(Q.m, 2*Q.m - 1).t().print("W at knots"); Rcout << std::endl;
 Q.wMat.print("W approximation"); Rcout << std::endl;
 Rcout << std::endl;
 
 // Calculate zeta, zeta.dot
 double w0max = Q.w0.max(); // Armadillo max function
 arma::vec zeta0dot = arma::exp(Q.shrinkFactor * (Q.w0-w0max));
 arma::vec zeta0    = Q.trape(zeta0dot, Q.taugrid);
 
 // zeta0[0] = 0, zeta0[1] = 1 by definition
 // rescale zeta0 so that it lies in [0,1] for tau != 0, 1
 double L = Q.L;
 double zeta0tot = zeta0[L-2];
 
 zeta0[0] = 0.0; zeta0[L-1] = 1.0;
 zeta0.subvec(1,L-2) = Q.taugrid[1] + (Q.taugrid[L-2] - Q.taugrid[1]) * zeta0.subvec(1,L-2) / zeta0tot;
 
 zeta0dot[0] = 0.0; zeta0dot[L-1] = 0.0;
 zeta0dot.subvec(1,L-2) = (Q.taugrid[L-2] - Q.taugrid[1]) * zeta0dot.subvec(1,L-2) / zeta0tot;
 
 zeta0.t().print("Zeta"); Rcout << std::endl;
 zeta0dot.t().print("Zeta dot"); Rcout << std::endl;
 Rcout << std::endl;
 
 // Output v matrix = w(zeta(tau))
 Rcout << "vMat is " << Q.vMat.n_rows << " by " << Q.vMat.n_cols << std::endl << std::endl;
 Q.vMat.print("vMat"); Rcout << std::endl;
 Q.vNormSq.t().print("vNormSq"); Rcout << std::endl;
 
 return;
 }	
 */
