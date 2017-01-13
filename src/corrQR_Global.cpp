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
void   adMCMC(void);
double dbase_joint_scl(double, vec &);
double dbasejoint(vec &);
void   trape(double *x, double *h, int length, double *integral);
double lpFn(vec, double temp);
double lpFn1(vec);
double lpFn2(double sigma);
double logpostFn(vec, double temp, bool llonly);

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
                      List A, List R, List muV, List SV,
                      IntegerVector blocks_, IntegerVector bSize);

/**** Global variables ****/
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
int npar;     // # parameters to track
int nblocks;  // # of MH blocks
bool verbose; // flag to print intermediate calc's
int ticker;   // how often to update screen
double temp;

// Adaptive MCMC
int refresh;  // adaption rate of adaptive MH
double decay;
ivec refresh_counter; //
vec acpt_target;      //
vec lm;               //

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

// Hyperparameters
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
  List A               = as<List>(A_);
  List R               = as<List>(R_);
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

  // MCMC parameters
  niter   = DIM[8];
  thin    = DIM[9];
  nsamp   = DIM[10];
  npar    = (m+1) * (p+1) + 2;
  nblocks = IMCMCPAR[0];
  refresh = IMCMCPAR[1];
  verbose = (bool) IMCMCPAR[2];
  ticker  = IMCMCPAR[3];
  temp    = DMCMCPAR[0];
  decay   = DMCMCPAR[1];

  refresh_counter = ivec(IMCMCPAR.begin() + 4, nblocks, TRUE);
  acpt_target     =  vec(DMCMCPAR.begin() + 2, nblocks, TRUE);
  lm              =  vec(DMCMCPAR.begin() + 2 + nblocks, nblocks, TRUE);

  // Prior parameters
  Init_Prior_Param(L, m, G, nblocks, nkap, HYP, A, R, MU_V, SV, BLOCKS, BLOCKS_SIZE);

  // Parameters
  gam0     = 0;
  sigma    = 1;
  nu       = 1;
  gam      = vec(p, fill::zeros);
  w0       = vec(L, fill::zeros);
  zeta0    = vec(L, fill::zeros);
  zeta0dot = vec(L, fill::zeros);
  wMat     = mat(p, L, fill::zeros);
  vMat     = mat(p, L, fill::zeros);

  // Intermediate calculations
  wgrid    = mat(L, G, fill::zeros);
  a        = mat(n, L, fill::zeros);
  aTilde   = mat(n, L, fill::zeros);
  vTilde   = mat(p, L, fill::zeros);
  b0dot    = vec(L, fill::zeros);
  bdot     = mat(p, L, fill::zeros);
  bPos     = mat(p, mid+1, fill::zeros);
  bNeg     = mat(p, mid+1, fill::zeros);
  vNormSq  = vec(L, fill::zeros);
  zknot    = vec(m, fill::zeros);
  Q0vec    = vec(n, fill::zeros);
  resLin   = vec(n,fill::zeros);
  llvec    = vec(n, fill::zeros);
  llgrid   = vec(G, fill::zeros);

  pgvec    = mat(G, p+1, fill::zeros);
  lb       = vec(10, fill::zeros);
  lw       = vec(nkap, fill::zeros);

  parSample = vec(PAR.begin(), PAR.size(), TRUE);

  // Output
  lpSample     = vec(nsamp, fill::zeros);           // stored log-likelihood
  acceptSample = mat(nsamp, nblocks, fill::zeros);  // stored MH acceptance history
  parStore     = mat(npar, nsamp, fill::zeros);     // stored posterior draws npar x nsamp

  //adMCMC();

  S[0].print();
  Agrid.slice(0).print("A");
  Rgrid.slice(0).print("R");

  return Rcpp::List::create(Rcpp::Named("X") = x,
                            Rcpp::Named("Y") = y,
                            Rcpp::Named("n") = n,
                            Rcpp::Named("PAR") = PAR);
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
  ldRgrid = vec(G);
  lpgrid  = vec(G);

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
/*
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
  }
}
*/
