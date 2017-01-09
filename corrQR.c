#include "R.h"
#include "Rmath.h"
#include "R_ext/Applic.h"

#define sqrt5 2.236067977499789696

#define bool int
#define true 1
#define false 0

/** Function headers ================================================= */
// Memory allocation & printing for vectors & matrices
double *vect(int n);
int *ivect(int n);
double **Matrix(int nr, int nc);
void Rprintvec(char *a, char *format, double *x, int n);
void Rprintmat(char *a, char *format, double **x, int m, int n, int flip);
void Rprintveci(char *a, char *format, int *x, int n);

// Linear algebra functions
void mmprod(double **a, double **b, double **c,
            int m, int k, int n,
            int atrans, int btrans, int ctrans);
void mvprod(double **a, double *b, double *c,
            int m, int k,
            int atrans);
void chol(double **R, int N, double tol, int *pivot, int *rank, int max_rank,
          double *d, double **A, int dopivoting, int padzero, double eps);
double inprod(double *x, double *y, int n);
double sumsquares(double *x, int n);
double sum(double *x, int n);
void set_lower_tri_zero(double **A, int n, int m);
double vmax(double *x, int n);
double vmin(double *x, int n);
void trisolve(double **R, int m, double *b, double *x, int transpose);
void triprod(double **R, int m, int n, double *x, double *b, int transpose);
double logsum(double *lx, int n);
double logmean(double *lx, int n);

// Helper functions
void trape(double *x, double *h, int n, double *c, int reverse);
double shrinkFn(double x);
double unitFn(double u);

// Transformation functions
double sigFn(double z);
double sigFn_inv(double s);
double nuFn(double z);
double nuFn_inv(double nu);

// Quantile functions
double q0(double u, double nu);
double Q0(double u, double nu);
double F0(double x, double nu);
double q0tail(double u);
double Q0tail(double u);
double lf0tail(double x);

static int Stopping_Rule(double x0, double x1, double tolerance);
void Max_Search_Golden_Section( double (*f)(double), double* a, double *fa, double* b, double* fb, double tolerance);
double ppFn0(double *wknot, double *w, double *postgrid);
double ppFn(double *wknot, double *w, double *postgrid);
double logpostFn(double *par, double temp, bool llonly, double *ll, double **pg);
double lpFn(double *par, double temp);
double lpFn2(double sigma);


void adMCMC(int niter, int thin, int npar,
            double *par, double **mu, double ***S, double *acpt_target,
            double decay, int refresh, double *lm, double temp,
            int nblocks, int **blocks, int *blocks_size,
            int *refresh_counter, int verbose, int ticker,
            double lpFn(double *, double), double *parsamp,
            double *acptsamp, double *lpsamp);

/** Global variable definitions ===================================== */
// Dimension parameters
int n;    // # obs
int p;    // # predictors
int q;    // # response variables
int L;    // # tau's (quantiles) at which betas are estimated
int mid;  // location in taugrid of median (0.50)
int m;    // dim of low rank approx to w functions
int G;    // dim G of approx to pi(lambda)
int nkap; // # of mixture components for prior on kappa_j

// Copula method
int copula;

// Data
double **x;       // covariates (n x p)
double **y;       // response (n x q)
double *taugrid;  // quantiles of interest (L x 1)

// MCMC parameters
int niter;     // MCMC iterations
int thin;      // thinning factor
int nsamp;     // # of MCMC samples to keep
int npar;      // # parameters to track
int nblocks;   // # of MH blocks
int refresh;   // adaption rate of adaptive MH
bool verbose; // flag to print intermediate calc's
int ticker;    // how often to update screen
double temp;
double decay;
int *refresh_counter; //
double *acpt_target;  //
double *lm;           //

// Prior parameters
double ***Agrid;   // storage for A_g = C_0*(g) C**^{-1}(g) (L x m x G)
double ***Rgrid;   // storage for R_g = chol(C**(g)) (m x m x G)
double   *ldRgrid; // log trace(R_g), normalizing constant
double   *lpgrid;  // log prior (lambda_j)

double **mu;  // initial MH block means
double ***S;  // initial MH block covariances
int **blocks; // each row b contains variable positions in parameter vector
// that are to be updated in block b of MCMC
int *blocks_size;  // vector of # of parameters in each MCMC block
int *cens;    // vector of indicators that obs i is censored
bool shrink;
double shrinkFactor;

// Hyperparameters
double asig, bsig;            // hyperparms for sigma
double *akap, *bkap, *lpkap;  // hyperparms for kappa

// Model parameters
double gam0;  // curr value of gamma_0
double sigma; // curr value of sigma
double nu;    // curr value of nu
double theta; // curr value of copula correlation coefficient

double *zeta0;     // each col has curr values of zeta(tau_l) (L x 1)
double *zeta0dot;  // each col has curr values of zeta.dot(tau_l) (L x 1)
double *gam;       // curr value of vector gamma for response j (p x 1)
double **w0;       // each col has curr values of w_0(tau_l) (L x q)
double ***wMat;    // each slice corresponds to current values of W_j (q x p x L)
double ***vMat;    // each elice corresponds to W_j(zeta(tau)) (q x p x L)

// Intermediate calculations
double   *b0dot;    // (L x 1)
double  **wgrid;    // temp to store A_g * low rank w knots for each g
double  **a;        // = (x*vMat) (n x L)
double   *aX;       // = (L x 1)
double  **vTilde;   // = (p x L)
double  **bdot;     // = x^T %*% beta.dot(tau) (p x L)
double  **bPos;     // (mid+1 x L)
double  **bNeg;     // (mid+1 X L)
double   *vNormSq;
double   *zknot;    // (m x 1)
double   *Q0Pos;
double   *Q0Neg;
double   *xLin;
double   *resLin;   // (n x 1)

double  **tau_x;   // [i,j] contains quantile of y_{i,j} given x_i and current parameters

double   *llvec;   // log-likelihood by obs (n x 1)
double   *llgrid;  // MV t-dist contrib to loglikelihood (G x 1)

double  **pgvec;   // each column contains p_g(W_j*) (G x (p+1)) in vectorized form for each response
double  *lb;       // used for shrinkage prior on gamma vector (10 x 1)
double  *lw;       // used for interpolation of marginal GP (nkap x 1)

double *par0;      // used in initial adjustment of sigma
double *parSample; // current parameter draw

int sig_pos;

/** Memory allocation & print functions for vectors / matrices === */
double * vect(int n){
  // Allocate vector of doubles of size n
  return (double *)R_alloc(n, sizeof(double));
}

int * ivect(int n){
  // Allocate vector of integers of size n
  return (int *)R_alloc(n, sizeof(int));
}

double ** Matrix(int nr, int nc){
  // Allocate matrix of doubles with nr rows and nc columns
  int i;
  double **m;
  double *p;

  m = (double **) R_alloc(nr, sizeof(double *));
  p = (double *)  R_alloc(nr*nc, sizeof(double));

  for (i = 0; i < nr; i++)
    m[i] = &p[i*nc];
  return m;
}

void Rprintvec(char *a, char *format, double *x, int n){
  // Print title in string a followed by elements of vector x (of doubles)
  int i;
  Rprintf("%s", a);
  for(i = 0; i < n; i++){
    Rprintf(format, x[i]);
    Rprintf(" ");
  }
  Rprintf("\n");
}

void Rprintveci(char *a, char *format, int *x, int n){
  // Print title in string a followed by elements of vector x (of integers)
  int i;
  Rprintf("%s", a);
  for(i = 0; i < n; i++){
    Rprintf(format, x[i]);
    Rprintf(" ");
  }
  Rprintf("\n");
}

void Rprintmat(char *a, char *format, double **x, int m, int n, int flip){
  // Print title in string a followed by elements of matrix x (of doubles)
  // Note: input flip currently not used
  int i, j;
  Rprintf("%s\n", a);
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      Rprintf(format, x[i][j]);
      Rprintf(" ");
    }
    Rprintf("\n");
  }
}

/** Linear algebra functions ============================= */
void mmprod(double **a, double **b, double **c,
            int m, int k, int n,
            bool atrans, bool btrans, bool ctrans){
  // Matrix multiplication of a (dim m x k) x b (dim k x n) and stores
  // results in c
  //  - flags atrans, btrans, ctrans indicates matrices are transposed
  //  - i.e. atrans = 1, btrans = 1, ctrans = 0 ==> c = a^T * b^T

  // Note: no size checking

  int i, j, l;
  if(!ctrans){
    if(atrans && btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[i][j] += a[l][i] * b[j][l];
    } else if (!atrans && btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[i][j] += a[i][l] * b[j][l];
    } else if (atrans && !btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[i][j] += a[l][i] * b[l][j];
    } else {
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[i][j] += a[i][l] * b[l][j];
    }
  } else {
    if(atrans && btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[j][i] += a[l][i] * b[j][l];
    } else if (!atrans && btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[j][i] += a[i][l] * b[j][l];
    } else if (atrans && !btrans){
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[j][i] += a[l][i] * b[l][j];
    } else {
      for(i = 0; i < m; i++)
        for(j = 0; j < n; j++)
          for(c[i][j] = 0.0, l = 0; l < k; l++) c[j][i] += a[i][l] * b[l][j];
    }
  }
}

void mvprod(double **a, double *b, double *c, int m, int k, bool atrans){
  // Matrix / vector multiplication of a (dim m x k) x b (dim k x 1) and
  // stores results in c
  //  - flags atrans indicates matrix a is transposed
  //  - i.e. atrans = 1  ==> c = a^T * b

  // Note: no size checking
  int i, l;
  if(atrans){
    for(i = 0; i < m; i++)
      for(c[i] = 0.0, l = 0; l < k; l++) c[i] += a[l][i] * b[l];
  } else {
    for(i = 0; i < m; i++)
      for(c[i] = 0.0, l = 0; l < k; l++) c[i] += a[i][l] * b[l];
  }
}

void chol(double **R, int N, double tol, int *pivot, int *rank, int max_rank,
          double *d, double **A, int dopivoting, int padzero, double eps){
  // Returns Cholesky decomposition of matrix A and stores results in R
  //  - N is dimension of A

  set_lower_tri_zero(R, N, max_rank);

  int i, a, l;
  double u, b;

  for(i = 0; i < N; i++){
    pivot[i] = i;
    d[i] = A[i][i] + eps * (1.0 + A[i][i]);
  }

  int k = 0, max_diag;
  for(max_diag = k, i = k + 1; i < N; i++)
    if(d[i] > d[max_diag])
      max_diag = i;
    int flag = (d[max_diag] > tol);


    while(flag){
      if(dopivoting){
        if(max_diag > k){
          a = pivot[k];
          pivot[k] = pivot[max_diag];
          pivot[max_diag] = a;

          b = d[k];
          d[k] = d[max_diag];
          d[max_diag] = b;

          for(i = 0; i < k; i++){
            b = R[i][k];
            R[i][k] = R[i][max_diag];
            R[i][max_diag] = b;
          }
        }
      }

      R[k][k] = sqrt(d[k]);

      for(i = k + 1; i < N; i++){
        u = A[pivot[i]][pivot[k]];
        for(R[k][i] = u, l = 0; l < k; l++)
          R[k][i] -= R[l][i] * R[l][k];
        R[k][i] /= R[k][k];
        d[i] -= R[k][i] * R[k][i];
      }

      k++;
      flag = (k < max_rank);
      if(flag){
        for(max_diag = k, i = k + 1; i < N; i++)
          if(d[i] > d[max_diag])
            max_diag = i;
          flag = (d[max_diag] > tol);
      }
    }

    rank[0] = k;
    if(padzero){
      for(l = k; l < N; l++)
        d[l] = 0.0;
    }
}

void set_lower_tri_zero(double **A, int n, int m ){
  // Set all elements below the diagonal of a to zero
  int i, j;
  for(i = 0; i < n; i++)
    for(j = i + 1; j < m; j++)
      A[j][i] = 0.0;
}

double inprod(double *x, double *y, int n){
  // Returns inner product of vectors x & y, each of size n
  double ip = 0.0;
  int i;
  for(i = 0; i < n; i++)
    ip += x[i] * y[i];
  return ip;
}

double sumsquares(double *x, int n){
  // Returns sum of squared elements in vector x of doubles
  double ss = 0.0;
  int i;
  for(i = 0; i < n; i++)
    ss += x[i] * x[i];
  return ss;
}

double sum(double *x, int n){
  // Returns sum of elements in vector x of length n
  double a = 0.0;
  int i;
  for(i = 0; i < n; i++) a += x[i];
  return a;
}

double vmax(double *x, int n){
  // Returns max element of double vector x of size n
  int i;
  double xmax = x[0];
  for(i = 1; i < n; i++) if(x[i] > xmax) xmax = x[i];
  return xmax;
}

double vmin(double *x, int n){
  // Returns min element of double vector x of size n
  int i;
  double xmin = x[0];
  for(i = 1; i < n; i++) if(x[i] < xmin) xmin = x[i];
  return xmin;
}

void trisolve(double **R, int m, double *b, double *x, int transpose){
  // Solves the system of equations Rx = b, where:
  //  - if transpose = true, R is lower triangular
  //  - if transpose = false, R is upper triangular
  int i, j;
  if(transpose){
    for(j = 0; j < m; j++){
      for(x[j] = b[j], i = 0; i < j; i++)
        x[j] -= x[i] * R[i][j];
      x[j] /= R[j][j];
    }
  } else {
    for(j = m - 1; j >= 0; j--){
      for(x[j] = b[j], i = j + 1; i < m; i++)
        x[j] -= R[j][i] * x[i];
      x[j] /= R[j][j];
    }
  }
}

void triprod(double **R, int m, int n, double *x, double *b, int transpose){
  // Calculates the product b = Rx, where R is a m x n matrix.
  // If tranpose = false, assumes R is upper triangular
  //  - Otherwise assume R is lower triangular
  int i, j;
  if(transpose){
    for(i = 0; i < m; i++)
      for(b[i] = 0.0, j = 0; j <= i; j++)
        b[i] += R[j][i] * x[j];
    for(; i < n; i++)
      for(b[i] = 0.0, j = 0; j < m; j++)
        b[i] += R[j][i] * x[j];
  } else{
    for(i = 0; i < m; i++)
      for(b[i] = 0.0, j = i; j < n; j++)
        b[i] += R[i][j] * x[j];
  }
}

double logsum(double *lx, int n){
  // For vector lx, returns log(exp(lx[1]) + ... exp(lx[n]))
  // Note: function is called on a vector of logged values lx, so
  //   this function returns log of the sum of unlogged values
  double lxmax = vmax(lx, n);
  double a = 0.0;

  for(int i = 0; i < n; i++) a += exp(lx[i] - lxmax);

  return lxmax + log(a);
}

double logmean(double *lx, int n){
  // lx is a vector of logged values
  // This function returns log of mean of unlogged values
  return logsum(lx, n) - log((double)n);
}


/** Helper functions ===================================== */
void trape(double *x, double *h, int n, double *c, int reverse){
  // Input:
  //  x = value of function to integrate at points h
  //  n = length of x & h
  //  c = stores value of integral at each location in h

  // Note: Pointers passed to function may correspond to a position
  //  in the vector other than the beginning, so reverse calculates
  //  the integral at the starting position going backwards.
  int i, j = 0;
  c[0] = 0.0;
  if(reverse){
    for(i = 1; i < n; i++){
      c[i] = c[i-1] + 0.5 * (h[j] - h[j-1]) * (x[j] + x[j-1]);
      j--;
    }
  } else {
    for(i = 1; i < n; i++){
      c[i] = c[i-1] + 0.5 * (h[j+1] - h[j]) * (x[j] + x[j+1]);
      j++;
    }
  }
}

double shrinkFn(double x){
  // Currently this function does nothing.
  // Included for possible changes at this location only.
  return 1.0;
}

double unitFn(double u){
  // Caps values close to zero or one
  if(u < 1.0e-15)
    u = 1.0e-15;
  else if(u > 1.0 - 1.0e-15)
    u = 1.0 - 1.0e-15;
  return u;
}


/** Transformation functions ==================================== */
// Priors are placed on transformations of sigma and nu, so these
// functions transform the values of sigma and nu to the values
// on which the prior is placed.

double sigFn(double z){ return exp(z/2.0); }
double sigFn_inv(double s){ return 2.0*log(s); }

double nuFn(double z){ return 0.5 + 5.5*exp(z/2.0); }
double nuFn_inv(double nu){ return 2.0*log((nu - 0.5)/5.5); }


/** Quantile functions ========================================== */

double q0(double u, double nu) {
  // Returns quantile density of prior guess for dist of Y | X = 0, nu
  return 1.0 / (dt(qt(unitFn(u), nu, 1, 0), nu, 0) * qt(0.9, nu, 1, 0));
}

double Q0(double u, double nu) {
  // Returns quantile of prior guess for dist of Y | X = 0, nu
  return qt(unitFn(u), nu, 1, 0) / qt(0.9, nu, 1, 0);
}

double F0(double x, double nu) {
  // Returns cdf of prior guess for distribution of Y | X = 0, nu
  return pt(x * qt(0.9, nu, 1, 0), nu, 1, 0);
}

double q0tail(double u) {
  return 1.0/dt(qt(u, 0.1, 1, 0), 0.1, 0);
}
double Q0tail(double u) {
  return qt(u, 0.1, 1, 0);
}
double lf0tail(double x){
  return dt(x, 0.1, 1);
}
double dbase_joint_scl(double b, double *gam){
  double a = 0.0;
  int j;
  for(j = 0; j < p; j++) a += dt(gam[j] / b, 1.0, 1) - log(b);
  return a;
}
double dbasejoint(double *gam){
  int i;
  double pp = 0.525;
  for(i = 0; i < 10; i++){
    lb[i] = dbase_joint_scl(qt(pp, 1.0, 1, 0), gam);
    pp += 0.05;
  }
  return logmean(lb, 10);
}

double ppFn0(double *wknot, double *w, double *postgrid){
  // Interpolates values of function w_0 at all quantiles of interest,
  // based on values tracked at knots.
  int i, l;
  double akapm, zss;
  for(i = 0; i < G; i++){
    mvprod(Agrid[i], wknot, wgrid[i], L, m, 0);
    trisolve(Rgrid[i], m, wknot, zknot, 1);
    zss = sumsquares(zknot, m);
    akapm = 0.1 + 0.5 * (double)m;
    llgrid[i] = -akapm * log1p(0.5 * zss / 0.1);
    postgrid[i] = llgrid[i] - ldRgrid[i] + lpgrid[i];
  }

  double lps = logsum(postgrid, G);

  for(i = 0; i < G; i++){
    postgrid[i] = exp(postgrid[i] - lps);
  }
  for(l = 0; l < L; l++){
    for(w[l] = 0.0, i = 0; i < G; i++){
      w[l] += wgrid[i][l] * postgrid[i];
    }
  }
  return lps;
}

double ppFn(double *wknot, double *w, double *postgrid){
  // Interpolates values of function w_j (j= 1, ... ,p) at all quantiles of interest,
  // based on values tracked at knots.

  // Note: differs from ppFn0 in that ppFn allows for a mixture of gamma priors on kappa
  // each with their own hyperparameters.  ppFn0 uses a Ga(0.1, 0.1) hardcoded prior.

  int i, j, l;
  double akapm, zss;
  for(i = 0; i < G; i++){
    mvprod(Agrid[i], wknot, wgrid[i], L, m, 0);
    trisolve(Rgrid[i], m, wknot, zknot, 1);
    zss = sumsquares(zknot, m);
    for(j = 0; j < nkap; j++){
      akapm = akap[j] + 0.5 * (double)m;
      lw[j] = -akapm * log1p(0.5 * zss/ bkap[j]) + lgamma(akapm) - lgamma(akap[j]) - .5 * (double)m * log(bkap[j]) + lpkap[j];
    }
    llgrid[i] = logsum(lw, nkap);
    postgrid[i] = llgrid[i] - ldRgrid[i] + lpgrid[i];
  }

  double lps = logsum(postgrid, G);
  for(i = 0; i < G; i++){
    postgrid[i] = exp(postgrid[i] - lps);
  }
  for(l = 0; l < L; l++){
    for(w[l] = 0.0, i = 0; i < G; i++){
      w[l] += wgrid[i][l] * postgrid[i];
    }
  }
  return lps;
}

double logpostFn(double *par, double temp, int llonly, double *ll, double **pg){

  int i, j, l, reach = 0, reach2 = 0;
  double w0max, zeta0tot, lps0, gam0, sigma, nu, QPos, QPosold, QNeg, QNegold, sigmat1, sigmat2, den0;

  // Calculate contribution to likelihood for all functions w_j (j=0, ..., p) for each of the
  // response variables (1, ..., q)
  for(i=0; i < n; i++){
    ll[i] = 0.0;
  }

  lps0 = 0;
  for(j=0; j < q; j++){
    reach2 = 0;

    lps0 = ppFn0(&par[reach], &w0[0][j], &pg[reach2][j]);
    reach  += m;
    reach2 += G;
    for(i = 0; i < p; i++){
      lps0 += ppFn(&par[reach], &vMat[j][i][0], &pg[reach2][j]);
      reach += m;
      reach2 += G;
    }
    reach += p+3;  // move to parameters for w functions for next response
  }

  if(temp > 0.0){
    for(i = 0; i < n; i++){
      //ll[i] = log(0.0); // initialize log-likelihood
      ll[i] = 0.0; // initialize log-likelihood
    }

    for(j = 0; j < q; j++){
      w0max = vmax(&w0[0][j], L);

      for(l = 0; l < L; l++){
        zeta0dot[l] = exp(shrinkFactor * (w0[l][j] - w0max));
      }

      // Integrate zeta0dot to get values of zeta0
      trape(&zeta0dot[1], &taugrid[1], L-2, &zeta0[1], 0);

      zeta0tot = zeta0[L-2];

      // Set endpoints for diffeomorphism on [0,1] to be 0 & 1
      zeta0[0]   = 0.0;
      zeta0[L-1] = 1.0;

      for(l = 1; l < L-1; l++){
        zeta0[l] = taugrid[1] + (taugrid[L-2] - taugrid[1]) * zeta0[l] / zeta0tot;
      }

      zeta0dot[0] = 0.0;
      zeta0dot[L-1] = 0.0;

      for(l = 1; l < L-1; l++){
        zeta0dot[l] = (taugrid[L-2] - taugrid[1]) * zeta0dot[l] / zeta0tot;
      }

      for(l = 0; l < L; l++){
        for(vNormSq[l] = 0.0, i = 0; i < p; i++){
          vNormSq[l] += vMat[j][i][l] * vMat[j][i][l];
        }
      }

      if(vmin(vNormSq, L) > 0.0){
        mmprod(vMat[j], x, a, L, p, n, 1, 1, 0); // a is L x n

        for(l = 0; l < L; l++){
          aX[l] = -vmin(a[l], n) / sqrt(vNormSq[l]);
        }

        for(i = 0; i < p; i++){
          for(l = 0; l < L; l++){
            vTilde[i][l] = vMat[j][i][l] / (aX[l] * sqrt(1.0 + vNormSq[l]));
          }
        }

        // Current values of parameters stored in vector par
        gam0 = par[reach++];
        gam  = par + reach;
        reach += p;

        sigma = sigFn(par[reach++]);
        nu    = nuFn(par[reach++]);
        theta = par[q*npar + 1];

        //Rprintf("sigma = %g, nu = %g\n", sigma, nu);

        for(i = 0; i < n; i++){
          xLin[i]   = gam0 + inprod(x[i], gam, p);
          resLin[i] = y[i][j] - xLin[i];
        }
        //Rprintvec("resLin = ", "%g ", resLin, n);

        for(l = 0; l < L; l++){
          b0dot[l] = sigma * q0(zeta0[l], nu) * zeta0dot[l];
        }
        for(i = 0; i < p; i++){
          for(l = 0; l < L; l++){
            bdot[i][l] = b0dot[l] * vTilde[i][l];
          }
        }

        trape(&b0dot[mid], &taugrid[mid], L - mid, Q0Pos, 0);
        Q0Pos[L-mid] = qt(1.0, 1.0, 1, 0);

        trape(&b0dot[mid], &taugrid[mid], mid + 1, Q0Neg, 1);
        Q0Neg[mid+1] = qt(1.0, 1.0, 1, 0);
        //Rprintvec("Q0Pos = ", "%g ", Q0Pos, L - mid + 1);
        //Rprintvec("Q0Neg = ", "%g ", Q0Neg, mid + 2);

        for(i = 0; i < p; i++){
          trape(&bdot[i][mid], &taugrid[mid], L - mid, bPos[j], 0);
          trape(&bdot[i][mid], &taugrid[mid], mid + 1, bNeg[j], 1);
        }

        sigmat1 = sigmat2 = sigma;

        /** l corresponds to index on tau grid of tau_{x_i}(y_i) **/
        for(i = 0; i < n; i++){
          if(resLin[i] == 0.0){
            for(den0 = b0dot[mid], j = 0; j < p; j++){
              den0 += x[i][j] * bdot[j][mid];
            }
            ll[i] += -log(den0);
          }
          else if(resLin[i] > 0.0){
            l = 0;
            QPosold = 0.0;
            for(QPos = Q0Pos[l], j = 0; j < p; j++){
              QPos += x[i][j] * bPos[j][l];
            }
            while(resLin[i] > QPos && l < L-mid-1){
              QPosold = QPos;
              l++;
              for(QPos = Q0Pos[l], j = 0; j < p; j++){
                QPos += x[i][j] * bPos[j][l];
              }
            }
            if(l == L - mid - 1)
              ll[i] += lf0tail(Q0tail(taugrid[L-2]) + (resLin[i] - QPos)/sigmat1) - log(sigmat1);
            else
              ll[i] += log(taugrid[mid+l] - taugrid[mid+l-1]) - log(QPos - QPosold);

            tau_x[i][j] = (taugrid[mid+l] + taugrid[mid+l-1])/2;

          }
          else {
            l = 0;
            QNegold = 0.0;
            for(QNeg = Q0Neg[l], j = 0; j < p; j++){
              QNeg += x[i][j] * bNeg[j][l];
            }
            while(resLin[i] < -QNeg && l < mid){
              QNegold = QNeg;
              l++;
              for(QNeg = Q0Neg[l], j = 0; j < p; j++){
                QNeg += x[i][j] * bNeg[j][l];
              }
            }

            if(l == mid)
              ll[i] += lf0tail(Q0tail(taugrid[1]) + (resLin[i] + QNeg)/sigmat2) - log(sigmat2);
            else
              ll[i] += log(taugrid[mid-l+1]-taugrid[mid-l]) - log(QNeg - QNegold);

            tau_x[i][j] = (taugrid[mid-l+1] + taugrid[mid-l])/2;
          }

          //if(ll[i] == qt(1.0, 1.0, 1, 0)) Rprintf("i = %d, ll[i] = %g, resLin[i] = %g, l = %d\n", i, ll[i], resLin[i], l);
        }

        // Contribution to likelihood for Frank's copula
        for(i = 0; i < n; i++){
          ll[i] += log(theta) +
            log(log(1 + (exp(-theta*tau_x[i][1])-1)*(exp(-theta*tau_x[i][2])-1)/(exp(-theta)-1)));
        }
      }
    }
  } else{
    for(i = 0; i < n; i++) ll[i] = 0.0;
  }

  double lp = temp * sum(ll, n);

  if(!llonly){
    lp += lps0 + dlogis(par[(m+1)*(p+1)+1], 0.0, 1.0, 1);
    if(shrink) lp += dbasejoint(par + m*(p+1) + 1);
  }
  return lp;
}

double lpFn(double *par, double temp){
  return logpostFn(par, temp, 0, llvec, pgvec);
}

/*
 double lpFn2(double sigma){
 par0[sig_pos] = sigma;
 return logpostFn(par0, 1.0, 0, llvec, pgvec);
 }
 */

void DEV(double *par, double *xVar, double *yVar, int *status, int *toShrink, double *hyper, int *dim, double *gridpars, double *tauG, double *devsamp, double *llsamp, double *pgsamp){

  int i, j, k, l;

  int reach = 0;
  shrink = toShrink[0];
  n = dim[reach++]; p = dim[reach++]; L = dim[reach++]; mid = dim[reach++];
  m = dim[reach++]; G = dim[reach++]; nkap = dim[reach++];
  int niter = dim[reach++], npar = (m+1)*(p+1) + 2;

  taugrid = tauG;
  asig = hyper[0]; bsig = hyper[1];
  akap = vect(nkap); bkap = vect(nkap); lpkap = vect(nkap);
  for(reach = 2, i = 0; i < nkap; i++){
    akap[i] = hyper[reach++];
    bkap[i] = hyper[reach++];
    lpkap[i] = hyper[reach++];
  }
  shrinkFactor = shrinkFn((double)p);

  reach = 0;
  Agrid = (double ***)R_alloc(G, sizeof(double **));
  Rgrid = (double ***)R_alloc(G, sizeof(double **));
  ldRgrid = vect(G);
  lpgrid = vect(G);

  for(i = 0; i < G; i++){
    Agrid[i] = Matrix(L, m);
    for(l = 0; l < L; l++) for(k = 0; k < m; k++) Agrid[i][l][k] = gridpars[reach++];

    Rgrid[i] = Matrix(m, m);
    for(k = 0; k < m; k++) for(l = 0; l < m; l++) Rgrid[i][l][k] = gridpars[reach++];

    ldRgrid[i] = gridpars[reach++];
    lpgrid[i] = gridpars[reach++];
  }

  reach = 0;
  x = Matrix(n, p);
  for(j = 0; j < p; j++) for(i = 0; i < n; i++) x[i][j] = xVar[reach++];
  y = yVar;
  cens = status;

  lb = vect(10);
  wgrid = Matrix(G, L);
  lw = vect(nkap);
  llgrid = vect(G);
  zknot = vect(m);
  vMat = Matrix(p, L);
  vTilde = Matrix(p, L);
  w0 = vect(L);
  zeta0dot = vect(L);
  zeta0 = vect(L);
  vNormSq = vect(L);
  a = Matrix(L, n);
  aX = vect(L);
  gam = vect(p);
  xLin = vect(n);
  resLin = vect(n);
  b0dot = vect(L);
  bdot = Matrix(p, L);
  Q0Pos = vect(L);
  bPos = Matrix(p, L);
  Q0Neg = vect(L);
  bNeg = Matrix(p, L);

  reach = 0;
  int iter, reach2 = 0, reach3 = 0;
  for(iter = 0; iter < niter; iter++){
    devsamp[iter] = -2.0 * logpostFn(par + reach, 1.0, 1, llsamp + reach2, pgsamp + reach3);
    reach += npar; reach2 += n; reach3 += G * (p+1);
  }
}

void BJQR(double *par, double *xVar, double *yVar, int *cop,
          int *status, int *toShrink,
          double *hyper, int *dim, double *gridpars, double *tauG, double *siglim,
          double *muVar, double *SVar, int *blocksVar, int *blocks_size,
          double *dmcmcpar, int *imcmcpar,
          double *parsamp, double *acptsamp, double *lpsamp){

  int i, j, k, l, b, reach;
  int mu_point = 0, S_point = 0, blocks_point = 0;

  /** Initialize global parameters **/

  // Dimension parameters
  n    = dim[0];
  p    = dim[1];
  q    = dim[2];
  L    = dim[3];
  mid  = dim[4];
  m    = dim[5];
  G    = dim[6];
  nkap = dim[7];

  // Data
  reach = 0;
  x = Matrix(n, p);
  y = Matrix(n, q);

  for(j = 0; j < p; j++){
    for(i = 0; i < n; i++){
      x[i][j] = xVar[reach++];
    }
  }

  for(j = 0; j < q; j++){
    for(i = 0; i < n; i++){
      y[i][j] = yVar[reach++];
    }
  }

  copula = cop[0];

  taugrid = tauG;

  // MCMC parameters
  niter = dim[8];
  thin  = dim[9];
  npar  = q*((m+1)*(p+1) + 2) + 1;

  nblocks = imcmcpar[0];
  refresh = imcmcpar[1];
  verbose = imcmcpar[2];
  ticker  = imcmcpar[3];
  temp    = dmcmcpar[0];
  decay   = dmcmcpar[1];
  refresh_counter = imcmcpar + 4;
  acpt_target     = dmcmcpar + 2;  // Pointer to element 3
  lm     = dmcmcpar + 2 + nblocks; // Pointer to element 3 + nblocks

  Rprintf("Got to 1!/n");

  // Prior parameters
  reach = 0;
  Agrid = (double ***)R_alloc(G, sizeof(double **));
  Rgrid = (double ***)R_alloc(G, sizeof(double **));
  ldRgrid = vect(G);
  lpgrid  = vect(G);

  Rprintf("Got to 2!/n");

  for(i = 0; i < G; i++){
    Agrid[i] = Matrix(L, m);
    for(l = 0; l < L; l++) for(k = 0; k < m; k++) Agrid[i][l][k] = gridpars[reach++];

    Rgrid[i] = Matrix(m, m);
    for(k = 0; k < m; k++) for(l = 0; l < m; l++) Rgrid[i][l][k] = gridpars[reach++];

    ldRgrid[i] = gridpars[reach++];
    lpgrid[i] = gridpars[reach++];
  }

  Rprintf("Got to 3!/n");

  mu     = (double **)R_alloc(nblocks, sizeof(double *));
  S      = (double ***)R_alloc(nblocks, sizeof(double **));
  blocks = (int **)R_alloc(nblocks, sizeof(int *));

  for(b = 0; b < nblocks; b++){
    mu[b] = muVar + mu_point; mu_point += blocks_size[b];
    S[b] = (double **)R_alloc(blocks_size[b], sizeof(double *));
    for(i = 0; i < blocks_size[b]; i++){
      S[b][i] = SVar + S_point;
      S_point += blocks_size[b];
    }
    blocks[b] = blocksVar + blocks_point; blocks_point += blocks_size[b];
  }
  cens = status;
  shrink = toShrink[0];
  shrinkFactor = shrinkFn((double)p);

  Rprintf("Got to 4!/n");

  // Hyperparameters
  asig = hyper[0]; bsig = hyper[1];
  akap = vect(nkap); bkap = vect(nkap); lpkap = vect(nkap);
  for(reach = 2, i = 0; i < nkap; i++){
    akap[i] = hyper[reach++];
    bkap[i] = hyper[reach++];
    lpkap[i] = hyper[reach++];
  }

  Rprintf("Got to 5!/n");

  // Model parameters
  zeta0    = vect(L);
  zeta0dot = vect(L);
  gam      = Matrix(p, q);
  w0       = Matrix(L, q);
  wMat = (double ***)R_alloc(q, sizeof(double **));
  vMat = (double ***)R_alloc(q, sizeof(double **));

  Rprintf("Got to 6!/n");

  for(j = 0; j < q; j++){
    wMat[j] = Matrix(p, L);
    wMat[j] = Matrix(p, L);
  }

  Rprintf("Got to 7!/n");

  // Intermediate calculations
  wgrid  = Matrix(G, L);
  a      = Matrix(L, n);
  vTilde = Matrix(p, L);
  bPos   = Matrix(p, L);
  bNeg   = Matrix(p, L);
  bdot   = Matrix(p, L);
  tau_x  = Matrix(n, j);

  aX      = vect(L);
  b0dot   = vect(L);
  vNormSq = vect(L);
  Q0Pos   = vect(L);
  Q0Neg   = vect(L);
  zknot   = vect(m);
  xLin    = vect(n);
  resLin  = vect(n);
  llvec   = vect(n);
  llgrid  = vect(G);

  pgvec = Matrix(G*(p+1), q);

  lb = vect(10);
  lw = vect(nkap);

  parSample = vect(npar);
  par0      = vect(npar);

  for(i = 0; i < npar; i++) parSample[i] = par[i];
  for(i = 0; i < npar; i++) par0[i]      = par[i];

  Rprintf("Got to 8!/n");

  /******** Need to update for multiple sigmas!!!! *********/
  //double sig_a = siglim[0], sig_b = siglim[1];
  //double fa = lpFn2(sig_a);
  //double fb = lpFn2(sig_b);
  //Max_Search_Golden_Section(lpFn2, &sig_a, &fa, &sig_b, &fb, 1.0e-5);
  //sig_pos = (m+1)*(p+1);

  for(j = 0; j < q; j++){
    parSample[(j+1)*npar - 1] = (siglim[2*j] + siglim[2*j+1]) / 2;
  }

  /**********************************************************/

  adMCMC(niter, thin, npar, parSample, mu, S, acpt_target, decay, refresh,
         lm, temp, nblocks, blocks, blocks_size, refresh_counter, verbose,
         ticker, lpFn, parsamp, acptsamp, lpsamp);
}

/*
 static int Stopping_Rule(double x0, double x1, double tolerance){
 double xm = 0.5 * fabs( x1 + x0 );

 if ( xm <= 1.0 ) return ( fabs( x1 - x0 ) < tolerance ) ? 1 : 0;
 return ( fabs( x1 - x0 ) < tolerance * xm ) ? 1 : 0;
 }

 void Max_Search_Golden_Section( double (*f)(double), double* a, double *fa, double* b, double* fb, double tolerance){
 static const double lambda = 0.5 * (sqrt5 - 1.0);
 static const double mu = 0.5 * (3.0 - sqrt5);         // = 1 - lambda
 double x1;
 double x2;
 double fx1;
 double fx2;

 x1 = *b - lambda * (*b - *a);
 x2 = *a + lambda * (*b - *a);
 fx1 = f(x1);
 fx2 = f(x2);

 if (tolerance <= 0.0) tolerance = 1.0e-5 * (*b - *a);

 while ( ! Stopping_Rule( *a, *b, tolerance) ) {
 if (fx1 < fx2) {
 *a = x1;
 *fa = fx1;
 if ( Stopping_Rule( *a, *b, tolerance) ) break;
 x1 = x2;
 fx1 = fx2;
 x2 = *b - mu * (*b - *a);
 fx2 = f(x2);
 } else {
 *b = x2;
 *fb = fx2;
 if ( Stopping_Rule( *a, *b, tolerance) ) break;
 x2 = x1;
 fx2 = fx1;
 x1 = *a + mu * (*b - *a);
 fx1 = f(x1);
 }
 }
 return;
 }
 */
//------ mcmc ------//

void adMCMC(int niter, int thin, int npar, double *par, double **mu, double ***S, double *acpt_target,
            double decay, int refresh, double *lm, double temp, int nblocks, int **blocks, int *blocks_size,
            int *refresh_counter, int verbose, int ticker, double lpFn(double *, double),
            double *parsamp, double *acptsamp, double *lpsamp){

  int b, i, j, ipar = 0, iparnew = 1;
  int iter, store_lp = 0, store_par = 0, store_acpt = 0;

  double lpval, lpvalnew, lp_diff, chs, lambda;

  int *chunk_size  = ivect(nblocks);
  int *blocks_rank = ivect(nblocks);
  int *run_counter = ivect(nblocks);

  double *acpt_chunk = vect(nblocks);
  double *alpha      = vect(nblocks);
  double *frac       = vect(nblocks);
  double *blocks_d   = vect(npar);
  double *par_incr   = vect(npar);
  double *zsamp      = vect(npar);
  double *alpha_run  = vect(nblocks);

  int **blocks_pivot = (int **)R_alloc(nblocks, sizeof(double *));

  double **parbar_chunk = (double **)R_alloc(nblocks, sizeof(double *));

  double **parstore = Matrix(2, npar);


  double ***R = (double ***)R_alloc(nblocks, sizeof(double **));


  for(b = 0; b < nblocks; b++){
    R[b] = Matrix(blocks_size[b], blocks_size[b]);
    blocks_pivot[b] = ivect(blocks_size[b]);
    chol(R[b], blocks_size[b], 0.0, blocks_pivot[b], blocks_rank + b, blocks_size[b], blocks_d, S[b], 0, 0, 1.0e-10);
  }

  for(b = 0; b < nblocks; b++){
    chunk_size[b] = 0;
    acpt_chunk[b] = 0.0;
    parbar_chunk[b] = vect(blocks_size[b]);
    for(i = 0; i < blocks_size[b]; i++){
      parbar_chunk[b][i] = 0.0;
    }
    frac[b] = sqrt(1.0 / ((double)refresh_counter[b] + 1.0));
  }

  for(i = 0; i < npar; i++){
    parstore[0][i] = par[i];
  }

  lpval = lpFn(parstore[ipar], temp);
  if(verbose) Rprintf("Initial lp = %g\n", lpval);


  for(b = 0; b < nblocks; b++){
    alpha_run[b]   = 0.0;
    run_counter[b] = 0;
  }

  GetRNGstate();
  for(iter = 0; iter < niter; iter++){
    for(b = 0; b < nblocks; b++){
      chunk_size[b]++;
      for(i = 0; i < blocks_size[b]; i++){
        zsamp[i] = rnorm(0.0, 1.0);
      }
      triprod(R[b], blocks_size[b], blocks_size[b], zsamp, par_incr, 1);
      for(i = 0; i < npar; i++){
        parstore[iparnew][i] = parstore[ipar][i];
      }
      lambda = lm[b] * sqrt(3.0 / rgamma(3.0, 1.0));
      for(i = 0; i < blocks_size[b]; i++){
        parstore[iparnew][blocks[b][i]] += lambda * par_incr[i];
      }
      lpvalnew = lpFn(parstore[iparnew], temp);
      lp_diff = lpvalnew - lpval;
      alpha[b] = exp(lp_diff); if(alpha[b] > 1.0) alpha[b] = 1.0;
      if(log(runif(0.0, 1.0)) < lp_diff){
        ipar = iparnew;
        iparnew = !ipar;
        lpval = lpvalnew;
      }
      alpha_run[b] = ((double)run_counter[b] * alpha_run[b] + alpha[b]) / ((double)(run_counter[b] + 1.0));
      run_counter[b]++;
    }
    if((iter + 1) % thin == 0){
      lpsamp[store_lp++] = lpval;
      for(i = 0; i < npar; i++){
        parsamp[store_par++] = parstore[ipar][i];
      }
      for(b = 0; b < nblocks; b++){
        acptsamp[store_acpt++] = alpha[b];
      }
    }

    if(verbose){
      if(niter < ticker || (iter + 1) % (niter / ticker) == 0){
        Rprintf("iter = %d, lp = %g ", iter + 1, lpval);
        Rprintvec("acpt = ", "%0.2f ", alpha_run, nblocks);
        for(b = 0; b < nblocks; b++){
          alpha_run[b] = 0.0;
          run_counter[b] = 0;
        }
      }
    }

    for(b = 0; b < nblocks; b++){
      chs = (double) chunk_size[b];
      if(chs < 1.0) chs = 1.0;

      acpt_chunk[b] = acpt_chunk[b] + (alpha[b] - acpt_chunk[b]) / chs;

      for(i = 0; i < blocks_size[b]; i++){
        parbar_chunk[b][i] += (parstore[ipar][blocks[b][i]] - parbar_chunk[b][i]) / chs;
      }

      if(chunk_size[b] == refresh * blocks_size[b]){
        refresh_counter[b]++;
        frac[b] = sqrt(1.0 / ((double)refresh_counter[b] + 1.0));
        for(i = 0; i < blocks_size[b]; i++){
          for(j = 0; j < i; j++){
            S[b][i][j] = (1.0 - frac[b]) * S[b][i][j] + frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][j] - mu[b][j]);
            S[b][j][i] = S[b][i][j];
          }
          S[b][i][i] = (1.0 - frac[b]) * S[b][i][i] + frac[b] * (parbar_chunk[b][i] - mu[b][i]) * (parbar_chunk[b][i] - mu[b][i]);
        }
        chol(R[b], blocks_size[b], 0.0, blocks_pivot[b], blocks_rank + b, blocks_size[b], blocks_d, S[b], 0, 0, 1.0e-10);

        for(i = 0; i < blocks_size[b]; i++){
          mu[b][i] += frac[b] * (parbar_chunk[b][i] - mu[b][i]);
        }

        lm[b] *= exp(frac[b] * (acpt_chunk[b] - acpt_target[b]));

        acpt_chunk[b] = 0.0;

        for(i = 0; i < blocks_size[b]; i++){
          parbar_chunk[b][i] = 0.0;
        }
        chunk_size[b] = 0;
      }
    }
  }
  PutRNGstate();
  for(i = 0; i < npar; i++){
    par[i] = parstore[ipar][i];
  }
}
