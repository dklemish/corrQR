void adMCMC(int niter, int thin, int npar, double *par, double **mu, double ***S, double *acpt_target,
            double decay, int refresh, double *lm, double temp, int nblocks, int **blocks, int *blocks_size,
            int *refresh_counter, int verbose, int ticker, double lpFn(double *, double),
            double *parsamp, double *acptsamp, double *lpsamp){

  int b, i, j;
  int iter;
  int store_lp   = 0;
  int store_par  = 0;
  int store_acpt = 0;
  int ipar = 0, iparnew = 1;

  double lpval, lpvalnew, lp_diff;

  double chs, lambda;


  double ***R = (double ***)R_alloc(nblocks, sizeof(double **));

  double **parbar_chunk = (double **)R_alloc(nblocks, sizeof(double *));
  double **parstore = mymatrix(2, npar);

  int **blocks_pivot = (int **)R_alloc(nblocks, sizeof(double *));
  int *blocks_rank = ivect(nblocks);

  int *chunk_size    = ivect(nblocks);
  int *run_counter   = ivect(nblocks);

  double *acpt_chunk = vect(nblocks);
  double *alpha      = vect(nblocks);
  double *frac       = vect(nblocks);
  double *blocks_d   = vect(npar);
  double *par_incr   = vect(npar);
  double *zsamp      = vect(npar);
  double *alpha_run = vect(nblocks);

  for(b = 0; b < nblocks; b++){
    R[b] = mymatrix(blocks_size[b], blocks_size[b]);
    blocks_pivot[b] = ivect(blocks_size[b]);
    parbar_chunk[b] = vect(blocks_size[b]);

    chol(R[b], blocks_size[b], 0.0, blocks_pivot[b], blocks_rank + b, blocks_size[b], blocks_d, S[b], 0, 0, 1.0e-10);

    chunk_size[b] = 0;
    acpt_chunk[b] = 0.0;
    for(i = 0; i < blocks_size[b]; i++)
      parbar_chunk[b][i] = 0.0;

    frac[b] = sqrt(1.0 / ((double)refresh_counter[b] + 1.0));

  }

  for(i = 0; i < npar; i++) parstore[0][i] = par[i];

  lpval = lpFn(parstore[ipar], temp);
  if(verbose) Rprintf("Initial lp = %g\n", lpval);

  for(b = 0; b < nblocks; b++) alpha_run[b] = 0.0;
  for(b = 0; b < nblocks; b++) run_counter[b] = 0;

  GetRNGstate();
  for(iter = 0; iter < niter; iter++){
    for(b = 0; b < nblocks; b++){
      chunk_size[b]++;
      for(i = 0; i < blocks_size[b]; i++)
        zsamp[i] = rnorm(0.0, 1.0);

      triprod(R[b], blocks_size[b], blocks_size[b], zsamp, par_incr, 1);

      for(i = 0; i < npar; i++)
        parstore[iparnew][i] = parstore[ipar][i];

      lambda = lm[b] * sqrt(3.0 / rgamma(3.0, 1.0));

      for(i = 0; i < blocks_size[b]; i++)
        parstore[iparnew][blocks[b][i]] += lambda * par_incr[i];

      lpvalnew = lpFn(parstore[iparnew], temp);

      lp_diff = lpvalnew - lpval;

      alpha[b] = exp(lp_diff);
      if(alpha[b] > 1.0)
        alpha[b] = 1.0;

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
      for(i = 0; i < npar; i++)
        parsamp[store_par++] = parstore[ipar][i];
      for(b = 0; b < nblocks; b++)
        acptsamp[store_acpt++] = alpha[b];
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

      for(i = 0; i < blocks_size[b]; i++)
        parbar_chunk[b][i] += (parstore[ipar][blocks[b][i]] - parbar_chunk[b][i]) / chs;

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

        for(i = 0; i < blocks_size[b]; i++)
          mu[b][i] += frac[b] * (parbar_chunk[b][i] - mu[b][i]);

        lm[b] *= exp(frac[b] * (acpt_chunk[b] - acpt_target[b]));

        acpt_chunk[b] = 0.0;

        for(i = 0; i < blocks_size[b]; i++)
          parbar_chunk[b][i] = 0.0;

        chunk_size[b] = 0;
      }
    }
  }
  PutRNGstate();
  for(i = 0; i < npar; i++)
    par[i] = parstore[ipar][i];
}
