corrQR <- function(x, y, sd, Rcorr,
                   nsamp = 1e3, thin = 10,
                   incr = 0.01,
                   ip = NULL,
                   nknots = 6,
                   hyper = list(sig = c(.1,.1),
                                lam = c(6,4),
                                kap = c(0.1,0.1,1)),
                   prox.range = c(.2,.95),
                   acpt.target = 0.15,
                   ref.size = 3,
                   blocking = "by.response",
                   expo = 2,
                   blocks.mu, blocks.S,
                   fix.nu = FALSE, fix.corr = FALSE){
  # Input parameters:
  #  x - Predictors (n by p matrix)
  #  y - Response (n by q matrix)
  #  sd - Random seed
  #  Rcorr - User provided correlation matrix for Gaussian copula
  #  nsamp - number of MCMC samples
  #  thin - thinning factor
  #  incr - increment in quantile grid tau to be analyzed
  #  par -
  #  nknots - number of knots in the low-rank approximations to GP priors
  #    for functions w_j (j=0, ..., p)
  #  hyper - hyperparameters on
  #   * sigma -
  #   * lambda - a_lambda and b_lambda hyperparameters for prior on
  #       lambda, the GP rescaling factor
  #   * kappa -
  # shrink -
  # prox.range - corresponds to range for rho(lambda) (see section 3.2)
  # acpt.target -
  # ref.size -
  # blocking - option for how to block parameters in MCMC updates
  # temp -
  # expo - exponent to be used in GP covariance function
  # blocks.mu -
  # blocks.s -
  # fix.nu - option on whether nu parameter should be included in MCMC
  # fix.corr - option on whether correlation matrix for Gaussian copula should
  #   be learned.

  #### Read in data ####
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  x.names <- dimnames(x)[[2]]

  y <- as.matrix(y)
  q <- ncol(y)
  y.names <- dimnames(y)[[2]]

  if(is.null(x.names)) x.names <- paste("X", 1:p, sep = "")
  if(is.null(y.names)) y.names <- paste("Y", 1:q, sep = "")

  # Scale & center (around the point selected by the algorithm
  # used to select a point in the convex hull) the covariates x
  x <- scale(x, chull.center(x))

  # Map selected copula to numeric code
  # if(copula == "gaussian") copulaMethod <- 1

  #### Necessary hyperparameters ####

  a.sig <- hyper$sig;
  if(is.null(a.sig))
    a.sig <- c(.1, .1)

  a.lam <- hyper$lam;
  if(is.null(a.lam))
    a.lam <- c(6, 4)

  a.kap <- hyper$kap;
  if(is.null(a.kap))
    a.kap <- c(1.5, 1.5, 1);

  a.kap <- matrix(a.kap, nrow = 3);
  nkap <- ncol(a.kap);
  a.kap[3,] <- log(a.kap[3,])

  hyperPar <- c(a.sig, c(a.kap))


  #### Set up quantile grid ####

  # Grid of quantiles of interest includes multiples of "incr"
  # plus additional points in the tail based on the # of data points

  Ltail <- ceiling(log(n*incr,2)) # of points in tail
  # Ltail <- ceiling(log(n*1000*incr,2)) # of points in tail

  # tau.g = grid of quantiles
  # L = # of quantiles
  # mid = array position of quantile closest to 0.5
  # reg.ix = indices of quantiles not in the tail
  # tau.k = reduced grid of quantiles at which functions will be tracked

  if(Ltail > 0){
    tau.tail <- incr / 2^(Ltail:1)
    tau.g <- c(0, tau.tail, seq(incr, 1 - incr, incr), 1 - tau.tail[Ltail:1], 1)
    L <- length(tau.g)
    mid <- which.min(abs(tau.g - 0.5))
    reg.ix <- (1:L)[-c(1 + 1:Ltail, L - 1:Ltail)]
  } else {
    tau.g <- seq(0, 1, incr)
    L <- length(tau.g)
    mid <- which.min(abs(tau.g - 0.5))
    reg.ix <- (1:L)
  }

  tau.k <- seq(0,1,len = nknots)

  #### Discrete approximation to prior on lambda ####
  # See section 3.2 of Yang & Tokdar

  # prox.grid  = points approximating possible values of lambda (scale parameter in GPs)
  # prior.grid = approximation to prior pdf for lambda at values of prox.grid
  # lp.grid    = log of prior.grid
  # d.kg       = (tau - tau')^expo for tau \in tau.k, tau' \in tau.g
  # d.kk       = (tau - tau')^expo for tau \in tau.k, tau' \in tau.k
  #  d.kg, d.kk used for computation of A_g (each g corresponds to different
  #  possible value of lambda)
  # gridmats   = GP parameters to pass to MCMC
  #  ith column contains A_g, R_g, the log determinant of R_g, and the
  #  log prior density for that value of lambda

  prox.grid  <- proxFn(max(prox.range), min(prox.range), 0.5)
  ngrid      <- length(prox.grid)
  lamsq.grid <- lamFn(prox.grid)^2
  prior.grid <- -diff(pbeta(c(1,
                              (prox.grid[-1] + prox.grid[-ngrid])/2,
                              0),
                            a.lam[1],
                            a.lam[2]))
  lp.grid <- log(prior.grid)

  d.kg <- abs(outer(tau.k, tau.g, "-"))^expo
  d.kk <- abs(outer(tau.k, tau.k, "-"))^expo

  A <- list()
  R <- list()
  log.det <- rep(0, ngrid)

  # Initialize gridmats
  K0 <- 0
  t1 <- Sys.time()
  for(i in 1:ngrid){
    # K.grid = covariances between elements in tau.k and tau.g
    # K.knot = covariances between elements in tau.k
    K.grid <- exp(-lamsq.grid[i] * d.kg)
    K.knot <- exp(-lamsq.grid[i] * d.kk)
    diag(K.knot) <- 1 + 1e-10   # for numerical stability

    R[[i]] <- chol(K.knot)
    A[[i]] <- t(solve(K.knot, K.grid))
    log.det[i] <- sum(log(diag(R[[i]])))

    K0 <- K0 + prior.grid[i] * K.knot
  }
  t2 <- Sys.time()

  # npar  = number of parameters for each response variable
  # niter = number of MCMC iterations
  # ncorr = number of correlation parameters for Gaussian copula
  npar <- (nknots+1) * (p+1) + 2
  niter  <- nsamp * thin
  ncorr <- q*(q-1)/2

  tot.par <- q*npar + ncorr

  dimpars <- c(n, p, q, L, mid - 1, nknots, ngrid, ncol(a.kap), niter, thin, nsamp, ncorr)

  #### Initialize parameter vector for better MCMC mixing
  # Use qrjoint on individual responses as starting estimate
  # for parameters of each response

  # par vector contains paramters to track in MCMC
  #  First response
  #    1 : nknots = value of w0_1 for 1st response
  #    (nknots+1) : 2*nknots = w1_1
  #    ...
  #    (p*nknots + 1) : (p+1)*nknots = wp_1
  #    (p+1)*nknots + 1 = gamma0_1
  #    (p+1)*nknots + 1 + 1:p = gamma_1
  #    (p+1)*(nknots+1) + 1 = sigma_1
  #    (p+1)*(nknots+1) + 2 = nu_1
  #  ...
  #  Qth response
  #    (q-1)*npar + 1:nknots = value of w0_q
  #    (q-1)*npar + (nknots+1) : 2*nknots = value of w1_q
  #    ...
  #    (q-1)*npar + (p*nknots + 1) : (p+1)*nknots = value of wp_q
  #    (q-1)*npar + (p+1)*nknots + 1 = gamma0_q
  #    (q-1)*npar + (p+1)*nknots + 1 + 1:p = gamma_q
  #    (q-1)*npar + (p+1)*(nknots+1) + 1 = sigma_q
  #    (q-1)*npar + (p+1)*(nknots+1) + 2 = nu_q
  #  Copula parameters
  #    q*npar + 1:ncorr = copula parameters

  set.seed(sd)

  # Initialize w functions, gamma0, gamma, sigma & nu
  par <- rep(0, tot.par)

  if(is.null(ip)){
    for(j in 1:q){
      suppressWarnings(
        invis <- capture.output(
          par[(j-1)*npar + 1:npar] <-
            qrjoint(x, y[,j], nsamp = 100, thin=1)$par
        )
      )
    }
  }else{
    par <- ip
  }

  # Build list that contains which indices of parameter vector
  # should be updated in each block update in each MCMC iteration
  if(blocking == "single"){
    blocks <- list(rep(TRUE, tot.par - ncorr))
  } else if(blocking == "by.response"){
    blocks <- replicate(q*(p + 3) + 1, rep(FALSE, tot.par - ncorr), simplify = FALSE)

    for(j in 1:q){
      # Parameters for w & gamma functions for one response variable in separate blocks
      for(i in 0:p){
        blocks[[(j-1)*(p+1) + i + 1]][c((j-1)*npar + (i * nknots + 1:nknots),
                                        (j-1)*npar + nknots*(p+1) + i + 1)] <- TRUE
      }

      # Parameters for gamma functions for one response variable in separate blocks
      blocks[[q*(p+1) + j]][(j-1)*npar + nknots*(p+1) + 1:(p+1)] <- TRUE

      # Parameters for sigma & nu for one response variable
      blocks[[q*(p+2) + j]][c((j-1)*npar + (nknots+1)*(p+1) + 1:2)] <- TRUE
    }

    # All parameters except correlation parameters (as these are sampled
    # in a different manner)
    blocks[[q*(p+3)+1]][1:(q*npar)] <- TRUE
  } else if(blocking == "by.function"){
    blocks <- replicate(p + 4, rep(FALSE, npar*q + ncorr), simplify = FALSE)

    # Parameters for w & gamma functions for all response variables
    for(i in 0:p){
      blocks[[i + 1]][c(sapply((1:q - 1)*npar, function(x) x + i*nknots + 1:nknots),
                        sapply((1:q - 1)*npar, function(x) x + nknots*(p+1) + i + 1))] <- TRUE
    }

    # Parameters for gamma functions for all response variables
    blocks[[p+2]][c(sapply((1:q - 1)*npar, function(x) x + nknots*(p+1) + 1:(p+1)))] <- TRUE

    # Parameters for sigma & nu for all response variables
    blocks[[p+3]][c(sapply((1:q - 1)*npar, function(x) x + (nknots+1)*(p+1) + 1:2))] <- TRUE

    # All parameters except correlation parameters (as these are sampled
    # in a different manner)
    blocks[[p+4]][1:(q*npar)] <- TRUE
  } else {
    # Sample each parameter separately
    blocks <- replicate(q*npar, rep(FALSE, q*npar), simplify = FALSE)
    for(i in 1:(q*npar)) blocks[[i]][i] <- TRUE
  }

  nblocks <- length(blocks)

  # If user selects a fixed nu, change update status for all nu values to False
  # in all blocks.
  if(fix.nu){
    for(j in 1:nblocks){
      blocks[[j]][c(sapply((1:q - 1)*npar, function(x) x + npar))] <- FALSE
    }
  }

  blocks.ix <- c(unlist(lapply(blocks, which))) - 1
  blocks.size <- sapply(blocks, sum)

  # Initialize prior mean & covariances for proposal distributions
  if(missing(blocks.mu)){
    blocks.mu <- lapply(blocks.size, function(n) vector("numeric", n))
  }

  if(missing(blocks.S)){
    blocks.S <- lapply(blocks.size, function(q) diag(1, q))

    if(blocking == "by.response"){
      for(i in 1:(q*(p+1))){
        blocks.S[[i]][1:nknots, 1:nknots] <- K0
      }
      for(i in 1:q){
        suppressWarnings(
          blocks.S[[q*(p+1) + i]] <- summary(rq(y[,i] ~ x, tau = 0.5),
                                             se = "boot",
                                             cov = TRUE)$cov
        )
      }
      for(i in 1:q){
        blocks.S[[q*(p+2) + i]] <- matrix(c(1,0,0,0.1), nrow=2, ncol=2)
      }

      # Determine prior joint covariance matrix of all blocks
      slist <- list()
      length(slist) <- q*(p+3)
      for(i in 1:q){
        for(j in 1:(p+1)){
          slist[[(i-1)*(p+3) + j]] <- K0
        }

        slist[[(i-1)*(p+3) + p + 2]] <-
          suppressWarnings(
            summary(rq(y[,i] ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov
          )
        slist[[i*(p+3)]] <- matrix(c(1,0,0,0.1), nrow=2, ncol=2)
      }

      blocks.S[[q*(p+3) + 1]] <- as.matrix(bdiag(slist))
    } else if(blocking == "by.function"){
      for(j in 1:q){
        for(i in 1:(p+1)){
          blocks.S[[i]][((j-1)*(nknots+1)+1):((j-1)*(nknots+1)+nknots),
                        ((j-1)*(nknots+1)+1):((j-1)*(nknots+1)+nknots)] <- K0
        }

        blocks.S[[p+2]][((j-1)*(p+1) + 1):(j*(p+1)),
                        ((j-1)*(p+1) + 1):(j*(p+1))] <-
          suppressWarnings(
            summary(rq(y[,j] ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov
          )
        blocks.S[[p+3]][(j-1)*2 + 1:2, (j-1)*2 + 1:2] <-
          matrix(c(1,0,0,0.1), nrow=2, ncol=2)

      }
      blocks.S[[p+3]][q*j, q*j] <- 0.1

      # Determine prior joint covariance matrix of all variables
      slist <- list()
      length(slist) <- p+3

      for(i in 1:q){
        for(j in 1:(p+1)){
          slist[[(i-1)*(p+3) + j]] <- K0
        }

        slist[[(i-1)*(p+3) + p + 2]] <-
          suppressWarnings(
            summary(rq(y[,i] ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov
          )

        slist[[i*(p+3)]] <- matrix(c(1,0,0,0.1), nrow=2, ncol=2)
      }

      blocks.S[[p+4]] <- as.matrix(bdiag(slist))
    }
  }

  # Initialize correlation paramters for Gaussian copula
  if(!missing(Rcorr)){
    reach <- 1

    for(i in 1:(q-1)){
      for(j in (i+1):q){
        par[q*npar + reach] <- Rcorr[i,j]
        reach <- reach + 1
      }
    }
  }

  imcmc.par <- c(nblocks, ref.size, TRUE, max(10, niter/1e4), rep(0, nblocks), fix.corr)
  dmcmc.par <- c(0.999, rep(acpt.target, nblocks), 2.38 / sqrt(blocks.size))

  tm.cpp <- system.time(
    oo <- .Call('corrQR_corr_qr_fit', PACKAGE='corrQR',
                par, x, y, hyperPar, dimpars, A, R,
                log.det, lp.grid, tau.g,
                blocks.mu, blocks.S,
                blocks.ix, blocks.size, dmcmc.par, imcmc.par)
  )
  set.seed(NULL)

  oo$x        <- x
  oo$y        <- y
  oo$xnames   <- x.names
  oo$ynames   <- y.names
  oo$tau.g    <- tau.g
  oo$dim      <- dimpars

  oo$A        <- A
  oo$R        <- R
  oo$log.det  <- log.det
  oo$lp.grid  <- lp.grid

  oo$prox     <- prox.grid
  oo$reg.ix   <- reg.ix
  oo$hyper    <- hyperPar
  oo$imcmcpar <- imcmc.par
  oo$dmcmcpar <- dmcmc.par
  oo$runtime  <- tm.cpp[3]

  samples  <- as.data.frame(oo$parsamp)
  parnames <- colnames(samples)

  reach <- 1

  for(i in 1:q){
    for(j in 0:p){
      for(k in 1:nknots){
        parnames[reach] <- paste("W", i, j, "(", k, ")", sep="")
        reach <- reach + 1
      }
    }
    for(j in 0:p){
      parnames[reach] <- paste("gamma", i, j, sep="")
      reach <- reach+1
    }
    parnames[reach] <- paste("sigma", i, sep="")
    parnames[reach+1] <- paste("nu", i, sep="")
    reach <- reach + 2
  }
  for(i in 1:(q-1)){
    for(j in (i+1):q){
      parnames[reach] <- paste("rho",i,j,sep="")
      reach <- reach + 1
    }
  }
  colnames(samples) <- parnames
  oo$parsamp <- samples

  class(oo) <- "corrQR"
  return(oo)
}

waic <- function(logliks, print = TRUE){
  lppd <- sum(apply(logliks, 1, logmean))
  p.waic.1 <- 2 * lppd - 2 * sum(apply(logliks, 1, mean))
  p.waic.2 <- sum(apply(logliks, 1, var))
  waic.1 <- -2 * lppd + 2 * p.waic.1
  waic.2 <- -2 * lppd + 2 * p.waic.2
  if(print) cat("WAIC.1 =", round(waic.1, 2), ", WAIC.2 =", round(waic.2, 2), "\n")
  invisible(c(WAIC1 = waic.1, WAIC2 = waic.2))
}

sum.sq    <- function(x) return(sum(x^2))
extract   <- function(lo, vn) return(lo[[vn]])
lamFn     <- function(prox) return(sqrt(-100*log(prox)))
nuFn      <- function(z) return(0.5 + 5.5*exp(z/2))
nuFn.inv  <- function(nu) return(2*log((nu - 0.5)/5.5))
sigFn     <- function(z, a.sig) return(exp(z/2))
sigFn.inv <- function(s, a.sig) return(2 * log(s))
unitFn    <- function(u) return(pmin(1 - 1e-10, pmax(1e-10, u)))
logmean   <- function(lx) return(max(lx) + log(mean(exp(lx - max(lx)))))
logsum    <- function(lx) return(logmean(lx) + log(length(lx)))
trape     <- function(x, h, len = length(x)) return(c(0, cumsum(.5 * (x[-1] + x[-len]) * (h[-1] - h[-len]))))
transform.grid <- function(w, ticks, dists){return((1-dists) * w[ticks] + dists * w[ticks+1])}

q0 <- function(u, nu = Inf) return(1 / (dt(qt(unitFn(u), df = nu), df = nu) * qt(.9, df = nu)))
Q0 <- function(u, nu = Inf) return(qt(unitFn(u), df = nu) / qt(.9, df = nu))
F0 <- function(x, nu = Inf) return(pt(x*qt(.9, df = nu), df = nu))

klGP <- function(lam1, lam2, nknots = 11){
  # Determine KL divergence between two multivariate normals with mean zero and
  # covariances determined by squared exponential covariance functions with
  # differing lambda / length parameters.

  # Used to choose specific lambda values to approximate the prior on lambda;
  # values are chosen so that the KL divergence between consecutive lambda values
  # is approximately 1.
  tau <- seq(0, 1, len = nknots)
  dd  <- outer(tau, tau, "-")^2

  K1        <- exp(-lam1^2 * dd)
  diag(K1)  <- 1 + 1e-10
  R1        <- chol(K1)
  log.detR1 <- sum(log(diag(R1)))

  K2        <- exp(-lam2^2 * dd)
  diag(K2)  <- 1 + 1e-10
  R2        <- chol(K2)
  log.detR2 <- sum(log(diag(R2)))

  return(log.detR2-log.detR1 - 0.5 * (nknots - sum(diag(solve(K2, K1)))))
}

proxFn <- function(prox.Max, prox.Min, kl.step = 1){
  prox.grid <- prox.Max
  j <- 1
  while(prox.grid[j] > prox.Min){
    prox1 <- prox.grid[j]
    prox2 <- prox.Min
    kk <- klGP(lamFn(prox1), lamFn(prox2))
    while(kk > kl.step){
      prox2 <- (prox1 + prox2)/2
      kk <- klGP(lamFn(prox1), lamFn(prox2))
    }
    j <- j + 1
    prox.grid <- c(prox.grid, prox2)
  }
  return(prox.grid)
}

trace.plot <- function(object){
  p       <- object$dim[2]
  q       <- object$dim[3]
  nknots  <- object$dim[6]
  ncorr   <- object$dim[12]
  npar    <- (nknots+1) * (p+1) + 2
  totpar  <- npar*q + ncorr

  pars <- cbind(1:nrow(object$parsamp), object$parsamp)
  colnames(pars)[1] <- "Iter"
  pl <- lapply(2:(totpar+1), function(.x)
    ggplot(pars, aes(x=Iter, y=pars[,.x])) +
      geom_line() + theme_bw() +
      ylab(colnames(pars)[.x])
  )
  ml <- marrangeGrob(pl, nrow=4, ncol=4)
  ml
}

ppFn0 <- function(w.knot, A, R, ld, lp, L, nknots, ngrid){
  w.grid     <- matrix(NA, L, ngrid)
  lpost.grid <- rep(NA, ngrid)
  for(i in 1:ngrid){
    r             <- sum.sq(backsolve(R[[i]], w.knot, transpose = TRUE))
    w.grid[,i]    <- A[[i]] %*% w.knot
    lpost.grid[i] <- -(0.5*nknots+1.5)*log1p(0.5*r/1.5) - ld[i] + lp[i]
  }

  lpost.sum <- logsum(lpost.grid)
  post.grid <- exp(lpost.grid - lpost.sum)
  w         <- c(w.grid %*% post.grid)

  return(list(w = w, lpost.sum = lpost.sum))
}

ppFn <- function(w.knot, A, R, ld, lp, L, nknots, ngrid, a.kap){
  w.grid     <- matrix(NA, L, ngrid)
  lpost.grid <- rep(NA, ngrid)
  for(i in 1:ngrid){
    r             <- sum.sq(backsolve(R[[i]], w.knot, transpose = TRUE))
    w.grid[,i]    <- A[[i]] %*% w.knot
    lpost.grid[i] <- logsum(-(nknots/2+a.kap[1,])*log1p(0.5*r/ a.kap[2,]) + a.kap[3,] + lgamma(a.kap[1,]+nknots/2)-lgamma(a.kap[1,])-.5*nknots*log(a.kap[2,])) - ld[i] + lp[i]
  }

  lpost.sum <- logsum(lpost.grid)
  post.grid <- exp(lpost.grid - lpost.sum)

  w <- c(w.grid %*% post.grid)
  return(list(w = w, lpost.sum = lpost.sum))
}


estFn <- function(par, x, y, A, R, ld, lp,
                  L, mid, nknots, ngrid, a.kap, a.sig,
                  tau.g, reg.ix, reduce = TRUE,
                  x.center = 0, x.scale = 1){
  n     <- length(y)
  p     <- ncol(x)
  q     <- ncol(y)
  npar  <- (p+1) * (nknots+1) + 2

  wKnot <- array(as.matrix(par[1,grep("W", colnames(par))]), dim=c(nknots,p+1,q))
  w0    <- matrix(0, nrow=L, ncol=q)
  wMat  <- array(0, dim=c(L, p, q))
  vMat  <- array(0, dim=c(L, p, q))
  zeta0.dot <- matrix(0, nrow=L, ncol=q)
  zeta0  <- matrix(0, nrow=L, ncol=q)
  beta0.hat <- matrix(NA, nrow=L, ncol=q)
  beta.hat <- array(NA, dim=c(L,p,q))

  # For each response variable i:
  for(i in 1:q){
    # Interpolate values of w functions from values at knots
    w0[,i]    <- ppFn0(wKnot[,1,i], A, R, ld, lp, L, nknots, ngrid)$w
    wPP       <- apply(wKnot[,-1,i], 2, ppFn, A, R, ld, lp, L, nknots, ngrid, a.kap)
    wMat[,,i] <- matrix(sapply(wPP, extract, vn = "w"), ncol = p)

    # Calculate zeta function based on values of w0
    zeta0.dot[,i]    <- exp(w0[,i] - max(w0[,i]))
    zeta0[-c(1,L),i] <- trape(zeta0.dot[-c(1,L),i], tau.g[-c(1,L)])
    zeta0.tot        <- zeta0[L-1,i]
    zeta0[2:(L-1),i] <- tau.g[2] + (tau.g[L-1]-tau.g[2])*zeta0[2:(L-1),i] / zeta0.tot
    zeta0[L,i]       <- 1
    zeta0.dot[,i]    <- (tau.g[L-1]-tau.g[2])*zeta0.dot[,i] / zeta0.tot
    zeta0.dot[c(1,L),i] <- 0

    # Calculate values of w_j(zeta(tau)) functions (j=1,...,p) via interpolation
    zeta0.ticks <- pmin(L-1,
                        pmax(1,
                             sapply(zeta0[,i], function(u) sum(tau.g <= u))))

    zeta0.dists <- (zeta0[,i] - tau.g[zeta0.ticks]) / (tau.g[zeta0.ticks+1] - tau.g[zeta0.ticks])
    vMat[,,i]   <- apply(wMat[,,i], 2, transform.grid, ticks = zeta0.ticks, dists = zeta0.dists)

    # Read in other parameters
    reach <- (i-1)*npar + (p+1)*nknots
    gam0  <- unlist(par[reach + 1])
    gam   <- unlist(par[reach + 2:(p+1)])
    sigma <- unlist(sigFn(par[reach + p + 2], a.sig))
    nu    <- unlist(nuFn(par[reach + p + 3]))

    # Calculate estimated beta0 function
    b0dot <- sigma * q0(zeta0[,i], nu) * zeta0.dot[,i]
    beta0.hat[mid:L, i] <- gam0 + trape(b0dot[mid:L], tau.g[mid:L], L - mid + 1)
    beta0.hat[mid:1, i] <- gam0 + trape(b0dot[mid:1], tau.g[mid:1], mid)

    vNorm  <- sqrt(rowSums(vMat^2))
    a      <- tcrossprod(vMat[,,i], x)
    aX     <- apply(-a, 1, max)/vNorm
    aX[is.nan(aX)] <- Inf
    aTilde <- vMat[,,i] / (aX * sqrt(1 + vNorm^2))
    ab0    <- b0dot * aTilde

    beta.hat[,,i] <- kronecker(rep(1,L), t(gam))

    beta.hat[mid:L,,i] <- beta.hat[mid:L,,i] +
      apply(ab0[mid:L,,drop=FALSE], 2, trape, h = tau.g[mid:L], len = L - mid + 1)
    beta.hat[mid:1,,i] <- beta.hat[mid:1,,i] +
      apply(ab0[mid:1,,drop=FALSE], 2, trape, h = tau.g[mid:1], len = mid)

    # Rescale functions to account for scaling & centering
    beta.hat[,,i] <- beta.hat[,,i] / x.scale
    beta0.hat[,i] <- beta0.hat[,i] - rowSums(beta.hat[,,i] * x.center)
  }

  betas <- array(0, dim=c(L, p+1, q))
  betas[,1,]  <- beta0.hat
  betas[,-1,] <- beta.hat

  if(reduce) betas <- betas[reg.ix,,]
  return(betas)
}


coef.corrQR <- function(object, burn.perc = 0.5, nmc = 200,
                        plotBeta = TRUE, reduce = TRUE,
                        co="blue", alp=0.3, nr=4, nc=4){
  # Dimension parameters
  n      <- object$dim[1]
  p      <- object$dim[2]
  q      <- object$dim[3]
  L      <- object$dim[4]
  mid    <- object$dim[5] + 1
  nknots <- object$dim[6]
  ngrid  <- object$dim[7]
  niter  <- object$dim[9]
  nsamp  <- object$dim[11]
  pars   <- cbind(1:niter, object$parsamp)
  colnames(pars)[1] <- "Iter"
  ss <- unique(round(nsamp * seq(burn.perc, 1, len = nmc + 1)[-1]))

  # Hyperparameters and data parameters
  a.sig    <- object$hyper[1:2]
  a.kap    <- matrix(object$hyper[-c(1:2)], nrow = 3)
  tau.g    <- object$tau.g
  reg.ix   <- object$reg.ix
  x.center <- outer(rep(1, L), attr(object$x, "scaled:center"))
  x.scale  <- outer(rep(1, L), attr(object$x, "scaled:scale"))

  beta.samp <- array(NA, dim=c(length(reg.ix),p+1,q,length(ss)))
  for(i in 1:length(ss)){
    beta.samp[,,,i] <- estFn(pars[ss[i],-1],
                             object$x, object$y,
                             object$A, object$R, object$log.det, object$lp.grid,
                             L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix,
                             reduce, x.center, x.scale)
  }

  if(reduce){
    tau.g <- tau.g[reg.ix]
    L     <- length(tau.g)
  }

  b.median <- list()
  b.low    <- list()
  b.high   <- list()
  for(i in 1:q){
    b.median[[i]] <- as.data.frame(cbind(tau.g, apply(beta.samp[,,i,], c(1,2), quantile, 0.5)))
    b.low[[i]]    <- as.data.frame(cbind(tau.g, apply(beta.samp[,,i,], c(1,2), quantile, 0.025)))
    b.high[[i]]   <- as.data.frame(cbind(tau.g, apply(beta.samp[,,i,], c(1,2), quantile, 0.975)))

    colnames(b.median[[i]]) <- c("Tau", "Intercept", object$xnames)
    colnames(b.low[[i]])    <- c("Tau", "Intercept", object$xnames)
    colnames(b.high[[i]])   <- c("Tau", "Intercept", object$xnames)
  }
  names(b.median) <- object$ynames
  names(b.low) <- object$ynames
  names(b.high) <- object$ynames

  beta.est <- list(b.median, b.low, b.high)
  names(beta.est) <- c("Median", "Low", "High")

  if(plotBeta == TRUE){
    for(i in 1:q){
      pl <- lapply(2:(p+2), function(.x)
        ggplot(b.median[[i]][-c(1,L),], aes(x=Tau)) +
          geom_line(aes(y=b.median[[i]][-c(1,L),.x])) +
          geom_ribbon(aes(ymin=b.low[[i]][-c(1,L),.x],
                          ymax=b.high[[i]][-c(1,L),.x]),
                      fill=co, alpha = alp) +
          theme_bw() +
          labs(x=expression(tau),
               y="Coefficient",
               title=colnames(b.median[[i]])[.x])
        # xlab(expression(tau)) +
        # ylab(colnames(b.median[[i]])[.x])
      )
      ml <- marrangeGrob(pl,
                         nrow=nr, ncol=nc,
                         top=object$ynames[i])
      print(ml)
    }
  }

  invisible(list(beta.samp=beta.samp, beta.est=beta.est))
}

summary.corrQR <- function(object, ntrace = 1000, plot.dev = TRUE, more.details = FALSE){
  # Extract necessary fields from output object
  dimpars <- object$dim
  n       <- dimpars[1]
  p       <- dimpars[2]
  q       <- dimpars[3]
  ngrid   <- dimpars[7]
  thin    <- dimpars[10]
  nsamp   <- dimpars[11]
  pars    <- as.matrix(object$parsamp)
  ss      <- unique(pmax(1, round(nsamp * (1:ntrace/ntrace)))) # Index for which draws to use for trace plot
  ndraws  <- length(ss)
  dimpars[9] <- ndraws

  # Deviance calculation
  sm <- .Call('corrQR_devianceCalc', PACKAGE='corrQR',
              pars[ss,],
              object$x, object$y,
              object$hyper, dimpars,
              object$A, object$R,
              object$log.det, object$lp.grid, object$tau.g)

  deviance  <- sm$devsamp
  ll        <- as.matrix(sm$llsamp)
  fit.waic  <- waic(ll)
  # pg        <- sm$pgsamp
  pg <- array(unlist(sm$pgsamp),
              dim=c(dim(sm$pgsamp[[1]])[1],
                    dim(sm$pgsamp[[1]])[2],
                    dim(sm$pgsamp[[1]])[3],
                    length(sm$pgsamp)))

  prox.samp <- array(NA, dim=c(ndraws, p+1, q))
  for(j in 1:q){
    for(i in 1:(p+1)){
      prox.samp[,i,j] <- object$prox[apply(pg[,i,j,], 2, function(pr) sample(length(pr), 1, prob = pr))]
    }
  }

  if(plot.dev == TRUE){
    devplot <- ggplot(data=data.frame(X=thin*ss, Deviance=deviance), aes(x=X, y=Deviance)) +
      geom_line() + labs(x=("Markov chain iteration"), title="Fit trace plot") + theme_bw()

    print(devplot)
  }
  # if(more.details) par(mfrow = c(2,2), mar = c(5,4,3,2)+.1)

  if(more.details){
    ngrid         <- length(object$prox)
    prior.grid    <- exp(object$lp.grid)
    lam.priorsamp <- lamFn(sample(object$prox, ntrace, replace = TRUE, prob = prior.grid))
    lam.prior.q   <- quantile(lam.priorsamp, pr = c(.025, .5, .975))

    lam.samp <- array(NA, dim=c(nsamp, p+1, q))

    for(i in 1:q){
      lam.samp[,,i] <- lamFn(prox.samp[,,i])
    }

    a <- min(lamFn(object$prox))
    b <- diff(range(lamFn(object$prox))) * 1.2

    for(j in 1:q){
      plot(thin * ss, lam.samp[,1,1], ty = "n", ylim = a + c(0, b * (p + 1)), bty = "n", ann = FALSE, axes = FALSE)
      axis(1)
      for(i in 1:(p+1)){
        abline(h = b * (i-1) + lamFn(object$prox), col = "gray")
        abline(h = b * (i - 1) + lam.prior.q, col = "red", lty = c(2,1,2))
        lines(thin * ss, b * (i-1) + lam.samp[,i,j], lwd = 1, col = 4)
        if(i %% 2) axis(2, at = b * (i-1) + lamFn(object$prox[c(1,ngrid)]), labels = round(object$prox[c(1,ngrid)],2), las = 1, cex.axis = 0.6)
        mtext(substitute(beta[index], list(index = i - 1)), side = 4, line = 0.5, at = a + b * (i - 1) + 0.4*b, las = 1)
      }
      title(xlab = "Markov chain iteration",
            ylab = "Proxmity posterior",
            main = paste("Mixing over GP scaling - Response", j))
    }

    #   plot(thin * ss, lam.samp[1,], ty = "n", ylim = a + c(0, b * (p + 1)), bty = "n", ann = FALSE, axes = FALSE)
    #   axis(1)
    #   for(i in 1:(p+1)){
    #     abline(h = b * (i-1) + lamFn(object$prox), col = "gray")
    #     abline(h = b * (i - 1) + lam.prior.q, col = "red", lty = c(2,1,2))
    #     lines(thin * ss, b * (i-1) + lam.samp[i,], lwd = 1, col = 4)
    #     if(i %% 2) axis(2, at = b * (i-1) + lamFn(object$prox[c(1,ngrid)]), labels = round(object$prox[c(1,ngrid)],2), las = 1, cex.axis = 0.6)
    #     mtext(substitute(beta[index], list(index = i - 1)), side = 4, line = 0.5, at = a + b * (i - 1) + 0.4*b, las = 1)
    #   }
    #   title(xlab = "Markov chain iteration", ylab = "Proxmity posterior", main = "Mixing over GP scaling")
    #
    #   theta <- as.mcmc(t(matrix(object$parsamp, ncol = nsamp)[,unique(ceiling(seq(0.5*nsamp,nsamp,1)))]))
    #   gg <- geweke.diag(theta, .1, .5)
    #   zvals <- gg$z
    #
    #   pp <- 2 * (1 - pnorm(abs(zvals)))
    #   plot(sort(pp), ylab = "Geweke p-values", xlab = "Parameter index (reordered)", main = "Convergence diagnosis", ty = "h", col = 4, ylim = c(0, 0.3), lwd = 2)
    #   abline(h = 0.05, col = 2, lty = 2)
    #   abline(a = 0, b = 0.1 / length(pp), col = 2, lty = 2)
    #   mtext(c("BH-10%", "5%"), side = 4, at = c(0.1, 0.05), line = 0.1, las = 0, cex = 0.6)
    #
    #   npar <- length(object$par)
    #   image(1:npar, 1:npar, cor(theta), xlab = "Parameter index", ylab = "Parameter index", main = "Parameter correlation")
    #
    # }
  }
  invisible(list(deviance = sm$devsamp, ll = sm$llsamp, pgsamp = pg, prox = prox.samp, waic = fit.waic))
}
