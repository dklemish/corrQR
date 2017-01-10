corrQR <- function(x, y, nsamp = 1e3, thin = 10,
                   incr = 0.01, initPar = "prior",
                   copula = "gaussian",
                   nknots = 6,
                   hyper = list(sig = c(.1,.1),
                                lam = c(6,4),
                                kap = c(0.1,0.1,1)),
                   shrink = FALSE,
                   prox.range = c(.2,.95),
                   acpt.target = 0.15,
                   ref.size = 3,
                   blocking = "by.response",
                   temp = 1, expo = 2,
                   blocks.mu, blocks.S, fix.nu = FALSE){
  # Input parameters:
  #  x - Predictors (n by p matrix)
  #  y - Response (n by q matrix)
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

  ## Read in data
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

  # Determine the grid of quantiles of interest as multiples of 0.01
  # plus additional points in the tail based on the # of data points

  Ltail <- ceiling(log(n*incr,2)) # of points in tail

  # tau.g = grid of quantiles
  # L = # of quantiles
  # mid = array position of quantile closest to 0.5
  # reg.ix = indices of quantiles not in the tail
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

  # tau.k = reduced grid of quantiles at which functions will be tracked
  tau.k <- seq(0,1,len = nknots)

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

  # prox.grid = points approximating possible values of lambda
  #  (scale parameter in GPs)
  # prior.grid = discrete approximation to value of prior pdf for
  #  possible values of lambda as stored in prox.grid
  # lp.grid = approximate log of prior distribution for lambda
  prox.grid  <- proxFn(max(prox.range), min(prox.range), 0.5)
  ngrid      <- length(prox.grid)
  lamsq.grid <- lamFn(prox.grid)^2
  prior.grid <- -diff(pbeta(c(1,
                              (prox.grid[-1] + prox.grid[-ngrid])/2,
                              0),
                            a.lam[1],
                            a.lam[2]))
  lp.grid <- log(prior.grid)

  # d.kg = (tau_i - tau'_j)^2 for tau_i \in tau.k (reduced quantile grid),
  #   tau_j \in tau.g (full quantile grid)
  d.kg <- abs(outer(tau.k, tau.g, "-"))^expo

  # d.kk = (tau_i - tau'_j)^2 for tau_i \in tau.k (reduced quantile grid),
  #   tau_j \in tau.k
  d.kk <- abs(outer(tau.k, tau.k, "-"))^expo

  # gridmats = GP parameters to pass to MCMC
  #  ith column contains A_g, R_g, the log determinant of R_g, and the
  #  log prior density for that value of lambda
  gridmats <- matrix(NA, nknots*(L + nknots)+2, ngrid)

  K0 <- 0
  t1 <- Sys.time()
  for(i in 1:ngrid){
    # K.grid = covariances between elements in tau.k and tau.g
    # K.knot = covariances between elements in tau.k
    K.grid <- exp(-lamsq.grid[i] * d.kg)
    K.knot <- exp(-lamsq.grid[i] * d.kk)
    diag(K.knot) <- 1 + 1e-10   # for numerical stability

    # R.knot, A.knot = R_g, A_g from paper
    R.knot <- chol(K.knot)
    A.knot <- solve(K.knot, K.grid)

    gridmats[,i] <- c(c(A.knot), c(R.knot), sum(log(diag(R.knot))), lp.grid[i])

    K0 <- K0 + prior.grid[i] * K.knot
  }
  t2 <- Sys.time()
  cat("Matrix calculation time per 1e3 iterations =", round(1e3 * as.numeric(t2 - t1), 2), "\n")

  niter <- nsamp * thin
  dimpars <- c(n, p, q, L, mid - 1, nknots, ngrid, ncol(a.kap), niter, thin, nsamp)

  # Number of parameters for each response variable
  npar <- (nknots+1) * (p+1) + 2

  # Measure of influence used to better initialize sigma before
  # starting MCMC
  infl <- rep(0, q)

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
  #    q*npar_j + 1 = 1 parameter copula parameter

  if(initPar[1] == "prior") {
    par <- rep(0, q*npar + q*(q-1)/2)
    if(fix.nu) par[npar * 1:q] <- nuFn.inv(fix.nu)

    for(j in 1:q){
      # Initialize gamma parameters
      beta.rq <- sapply(tau.g, function(a) return(coef(rq(y[,j] ~ x, tau = a))))
      v <- bs(tau.g, df = 5)
      rq.lm <- apply(beta.rq, 1, function(z) return(coef(lm(z ~ v))))

      delta <- tau.g[2]
      tau.0 <- tau.g[mid]
      rq.tau0 <- c(c(1, predict(v, tau.0)) %*% rq.lm)
      rq.delta <- c(c(1, predict(v, delta)) %*% rq.lm)
      rq.deltac <- c(c(1, predict(v, 1 - delta)) %*% rq.lm)

      par[(j-1)*npar + nknots*(p+1) + 1:(p+1)] <- as.numeric(rq.tau0)

      # Initialize transformed value of sigma
      sigma <- 1
      par[(j-1)*npar + (nknots+1)*(p+1) + 1] <- sigFn.inv(sigma, a.sig)

      # Initialize values of w_0 to w_p functions
      for(i in 1:(p+1)){
        kapsq <- sum(exp(a.kap[3,]) * (a.kap[2,] / a.kap[1,]))
        lam.ix <- sample(length(lamsq.grid), 1, prob = prior.grid)
        R <- matrix(gridmats[L*nknots + 1:(nknots*nknots),lam.ix], nknots, nknots)
        z <- sqrt(kapsq) * c(crossprod(R, rnorm(nknots)))
        par[(j-1)*npar + (i - 1) * nknots + 1:nknots] <- z - mean(z)
      }

      beta.hat <- estFn(par, x, y[,j], gridmats, L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix, FALSE)
      qhat <- tcrossprod(cbind(1, x), beta.hat)

      infl[j] <- max(max((y[,j] - qhat[,mid])/(qhat[,ncol(qhat) - 1] - qhat[,mid])),
                     max((qhat[,mid] - y[,j])/(qhat[,mid] - qhat[,2])))
    }
  } else if (initPar[1] == "RQ"){

    # par contains MCMC parameters to track
    #  - npar paramters for each response
    #  - q*(q-1)/2 correlation parameters for Gaussian copula
    par <- rep(0, q*npar + q*(q-1)/2)

    for(j in 1:q){
      # Initialize gamma parameters
      beta.rq <- sapply(tau.g, function(a) return(coef(rq(y[,j] ~ x, tau = a))))
      v <- bs(tau.g, df = 5)
      rq.lm <- apply(beta.rq, 1, function(z) return(coef(lm(z ~ v))))

      delta <- tau.g[2]
      tau.0 <- tau.g[mid]
      rq.tau0 <- c(c(1, predict(v, tau.0)) %*% rq.lm)
      rq.delta <- c(c(1, predict(v, delta)) %*% rq.lm)
      rq.deltac <- c(c(1, predict(v, 1 - delta)) %*% rq.lm)

      par[(j-1)*npar + nknots*(p+1) + 1:(p+1)] <- as.numeric(rq.tau0)

      # Initialize sigma
      nu <- ifelse(fix.nu, fix.nu, nuFn(0))

      sigma <- min((rq.delta[1] - rq.tau0[1]) / Q0(delta, nu), (rq.deltac[1] - rq.tau0[1]) / Q0(1 - delta, nu))
      par[(j-1)*npar + (nknots+1)*(p+1) + 1]  <- sigFn.inv(sigma, a.sig)

      epsilon <- 0.1 * min(diff(sort(tau.k)))
      tau.knot.plus <- pmin(tau.k + epsilon, 1)
      tau.knot.minus <- pmax(tau.k - epsilon, 0)
      beta.rq.plus <- cbind(1, predict(v, tau.knot.plus)) %*% rq.lm
      beta.rq.minus <- cbind(1, predict(v, tau.knot.minus)) %*% rq.lm
      zeta0.plus <- F0((beta.rq.plus[,1] - rq.tau0[1]) / sigma, nu)
      zeta0.minus <- F0((beta.rq.minus[,1] - rq.tau0[1]) / sigma, nu)
      zeta0.dot.knot <- (zeta0.plus - zeta0.minus) / (tau.knot.plus - tau.knot.minus)
      w0.knot <- log(pmax(epsilon, zeta0.dot.knot)) / shrinkFn(p)
      w0.knot <- (w0.knot - mean(w0.knot))

      w0PP <- ppFn0(w0.knot, gridmats, L, nknots, ngrid)
      w0 <- w0PP$w

      zeta0.dot <- exp(shrinkFn(p) * (w0 - max(w0)))
      zeta0 <- trape(zeta0.dot[-c(1,L)], tau.g[-c(1,L)], L-2)
      zeta0.tot <- zeta0[L-2]
      zeta0 <- c(0, tau.g[2] + (tau.g[L-1]-tau.g[2])*zeta0 / zeta0.tot, 1)
      zeta0.dot <- (tau.g[L-1]-tau.g[2])*zeta0.dot / zeta0.tot
      zeta0.dot[c(1,L)] <- 0

      # Initialize w0_j
      par[(j-1)*npar_j + 1:nknots] <- w0.knot

      beta0.dot <- sigma * q0(zeta0, nu) * zeta0.dot

      tilt.knot <- tau.g[tilt.ix <- sapply(tau.k, function(a) which.min(abs(a - zeta0)))]
      tau.knot.plus  <- pmin(tilt.knot + epsilon, 1)
      tau.knot.minus <- pmax(tilt.knot - epsilon, 0)
      beta.rq.plus  <- cbind(1, predict(v, tau.knot.plus)) %*% rq.lm
      beta.rq.minus <- cbind(1, predict(v, tau.knot.minus)) %*% rq.lm

      beta.dot.knot <- (beta.rq.plus[,-1,drop=FALSE] - beta.rq.minus[,-1,drop=FALSE]) /  (tau.knot.plus - tau.knot.minus)

      # Initialize w(1:p)_j
      par[(j-1)*npar + nknots + 1:(nknots*p)] <- c(beta.dot.knot)
      beta.hat <- estFn(par, x, y[,j], gridmats, L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix, FALSE)
      qhat <- tcrossprod(cbind(1, x), beta.hat)
      infl[j] <- max(max((y[,j] - qhat[,mid])/(qhat[,ncol(qhat) - 1] - qhat[,mid])),
                     max((qhat[,mid] - y[,j])/(qhat[,mid] - qhat[,2])))
    }
  }

  # Build list that contains which indices of parameter vector
  # should be updated in each block update in each MCMC iteration
  if(blocking == "single"){
    blocks <- list(rep(TRUE, npar*q + 1))
  } else if(blocking == "by.response"){
    blocks <- replicate(q*(p + 3) + 1, rep(FALSE, npar*q + 1), simplify = FALSE)

    for(j in 1:q){
      # Parameters for w & gamma functions for one response variable in separate blocks
      for(i in 0:p){
        blocks[[(j-1)*(p+1) + i + 1]][c((j-1)*npar + (i * nknots + 1:nknots),
                                        (j-1)*npar + nknots*(p+1) + i + 1)] <- TRUE
      }

      # Parameters for gamma functions for one response variable in separate blocks
      blocks[[q*(p+1) + j]][(j-1)*npar + nknots*(p+1) + 1:(p+1)] <- TRUE

      # Parameters for sigma & nu for one response variable & correlation coefficient
      blocks[[q*(p+2) + j]][c((j-1)*npar + (nknots+1)*(p+1) + 1:2, q*npar+1)] <- TRUE
    }

    # All parameters
    blocks[[q*(p+3)+1]][1:(q*npar + 1)] <- TRUE
  } else if(blocking == "by.function"){
    blocks <- replicate(p + 4, rep(FALSE, npar*q + 1), simplify = FALSE)

    # Parameters for w & gamma functions for all response variables
    for(i in 0:p){
      blocks[[i + 1]][c(sapply((1:q - 1)*npar, function(x) x + i*nknots + 1:nknots),
                        sapply((1:q - 1)*npar, function(x) x + nknots*(p+1) + i + 1))] <- TRUE
    }

    # Parameters for gamma functions for all response variables
    blocks[[p+2]][c(sapply((1:q - 1)*npar, function(x) x + nknots*(p+1) + 1:(p+1)))] <- TRUE

    # Parameters for sigma & nu for all response variables & correlation coefficient
    blocks[[p+3]][c(sapply((1:q - 1)*npar, function(x) x + (nknots+1)*(p+1) + 1:2),
                    q*npar + 1)] <- TRUE

    # All parameters
    blocks[[p+4]][1:(q*npar + 1)] <- TRUE
  } else {
    blocks <- replicate(q*npar+1, rep(FALSE, q*npar + 1), simplify = FALSE)
    for(i in 1:(q*npar + 1)) blocks[[i]][i] <- TRUE
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

  if(missing(blocks.mu)) blocks.mu <- rep(0, sum(blocks.size))

  if(missing(blocks.S)){
    blocks.S <- lapply(blocks.size, function(q) diag(1, q))
    # if(substr(blocking, 1, 2) == "by"){
    #   for(i in 1:(p+1)) blocks.S[[i]][1:nknots, 1:nknots] <- K0
    #   if(as.numeric(substr(blocking, 4,5)) > 1){
    #     blocks.S[[p + 2]] <- summary(rq(y ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov
    #     blocks.S[[p + 3]] <- matrix(c(1, 0, 0, .1), 2, 2)
    #   }
    #   if(as.numeric(substr(blocking, 4,5)) == 5){
    #     slist <- list(); length(slist) <- p + 3
    #     for(i in 1:(p+1)) slist[[i]] <- K0
    #     slist[[p+2]] <- summary(rq(y ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov
    #     slist[[p+3]] <- matrix(c(1, 0, 0, .1), 2, 2)
    #     blocks.S[[p + 4]] <- as.matrix(bdiag(slist))
    #   }
    # }
    if(blocking == "by.response"){
      for(i in 1:(q*(p+1))){
        blocks.S[[i]][1:nknots, 1:nknots] <- K0
      }
      for(i in 1:q){
        blocks.S[[q*(p+1) + i]] <- summary(rq(y[,i] ~ x, tau = 0.5),
                                           se = "boot",
                                           cov = TRUE)$cov
      }
      for(i in 1:q){
        blocks.S[[q*(p+2) + i]] <- matrix(c(1,0,0,
                                            0,0.1,0,
                                            0,0,0.1), nrow=3, ncol=3)
      }

      # Determine prior joint covariance matrix of all blocks
      slist <- list()
      length(slist) <- q*(p+3) + 1
      for(i in 1:q){
        for(j in 1:(p+1)){
          slist[[(i-1)*(p+3) + j]] <- K0
        }

        slist[[(i-1)*(p+3) + p + 2]] <- summary(rq(y[,i] ~ x, tau = 0.5),
                                                se = "boot",
                                                cov = TRUE)$cov
        slist[[i*(p+3)]] <- matrix(c(1,0,0,0.1), nrow=2, ncol=2)
      }
      slist[[q*(p+3) + 1]] <- 0.1

      blocks.S[[q*(p+3) + 1]] <- as.matrix(bdiag(slist))
    } else if(blocking == "by.function"){
      for(j in 1:q){
        for(i in 1:(p+1)){
          blocks.S[[i]][((j-1)*(nknots+1)+1):((j-1)*(nknots+1)+nknots),
                        ((j-1)*(nknots+1)+1):((j-1)*(nknots+1)+nknots)] <- K0
        }

        blocks.S[[p+2]][((j-1)*(p+1) + 1):(j*(p+1)),
                        ((j-1)*(p+1) + 1):(j*(p+1))] <-
          summary(rq(y[,j] ~ x, tau = 0.5), se = "boot", cov = TRUE)$cov

        blocks.S[[p+3]][(j-1)*2 + 1:2, (j-1)*2 + 1:2] <-
          matrix(c(1,0,0,0.1), nrow=2, ncol=2)

      }
      blocks.S[[p+3]][q*2 + 1, q*2 + 1] <- 0.1

      # Determine prior joint covariance matrix of all variables
      slist <- list()
      length(slist) <- q*(p+3) + 1
      for(i in 1:q){
        for(j in 1:(p+1)){
          slist[[(i-1)*(p+3) + j]] <- K0
        }

        slist[[(i-1)*(p+3) + p + 2]] <- summary(rq(y[,i] ~ x, tau = 0.5),
                                                se = "boot",
                                                cov = TRUE)$cov
        slist[[i*(p+3)]] <- matrix(c(1,0,0,0.1), nrow=2, ncol=2)
      }
      slist[[q*(p+3) + 1]] <- 0.1

      blocks.S[[p+4]] <- as.matrix(bdiag(slist))
    }

    blocks.S <- unlist(blocks.S)
  }

  imcmc.par <- c(nblocks, ref.size, TRUE, max(10, niter/1e4), rep(0, nblocks))
  dmcmc.par <- c(temp, 0.999, rep(acpt.target, nblocks), 2.38 / sqrt(blocks.size))

  # cat("par = ", par)
  # #cat("x = ", as.double(x))
  # #cat("y = ", as.double(y))
  # cat("copula = ", copulaMethod)
  # cat("hyper = ", hyperPar)
  # cat("dim = ", dim)
  # cat("gridmats = ", gridmats)
  # cat("tau.g = ", tau.g)
  # #cat("muV = ", blocks.mu)
  # #cat("SV = ", blocks.S)
  # #cat("blocks.size = ", blocks.size)
  #print(par)
  cat("length(par) =", length(par), "\n")
  cat("dim(x) =", dim(x), "\n")
  cat("dim(y) =", dim(y), "\n")
  cat("copulaMethod =", copulaMethod, "\n")
  cat("length(hyperPar) =", length(hyperPar), "\n")
  #print(dimpars)
  cat("dim(gridmats) =", dim(gridmats), "\n")
  #print(tau.g)
  cat("length(blocks.mu) =", length(blocks.mu), "\n")
  cat("length(blocks.S) =", length(blocks.S), "\n")
  print(blocks.size)

  tm.c <- system.time(
    oo <- .C("BJQR", par = as.double(par),
             x = as.double(x), y = as.double(y),
             copula = as.integer(copulaMethod),
             shrink = as.integer(shrink), hyper = as.double(hyperPar),
             dim = as.integer(dimpars), gridmats = as.double(gridmats),
             tau.g = as.double(tau.g),
             siglim = as.double(sigFn.inv(c(1.0 * infl * sigma, 10.0 * infl * sigma), a.sig)),
             muV = as.double(blocks.mu), SV = as.double(blocks.S),
             blocks = as.integer(blocks.ix), blocks.size = as.integer(blocks.size),
             dmcmcpar = as.double(dmcmc.par), imcmcpar = as.integer(imcmc.par),
             parsamp = double(nsamp * length(par)),
             acptsamp = double(nsamp * nblocks), lpsamp = double(nsamp))
  )
  cat("elapsed time:", round(tm.c[3]), "seconds\n")

  oo$x <- x
  oo$y <- y
  oo$xnames <- x.names
  oo$ynames <- y.names
  oo$gridmats <- gridmats
  oo$prox <- prox.grid
  oo$reg.ix <- reg.ix
  oo$runtime <- tm.c[3]

  class(oo) <- "qrjoint"
  return(oo)
}

estFn <- function(par, x, y, gridmats, L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix, reduce = TRUE, x.ce = 0, x.sc = 1){

  n <- length(y)
  p <- ncol(x)

  wKnot <- matrix(par[1:(nknots*(p+1))], nrow = nknots)
  w0PP  <- ppFn0(wKnot[,1], gridmats, L, nknots, ngrid)
  w0    <- w0PP$w

  wPP   <- apply(wKnot[,-1,drop=FALSE], 2, ppFn, gridmats = gridmats, L = L, nknots = nknots, ngrid = ngrid, a.kap = a.kap)

  vMat <- matrix(sapply(wPP, extract, vn = "w"), ncol = p)

  zeta0.dot <- exp(shrinkFn(p) * (w0 - max(w0)))
  zeta0 <- trape(zeta0.dot[-c(1,L)], tau.g[-c(1,L)], L-2)
  zeta0.tot <- zeta0[L-2]
  zeta0 <- c(0, tau.g[2] + (tau.g[L-1]-tau.g[2])*zeta0 / zeta0.tot, 1)
  zeta0.dot <- (tau.g[L-1]-tau.g[2])*zeta0.dot / zeta0.tot
  zeta0.dot[c(1,L)] <- 0

  reach <- nknots*(p+1)
  gam0 <- par[reach + 1]; reach <- reach + 1
  gam <- par[reach + 1:p]; reach <- reach + p
  sigma <- sigFn(par[reach + 1], a.sig); reach <- reach + 1
  nu <- nuFn(par[reach + 1]);

  b0dot <- sigma * q0(zeta0, nu) * zeta0.dot
  beta0.hat <- rep(NA, L)
  beta0.hat[mid:L] <- gam0 + trape(b0dot[mid:L], tau.g[mid:L], L - mid + 1)
  beta0.hat[mid:1] <- gam0 + trape(b0dot[mid:1], tau.g[mid:1], mid)

  vNorm <- sqrt(rowSums(vMat^2))
  a <- tcrossprod(vMat, x)
  aX <- apply(-a, 1, max)/vNorm
  aX[is.nan(aX)] <- Inf
  aTilde <- vMat / (aX * sqrt(1 + vNorm^2))
  ab0 <- b0dot * aTilde

  beta.hat <- kronecker(rep(1,L), t(gam))
  beta.hat[mid:L,] <- beta.hat[mid:L,] + apply(ab0[mid:L,,drop=FALSE], 2, trape, h = tau.g[mid:L], len = L - mid + 1)
  beta.hat[mid:1,] <- beta.hat[mid:1,] + apply(ab0[mid:1,,drop=FALSE], 2, trape, h = tau.g[mid:1], len = mid)
  beta.hat <- beta.hat / x.sc
  beta0.hat <- beta0.hat - rowSums(beta.hat * x.ce)
  betas <- cbind(beta0.hat, beta.hat)
  if(reduce) betas <- betas[reg.ix,,drop = FALSE]
  return(betas)
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

ppFn0 <- function(w.knot, gridmats, L, nknots, ngrid){
  w.grid <- matrix(NA, L, ngrid)
  lpost.grid <- rep(NA, ngrid)
  for(i in 1:ngrid){
    A <- matrix(gridmats[1:(L*nknots),i], nrow = nknots)
    R <- matrix(gridmats[L*nknots + 1:(nknots*nknots),i], nrow = nknots)
    r <- sum.sq(backsolve(R, w.knot, transpose = TRUE))
    w.grid[,i] <- colSums(A * w.knot)
    lpost.grid[i] <- -(0.5*nknots+1.5)*log1p(0.5*r/1.5) - gridmats[nknots*(L+nknots)+1,i] + gridmats[nknots*(L+nknots)+2,i]
  }
  lpost.sum <- logsum(lpost.grid)
  post.grid <- exp(lpost.grid - lpost.sum)
  w <- c(w.grid %*% post.grid)
  return(list(w = w, lpost.sum = lpost.sum))
}

ppFn <- function(w.knot, gridmats, L, nknots, ngrid, a.kap){
  w.grid <- matrix(NA, L, ngrid)
  lpost.grid <- rep(NA, ngrid)
  for(i in 1:ngrid){
    A <- matrix(gridmats[1:(L*nknots),i], nrow = nknots)
    R <- matrix(gridmats[L*nknots + 1:(nknots*nknots),i], nrow = nknots)
    r <- sum.sq(backsolve(R, w.knot, transpose = TRUE))
    w.grid[,i] <- colSums(A * w.knot)
    lpost.grid[i] <- (logsum(-(nknots/2+a.kap[1,])*log1p(0.5*r/ a.kap[2,]) + a.kap[3,] + lgamma(a.kap[1,]+nknots/2)-lgamma(a.kap[1,])-.5*nknots*log(a.kap[2,]))
                      - gridmats[nknots*(L+nknots)+1,i] + gridmats[nknots*(L+nknots)+2,i])
  }
  lpost.sum <- logsum(lpost.grid)
  post.grid <- exp(lpost.grid - lpost.sum)
  w <- c(w.grid %*% post.grid)
  return(list(w = w, lpost.sum = lpost.sum))
}

lamFn <- function(prox) return(sqrt(-100*log(prox)))
nuFn <- function(z) return(0.5 + 5.5*exp(z/2))
nuFn.inv <- function(nu) return(2*log((nu - 0.5)/5.5))
sigFn <- function(z, a.sig) return(exp(z/2))
sigFn.inv <- function(s, a.sig) return(2 * log(s))
unitFn <- function(u) return(pmin(1 - 1e-10, pmax(1e-10, u)))

q0 <- function(u, nu = Inf) return(1 / (dt(qt(unitFn(u), df = nu), df = nu) * qt(.9, df = nu)))
Q0 <- function(u, nu = Inf) return(qt(unitFn(u), df = nu) / qt(.9, df = nu))
F0 <- function(x, nu = Inf) return(pt(x*qt(.9, df = nu), df = nu))

sum.sq <- function(x) return(sum(x^2))
extract <- function(lo, vn) return(lo[[vn]])
logmean <- function(lx) return(max(lx) + log(mean(exp(lx - max(lx)))))
logsum <- function(lx) return(logmean(lx) + log(length(lx)))
shrinkFn <- function(x) return(1) ##(1/(1 + log(x)))
trape <- function(x, h, len = length(x)) return(c(0, cumsum(.5 * (x[-1] + x[-len]) * (h[-1] - h[-len]))))

klGP <- function(lam1, lam2, nknots = 11){
  tau <- seq(0, 1, len = nknots)
  dd <- outer(tau, tau, "-")^2
  K1 <- exp(-lam1^2 * dd); diag(K1) <- 1 + 1e-10; R1 <- chol(K1); log.detR1 <- sum(log(diag(R1)))
  K2 <- exp(-lam2^2 * dd); diag(K2) <- 1 + 1e-10; R2 <- chol(K2); log.detR2 <- sum(log(diag(R2)))
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
