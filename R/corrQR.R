corrQR <- function(x, y, sd, Rcorr,
                   nsamp = 1e3, thin = 10,
                   incr = 0.01,
                   #initPar = "prior",
                   ip,
                   #copula = "gaussian",
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
                   fix.nu = FALSE, fix.corr = TRUE){
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

  par <- rep(0, tot.par)

  set.seed(sd)

  # Initialize w functions, gamma0, gamma, sigma & nu
  # if(initPar == "prior"){
  #   for(j in 1:q){
  #     suppressWarnings(
  #       invis <- capture.output(
  #         par[(j-1)*npar + 1:npar] <-
  #           qrjoint(x, y[,j], nsamp = 100, thin=1)$par
  #       )
  #     )
  #   }
  # }else if(initPar == "user"){
  par <- ip
  # }

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
  oo$prox     <- prox.grid
  oo$reg.ix   <- reg.ix
  oo$dim      <- dimpars
  oo$imcmcpar <- imcmc.par
  oo$dmcmcpar <- dmcmc.par
  oo$runtime  <- tm.cpp[3]

  samples <- as.data.frame(cbind(1:nsamp, oo$parsamp))
  parnames <- colnames(samples)

  parnames[1] <- "Iter"
  reach       <- 2

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

lamFn     <- function(prox) return(sqrt(-100*log(prox)))
nuFn      <- function(z) return(0.5 + 5.5*exp(z/2))
nuFn.inv  <- function(nu) return(2*log((nu - 0.5)/5.5))
sigFn     <- function(z, a.sig) return(exp(z/2))
sigFn.inv <- function(s, a.sig) return(2 * log(s))
unitFn    <- function(u) return(pmin(1 - 1e-10, pmax(1e-10, u)))
logmean   <- function(lx) return(max(lx) + log(mean(exp(lx - max(lx)))))
logsum    <- function(lx) return(logmean(lx) + log(length(lx)))
trape     <- function(x, h, len = length(x)) return(c(0, cumsum(.5 * (x[-1] + x[-len]) * (h[-1] - h[-len]))))

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

  pl <- lapply(2:(totpar+1), function(.x)
    ggplot(object$parsamp, aes(x=Iter, y=object$parsamp[,.x])) +
      geom_line() + theme_bw() +
      ylab(colnames(object$parsamp)[.x])
  )
  ml <- marrangeGrob(pl, nrow=4, ncol=4)
  ml
}
# coef.qrjoint <- function(object, burn.perc = 0.5, nmc = 200, plot = FALSE, show.intercept = TRUE, reduce = TRUE, ...){
#   niter <- object$dim[8]
#   nsamp <- object$dim[10]
#   pars <- matrix(object$parsamp, ncol = nsamp)
#   ss <- unique(round(nsamp * seq(burn.perc, 1, len = nmc + 1)[-1]))
#
#   n <- object$dim[1]
#   p <- object$dim[2]
#   L <- object$dim[3]
#   mid <- object$dim[4] + 1
#   nknots <- object$dim[5]
#   ngrid <- object$dim[6]
#
#   a.sig  <- object$hyper[1:2]
#   a.kap  <- matrix(object$hyper[-c(1:2)], nrow = 3)
#   tau.g  <- object$tau.g
#   reg.ix <- object$reg.ix
#   x.ce   <- outer(rep(1, L), attr(object$x, "scaled:center"))
#   x.sc   <- outer(rep(1,L), attr(object$x, "scaled:scale"))
#
#   base.bundle <- list()
#   if(object$fbase.choice == 1){
#     base.bundle$q0 <- function(u, nu = Inf) return(1 / (dt(qt(unitFn(u), df = nu), df = nu) * qt(.9, df = nu)))
#     base.bundle$Q0 <- function(u, nu = Inf) return(qt(unitFn(u), df = nu) / qt(.9, df = nu))
#     base.bundle$F0 <- function(x, nu = Inf) return(pt(x*qt(.9, df = nu), df = nu))
#   } else {
#     base.bundle$q0 <- function(u, nu = Inf) return(1 / (dunif(qunif(u, -1,1), -1,1)))
#     base.bundle$Q0 <- function(u, nu = Inf) return(qunif(u, -1,1))
#     base.bundle$F0 <- function(x, nu = Inf) return(punif(x, -1,1))
#   }
#
#   beta.samp <- apply(pars[,ss],
#                      2,
#                      function(p1)
#                        c(estFn(p1, object$x, object$y, object$gridmats, L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix, reduce, x.ce, x.sc, base.bundle)))
#
#   if(reduce) tau.g <- tau.g[reg.ix]
#   L <- length(tau.g)
#
#   if(plot){
#     nr <- ceiling(sqrt(p+show.intercept))
#     nc <- ceiling((p+show.intercept)/nr)
#     par(mfrow = c(nr, nc))
#   }
#
#   reach <- 0
#   beta.hat <- list()
#   plot.titles <- c("Intercept", object$xnames)
#   j <- 1
#   b <- beta.samp[reach + 1:L,]
#
#   beta.hat[[j]] <-
#     getBands(b, plot = (plot & show.intercept),
#              add = FALSE,
#              x = tau.g,
#              xlab = "tau",
#              ylab = "Coefficient", bty = "n", ...)
#
#   if(plot & show.intercept) title(main = plot.titles[j])
#
#   reach <- reach + L
#
#   for(j in 2:(p+1)){
#     b <- beta.samp[reach + 1:L,]
#     beta.hat[[j]] <-
#       getBands(b, plot = plot,
#                add = FALSE,
#                x = tau.g,
#                xlab = "tau",
#                ylab = "Coefficient", bty = "n", ...)
#     if(plot) {
#       title(main = plot.titles[j])
#       abline(h = 0, lty = 2, col = 4)
#     }
#
#     reach <- reach + L
#   }
#
#   names(beta.hat) <- plot.titles
#
#   invisible(list(beta.samp = beta.samp, beta.est = beta.hat))
# }

# getBands <- function(b, col = 2, lwd = 1,
#                      plot = TRUE, add = FALSE,
#                      x = seq(0,1,len=nrow(b)), remove.edges = TRUE, ...){
#
#   colRGB   <- col2rgb(col)/255
#   colTrans <- rgb(colRGB[1], colRGB[2], colRGB[3], alpha = 0.2)
#
#   b.med <- apply(b, 1, quantile, pr = .5)
#   b.lo <- apply(b, 1, quantile, pr = .025)
#   b.hi <- apply(b, 1, quantile, pr = 1 - .025)
#
#   L <- nrow(b)
#   ss <- 1:L; ss.rev <- L:1
#
#   if(remove.edges){
#     ss <- 2:(L-1); ss.rev <- (L-1):2
#   }
#   if(plot){
#     if(!add)
#       plot(x[ss], b.med[ss], ty = "n", ylim = range(c(b.lo[ss], b.hi[ss])), ...)
#
#     polygon(x[c(ss, ss.rev)], c(b.lo[ss], b.hi[ss.rev]), col = colTrans, border = colTrans)
#     lines(x[ss], b.med[ss], col = col, lwd = lwd)
#   }
#   invisible(cbind(b.lo, b.med, b.hi))
# }

# estFn <- function(par, x, y, gridmats, L, mid, nknots, ngrid, a.kap, a.sig, tau.g, reg.ix, reduce = TRUE, x.ce = 0, x.sc = 1, base.bundle){
#
#   n <- length(y)
#   p <- ncol(x)
#   wKnot <- matrix(par[1:(nknots*(p+1))], nrow = nknots)
#   w0PP  <- ppFn0(wKnot[,1], gridmats, L, nknots, ngrid)
#   w0    <- w0PP$w
#   wPP   <- apply(wKnot[,-1,drop=FALSE], 2, ppFn, gridmats = gridmats, L = L, nknots = nknots, ngrid = ngrid, a.kap = a.kap)
#   wMat  <- matrix(sapply(wPP, extract, vn = "w"), ncol = p)
#
#   zeta0.dot <- exp(shrinkFn(p) * (w0 - max(w0)))
#   zeta0     <- trape(zeta0.dot[-c(1,L)], tau.g[-c(1,L)], L-2)
#   zeta0.tot <- zeta0[L-2]
#   zeta0     <- c(0, tau.g[2] + (tau.g[L-1]-tau.g[2])*zeta0 / zeta0.tot, 1)
#   zeta0.dot <- (tau.g[L-1]-tau.g[2])*zeta0.dot / zeta0.tot
#   zeta0.dot[c(1,L)] <- 0
#   zeta0.ticks <- pmin(L-1, pmax(1, sapply(zeta0, function(u) sum(tau.g <= u))))
#   zeta0.dists <- (zeta0 - tau.g[zeta0.ticks]) / (tau.g[zeta0.ticks+1] - tau.g[zeta0.ticks])
#   vMat      <- apply(wMat, 2, transform.grid, ticks = zeta0.ticks, dists = zeta0.dists)
#
#   reach <- nknots*(p+1)
#   gam0 <- par[reach + 1]; reach <- reach + 1
#   gam <- par[reach + 1:p]; reach <- reach + p
#   sigma <- sigFn(par[reach + 1], a.sig); reach <- reach + 1
#   nu <- nuFn(par[reach + 1]);
#
#   b0dot <- sigma * base.bundle$q0(zeta0, nu) * zeta0.dot
#   beta0.hat <- rep(NA, L)
#   beta0.hat[mid:L] <- gam0 + trape(b0dot[mid:L], tau.g[mid:L], L - mid + 1)
#   beta0.hat[mid:1] <- gam0 + trape(b0dot[mid:1], tau.g[mid:1], mid)
#
#   vNorm  <- sqrt(rowSums(vMat^2))
#   a      <- tcrossprod(vMat, x)
#   aX     <- apply(-a, 1, max)/vNorm
#   aX[is.nan(aX)] <- Inf
#   aTilde <- vMat / (aX * sqrt(1 + vNorm^2))
#   ab0    <- b0dot * aTilde
#
#   beta.hat <- kronecker(rep(1,L), t(gam))
#   beta.hat[mid:L,] <- beta.hat[mid:L,] + apply(ab0[mid:L,,drop=FALSE], 2, trape, h = tau.g[mid:L], len = L - mid + 1)
#   beta.hat[mid:1,] <- beta.hat[mid:1,] + apply(ab0[mid:1,,drop=FALSE], 2, trape, h = tau.g[mid:1], len = mid)
#   beta.hat <- beta.hat / x.sc
#   beta0.hat <- beta0.hat - rowSums(beta.hat * x.ce)
#   betas <- cbind(beta0.hat, beta.hat)
#   if(reduce) betas <- betas[reg.ix,,drop = FALSE]
#   return(betas)
# }
