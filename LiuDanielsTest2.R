library(copula)
library(mvtnorm)
library(MCMCpack)
library(ggplot2)

set.seed(6)

Q      <- 3
n.obs  <- 1000
n.samp <- 10000
R <- matrix(c(1, 0.8, -0.3, 0.8, 1, -0.5, -0.3, -0.5, 1), nrow=Q)

# Liu & Daniels paper
R.samp <- array(0, dim=c(Q,Q,n.samp))
accept <- rep(0, n.samp)
R.samp[,,1] <- diag(Q)

log.curr.det <- log(det(R.samp[,,1]))
expo <- -(Q+1)/2
uniform.accept.draws <- runif(n.samp)

# Target prior \propto 1
# Candidiate prior \propto |R|^((Q+1)/2)
for(i in 2:n.samp){
  Z <- rmvnorm(n.obs, sigma=R)
  #Zt <- t(Z)

  # Calculate current D since \sum Z_{ij}^2 = d_{jj}^{-2} \forall j
  D <- diag(apply(Z^2, 2, sum)^(-0.5))

  # Calculate parameter expansion & draw new sigma matrix
  epsilon <- Z %*% D
  S       <- matrix(apply(apply(epsilon, 1, function(x) x %*% t(x)), 1, sum),
                    nrow=Q)
  Sigma   <- riwish(n.obs, S)

  # Drawing Sigma implicitly draws new D & R, so calculate new D & R
  D.new   <- diag(diag(Sigma)^-0.5)
  R.prop  <- D.new %*% Sigma %*% D.new

  # MH acceptance step
  log.prop.det <- log(det(R.prop))
  accept.prob  <- min(1, exp(expo*(log.prop.det - log.curr.det)))

  if(uniform.accept.draws[i] < accept.prob){
    R.samp[,,i]  <- R.prop
    log.curr.det <- log.prop.det
    accept[i] <- 1
  } else{
    R.samp[,,i]  <- R.samp[,,i-1]
  }
}
apply(R.samp[,,101:n.samp],c(1,2),summary)

ggplot() +
  geom_histogram(data=data.frame(x=R.samp[1,2,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()
ggplot() +
  geom_histogram(data=data.frame(x=R.samp[1,3,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()
ggplot() +
  geom_histogram(data=data.frame(x=R.samp[2,3,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()


### Tabet thesis
R.samp2      <- array(0, dim=c(Q,Q,n.samp))
#alpha.samp   <- matrix(0, nrow=n.samp, ncol = Q)
R.samp2[,,1] <- 0.5*diag(Q)

for(i in 2:n.samp){
  Z <- rmvnorm(n.obs, sigma=R)
  Zt <- t(Z)

  curr.R.inv     <- solve(R.samp2[,,i-1])
  #alpha.samp[i,] <- rgamma(n = Q, shape=(Q+1)/2, rate=1)
  #alpha.samp     <- rgamma(n = Q, shape=(Q+1)/2, rate=1)
  alpha.samp <- c(3,2,1)
  #D              <- diag(sqrt(diag(curr.R.inv)/(2*alpha.samp[i,])))
  D              <- diag(sqrt(diag(curr.R.inv)/(2*alpha.samp)))
  D.inv          <- solve(D)
  eps.star       <- D %*% Zt
  S              <- eps.star %*% t(eps.star)
  Sigma          <- riwish(n.obs + Q + 1, S)
  D.inv          <- diag(diag(Sigma)^-0.5)
  R.samp2[,,i]   <- D.inv %*% Sigma %*% D.inv
}
apply(R.samp2[,,101:n.samp],c(1,2),summary)

ggplot() +
  geom_histogram(data=data.frame(x=R.samp2[1,2,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()
ggplot() +
  geom_histogram(data=data.frame(x=R.samp2[1,3,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()
ggplot() +
  geom_histogram(data=data.frame(x=R.samp2[2,3,101:n.samp]),
                 aes(x, ..density..),
                 binwidth=0.0025) +
  theme_bw()

