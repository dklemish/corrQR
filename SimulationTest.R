library(copula)
library(corrQR)
library(invgamma)

set.seed(4)

# Set up variables
X  <- rep(seq(1:30),50)
X  <- X[order(X)]
n  <- length(X)
Y1 <- rep(NA, n)
Y2 <- rep(NA, n)

# Generate mixture components
mix1 <- rbinom(n, 1, X/100)
X.1  <- X[mix1==0]
X.2  <- X[mix1==1]

# Generate underlying correlations
rho <- 0.9
U   <- rCopula(n, normalCopula(rho, dim=2, dispstr = "un"))

Y1[mix1==0] <- qnorm(U[mix1==0, 1], mean=X.1, sd=4)
Y1[mix1==1] <- qnorm(U[mix1==1, 1], mean=3.5*log(X.2)^2, sd=X.2/10)

Y2 <- qinvgamma(U[,2], shape = 2*log(X), rate = 2*X)

plot(X, Y1, ylim=c(min(Y1, Y2), max(Y1, Y2)))
lines(X, Y2, col="red", type="p")

par(mfrow=c(2,2))
plot(Y1[X==6], Y2[X==6])
plot(Y1[X==12], Y2[X==12])
plot(Y1[X==18], Y2[X==18])
plot(Y1[X==24], Y2[X==24])
par(mfrow=c(1,1))

plot(X[mix1==0], Y1[mix1==0])
lines(X[mix1==1], Y1[mix1==1], col="red", type="p")

