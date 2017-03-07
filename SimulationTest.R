library(copula)
library(corrQR)

set.seed(4)

### Model 1
# Set up variables
X1 <- rep(seq(1:20),50)
X1 <- X1[order(X1)]
n  <- length(X1)
X2 <- rep(seq(1:10),100)

Y1 <- rep(NA, n)
Y2 <- rep(NA, n)

# Generate underlying correlations
rho <- 0.9
U   <- rCopula(n, normalCopula(rho, dim=2, dispstr = "un"))
Y1  <- qgamma(U[,1], shape=4 + X1, rate=1)
# Y2  <- qgamma(U[,2], shape=X2^2, rate=2)
Y2  <- qgamma(U[,2], shape=3*X2, rate=2)

X   <- data.frame(X1 = X1, X2=X2)
Y   <- data.frame(Y1 = Y1, Y2=Y2)

plot(X2, Y2, xlim=c(1,20))
lines(X1, Y1, col="blue", type="p")

par(mfrow=c(2,2))
plot(Y1[X1==5 & X2==5], Y2[X1==5 & X2==5])
plot(Y1[X1==10 & X2==5], Y2[X1==10 & X2==5])
plot(Y1[X1==15 & X2==10], Y2[X1==15 & X2==10])
plot(Y1[X1==20 & X2==10], Y2[X1==20 & X2==10])
par(mfrow=c(1,1))

test1 <- corrQR(X, Y, 6, nsamp=1000, thin=5)



# Generate mixture components
mix1 <- rbinom(n, 1, X/100)
X.1  <- X[mix1==0]
X.2  <- X[mix1==1]

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

