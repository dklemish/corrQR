library(copula)
library(corrQR)
library(ggplot2)
library(gridExtra)

set.seed(4)

### Model 1
# Set up variables
X1 <- rep(seq(1:10),200)
X1 <- X1[order(X1)]
n  <- length(X1)
X2 <- rep(seq(1:10),200)

Y1 <- rep(NA, n)
Y2 <- rep(NA, n)

# Generate underlying correlations
rho <- 0.9
U   <- rCopula(n, normalCopula(rho, dim=2, dispstr = "un"))
Y1  <- qgamma(U[,1], shape=4 + X1, rate=1)
Y2  <- qgamma(U[,2], shape=5*X2, rate=2)

X   <- data.frame(X1 = X1, X2=X2)
Y   <- data.frame(Y1 = Y1, Y2=Y2)

dat <- data.frame(X1=X1, X2=X2, Y1=Y1, Y2=Y2)

# Plot data
g1 <- ggplot(data=dat[X1==2 & X2==2,]) +
  geom_point(aes(x=Y1, y=Y2), color="blue") +
  labs(x=bquote(Y[1]~'|'~X[1]==2~','~X[2]==2),
       y=bquote(Y[2]~'|'~X[1]==2~','~X[2]==2)) +
  theme_bw()
g2 <- ggplot(data=dat[X1==2 & X2==8,]) +
  geom_point(aes(x=Y1, y=Y2), color="blue") +
  labs(x=bquote(Y[1]~'|'~X[1]==2~','~X[2]==8),
       y=bquote(Y[2]~'|'~X[1]==2~','~X[2]==8)) +
  theme_bw()
g3 <- ggplot(data=dat[X1==5 & X2==5,]) +
  geom_point(aes(x=Y1, y=Y2), color="blue") +
  labs(x=bquote(Y[1]~'|'~X[1]==5~','~X[2]==5),
       y=bquote(Y[2]~'|'~X[1]==5~','~X[2]==5)) +
  theme_bw()
g4 <- ggplot(data=dat[X1==10 & X2==10,]) +
  geom_point(aes(x=Y1, y=Y2), color="blue") +
  labs(x=bquote(Y[1]~'|'~X[1]==10~','~X[2]==10),
       y=bquote(Y[2]~'|'~X[1]==10~','~X[2]==10)) +
  theme_bw()
grid.arrange(g1, g2, g3, g4, ncol=2)

# Fit model
test1 <- corrQR(X, Y, 6, nsamp=2000, thin=5)
coef(test1, nr=2, nc=2)
