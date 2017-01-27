# Question 1
load("heads.Rdata")

Y <- heads
n <- nrow(Y)
p <- ncol(Y)
ones <- rep(1, n)
Ybar <- apply(Y, 2, mean)
C <- diag(n) - ones %*% t(ones)/n
Yc <- C %*% Y
S <- t(Yc) %*% Yc / (n-1)

L    <- eigen(S)$values
V    <- eigen(S)$vectors
Fmat <- Yc %*% V

CFY <- cor(Y, Fmat)[,1:2]
plot(1.5*c(-1,1),1.5*c(-1,1),
     type="n", xlab="PC1 cor", ylab="PC2 cor") 
x<-seq(-1,1,length=100)
lines( x, sqrt(1-x^2))
lines( x, -sqrt(1-x^2))
abline(h=0); abline(v=0)
text(CFY,labels=rownames(CFY)) 
points(CFY,pch=16,col=1:4)


cbind( L, L/sum(L), cumsum(L)/sum(L))

# Question 2
rm(list=ls())
load("phoneme.Rdata")

Y <- phoneme
n <- nrow(Y)
p <- ncol(Y)
ones <- rep(1, n)
Ybar <- apply(Y, 2, mean)
C <- diag(n) - ones %*% t(ones)/n
Yc <- C %*% Y
S <- t(Yc) %*% Yc / (n-1)
L    <- eigen(S)$values
V    <- eigen(S)$vectors

actual.sounds <- rownames(Y)
sounds <- unique(actual.sounds)

par(mfrow=c(1,2))
plot(L)
plot(cumsum(L)/sum(L))
par(mfrow=c(1,1))

Fmat <- Yc %*% V
cols <- rep("black", n)
cols[actual.sounds==sounds[2]] <- "red"
cols[actual.sounds==sounds[3]] <- "blue"
cols[actual.sounds==sounds[4]] <- "yellow"
cols[actual.sounds==sounds[5]] <- "green"
pairs(Fmat[,1:4], col=cols)

phoneme_mean <- matrix(0, nrow=5, ncol=4)
i <- 1
for(nm in sounds){
  phoneme_mean[i, ] <- apply(Fmat[rownames(Y)==nm, 1:4], 2, mean)
  i <- i + 1
}
rownames(phoneme_mean) <- unique(rownames(Y))

phoneme_diff <- matrix(0, nrow=n, ncol=5)
for(i in 1:n){
  for(j in 1:5){
    phoneme_diff[i,j] <- sum((Fmat[i,1:4] - phoneme_mean[j,])^2)
  }
}

apply(phoneme_diff, 1, which.min)

fitted.sound <- sounds[apply(phoneme_diff, 1, which.min)]
