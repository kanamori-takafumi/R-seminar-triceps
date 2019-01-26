##############################
## R seminar 4-1: sparse learning: lasso
##############################
# setting  
par(family="mono")

require(glmnet)      # glmnet

# data setup
n <- 30       # data size
d <- 50       # x dim
s <- 10       # num. of non-zero elements

# data generate
beta <- c(rep(1,s),rep(0,d-s))
beta

X <- matrix(rnorm(n*d),n)
y <- X %*% beta + rnorm(n,sd=0.01)

# estimate
la <- glmnet(X,y,alpha=1)       # lasso
ri <- glmnet(X,y,alpha=0)       # ridge

# plot
par(mfrow=c(1,2))
plot(la$b[1:d,ncol(la$b)],lwd=2,t='h',m="lasso") # lasso
plot(ri$b[1:d,ncol(ri$b)],lwd=2,t='h',m="ridge") # ridge

# path following
par(mfrow=c(1,2))
plot(la,main="lasso")       # lasso
plot(ri,main="ridge")       # ridge



##############################
## R seminar 4-2: erastic net
##############################
require(glmnet)

# data setting
n <- 50   # data size
d <- 100   # data dim
s <- 10    # nom. of non-zero elements

# true parameter
beta <- c(rep(1,s),rep(0,d-s))      
beta

# data generate
X <- matrix(rnorm(n*d),n)
y <- X %*% beta + rnorm(n,sd=0.01)

# estimate
la <- glmnet(X,y,alpha=1)           # lasso
el <- glmnet(X,y,alpha=0.1)         # erastic net

# plot: estimated coefficients
par(mfrow=c(1,2))
plot(la$b[1:d,ncol(la$b)],type='h',m="lasso")
plot(el$b[1:d,ncol(el$b)],type='h',m="erastic net")

# plot: num. of selected variables
# erastic net
par(mfrow=c(1,1))
plot(colSums(abs(el$b)),colSums(el$b!=0),col=2,type='l',lwd=2,xlab='weakness of regularization',ylab="num: non")
# lasso
lines(colSums(abs(la$b)),colSums(la$b!=0),lwd=2,lty=2)
# upper bound of lasso
lines(cbind(colSums(abs(el$b)),n),lty=3)
legend("topleft",c("erastic","lasso"),col=c(2,1),lty=c(1,2),lwd=c(2,2),bg="white")


##############################
## R seminar 4-3: fused lasso
############################## 
require(HDPenReg)      # EMfusedlasso

# data 
beta <- rnorm(5) %x% rep(1,10)
x <- diag(length(beta))
y <- as.vector(x %*% beta + rnorm(50,sd=0.2))

# data plot
par(mfrow=c(1,1)); plot(y,lwd=2)
lines(beta,col=2,lwd=3)

# fused lasso: EMfusedlasso(I, y, lambda1, lambda2, ...)
res <- EMfusedlasso(diag(length(y)), y, 0, 1, intercept=FALSE)

# plot
par(mfrow=c(1,1)); plot(y,lwd=2)
lines(beta,col=2,lwd=3)
lines(res$coef,main="fused-lasso",col=4,lty=2,lwd=3)
legend("topright",legend=c("True","Fitted"),lwd=3,col=c(2,4),lty=c(1,2))

