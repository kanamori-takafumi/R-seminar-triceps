##############################
## R seminar 2-1: least square method
##############################
# setting  
par(family="mono")
 
require(carData)  # UN data
data(UN); par(mfrow=c(1,1))
plot(UN[,4],UN[,7],log='xy', xlab=colnames(UN)[4],ylab=colnames(UN)[7],main="Data") # plot data: GDP vs. Infant Mortality

# Least Square Method
logUN <- log(na.omit(UN[,c(4,7)]))          
res <- lm(infantMortality~ppgdp,data=logUN);res                                         
y <- predict(res)  # predict
# plot data: GDP vs. Infant Mortality
par(mfrow=c(1,1)); plot(UN[,4],UN[,7],log='xy', xlab=colnames(UN)[4],ylab=colnames(UN)[7])
lines(exp(logUN$ppgdp), exp(y), col=2, lwd=2)



##############################
## R seminar 2-2: Ridge regression
##############################
require(glmnet)    # glmnet

# example: data
n <- 100           # sample size
x <- runif(n,min=-2,max=2); y <- sin(2*pi*x)/x + rnorm(n,sd=0.5)
# data plot
par(mfrow=c(1,1)); plot(x,y,main="data")

# ridge regression
degree <- 8        # degree
lambda <- 10^(-5)  # lambda
# data matrix
Phi <- outer(x,1:degree,FUN="^")

# ridge regression
gl <- glmnet(Phi,y,alpha=0, family="gaussian",lambda=lambda); gl
# prediction
tx <- seq(-2,2,l=100)                 # test points
testPhi <- outer(tx,1:degree,FUN="^") # test data matrix
predy <- predict(gl,newx=testPhi)     # prediction
# plot estimator
par(mfrow=c(1,1)); plot(x,y); lines(tx,predy,lwd=2,col=2)

# strong regularization
lambda <- 10^(-1)  # lambda
# ridge regression
gl <- glmnet(Phi,y,alpha=0,family="gaussian",lambda=lambda)
# prediction
predy <- predict(gl,newx=testPhi)     # prediction
# plot
par(mfrow=c(1,1)); plot(x,y,main='strong regularization'); lines(tx,predy,lwd=2,col=2)



##############################
## R seminar 2-3: cross validation
##############################
require(glmnet)    # glmnet

# example: data
n <- 100           # sample size
x <- runif(n,min=-2,max=2); y <- sin(2*pi*x)/x + rnorm(n,sd=0.5)
par(mfrow=c(1,1)); plot(x,y,main="data")

# cross validation for lambda
# data matrix
Phi <- outer(x,1:8,FUN="^")
# CV for ridge regression
l <- 10^(-10:0); l

cvgl <- cv.glmnet(Phi,y,alpha=0,family="gaussian", nfold=10, lambda=l)
# optimal lambda
cvgl$lambda.min
# plot cross validation error
# par(mfrow=c(1,1)); plot(cvgl$lambda, cvgl$cvm, log='xy', lwd=2)

# estimation: optimal lambda
gl <- glmnet(Phi,y,alpha=0,family="gaussian", lambda=cvgl$lambda.min)
# estimation: lambda=0.1
gl2 <- glmnet(Phi,y,alpha=0,family="gaussian",lambda=0.1)

# prediction
tx <- seq(-2,2,l=100)            # test points
testPhi <- outer(tx,1:8,FUN="^") # test data matrix
py  <- predict(gl,newx=testPhi)  # y: optimal lambda
py2 <- predict(gl2,newx=testPhi) # y: large lambda

# plot
par(mfrow=c(1,1)); plot(x,y)   
lines(tx,py, lwd=2,col=2) # y: optimal lambda
lines(tx,py2,lwd=2,col=4) # y: large lambda



##############################
## R seminar 2-4: robust
##############################
# data
require(carData)   # Davis
data(Davis)
par(mfrow=c(1,1)); plot(Davis$weight, Davis$height, xlab='weight', ylab='height',lwd=2)

# LSM: an outlier
a <- lm(height~weight,data=Davis); a
# LSM: no outlier
lm(height~weight,data=Davis[-12,])    
# robust regression
require(MASS)   #  rlm
b <- rlm(height~weight,data=Davis); b

# prediction
tx <- data.frame(weight=seq(40,160,l=100)) # test x
pred_LSM <- predict(a,tx) # LSM
pred_rob <- predict(b,tx) # robust
# plot
par(mfrow=c(1,1)); plot(Davis$weight,Davis$height, xlab='weight',ylab='height')
lines(tx$weight,pred_LSM,col=2,lwd=3)       # LSM
lines(tx$weight,pred_rob,col=4,lwd=3,lty=2) # robust

