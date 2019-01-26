##############################
## R seminar 5-1: decision tree
##############################
# setting  
par(family="mono")

require(rpart)      # rpart
require(rpart.plot) # rpart.plot

# stagec data (stage C prostate cancer)
d <- data.frame(x=stagec[3:8], y=as.factor(stagec$pgstat))
dim(d)
par(mfrow=c(1,1)); plot(d)

# rpart: cp=0.01 (default)
rp <- rpart(y~.,d) # rpart
rp$control$cp      # regularization par.
# training error
mean(predict(rp,type='class') != d$y)   
# plot
par(mfrow=c(1,1)); rpart.plot(rp, roundint=FALSE)


# rpart with variables: x.grade, x.g2
rpf <- rpart(y ~ x.grade + x.g2, d) # rpart
rpf$control$cp # regularization par.
# training error
mean(predict(rpf,type='class') != d$y) 
# plot
par(mfrow=c(1,1)); rpart.plot(rpf, roundint=FALSE)


# rpart: cp=0.1
rq <- rpart(y~.,d,control=rpart.control(cp=0.1))
rq$control$cp # regularization par.
# training error
mean(predict(rq,type='class')!=d$y)
# plot
par(mfrow=c(1,1)); rpart.plot(rq, roundint=FALSE)


# spiral data
require(mlbench) # mlbench.spirals
# training data
d <- data.frame(mlbench.spirals(200, 1.3,sd=0.1))
# test data
td <- data.frame(mlbench.spirals(1000,1.3,sd=0.1))

# plot training data
par(mfrow=c(1,1)); plot(d[,1:2],col=d$c,lwd=2)

# rpart
rp <- rpart(classes~., d) # learning
pred <- predict(rp,d,type='class') # prediction
# training error
mean(pred != d$c)
# test error
mean(predict(rp,td,type='class')!=td$c) 

# plot data
par(mfrow=c(1,2))
plot(d[,1:2],col=d$c,lwd=2); plot(d[,1:2], col=pred,lwd=2)
# plot tree
rpart.plot(rp, roundint=FALSE)



##############################
## R seminar 5-2: bagging
##############################

# spiral data
require(mlbench) # mlbench.spirals
# training data
d <- data.frame(mlbench.spirals(200, 1.3,sd=0.1))
# test data
td <- data.frame(mlbench.spirals(1000,1.3,sd=0.1))

par(mfrow=c(1,1))
plot(d[1:2],col=d$c,lwd=2,main='training data')

# learning
require(rpart) # rpart
require(ipred) # bagging

# rpart
rp <- rpart(classes~., d)
# bagging with 100 trees
ba <- bagging(classes~., d, mfinal=100)

# rpart
rp_pred <- predict(rp,td,type='class') # prediction
mean(rp_pred != td$c)                  # test error

# bagging
ba_pred <- predict(ba,td,type='class') # prediction
mean(ba_pred != td$c)                  # test error

# plot
par(mfrow=c(1,2))
plot(td[,1:2],col=rp_pred,main='rpart')
plot(td[,1:2],col=ba_pred,main='bagging')


##############################
## R seminar 5-3: random forest
############################## 

require(randomForest) # randomForest
require(rpart) # rpart
require(ipred) # bagging

# spiral data
require(mlbench) # mlbench.spirals
 d <- data.frame(mlbench.spirals(200, 1.3,sd=0.1))  # training data
td <- data.frame(mlbench.spirals(1000,1.3,sd=0.1)) # test data
par(mfrow=c(1,1)); plot(d[,1:2],col=d$c,lwd=2)

# random forest with 100 trees
rf <- randomForest(classes~.,d,ntree=100)
rf_pred <- predict(rf,td,type='class') # prediction
mean(rf_pred != td$c)                  # test error

# other methods: rpart
rp <- rpart(classes~., d)
rp_pred <- predict(rp,td,type='class') # prediction
mean(rp_pred != td$c)                  # test error

# other methods: bagging
ba <- bagging(classes~., d, mfinal=100)
ba_pred <- predict(ba,td,type='class') # prediction
mean(ba_pred != td$c)                  # test error

# plot: rpart vs. RandomForest
par(mfrow=c(1,2))
plot(td[,1:2],col=rp_pred,main='rpart')
plot(td[,1:2],col=rf_pred,main='RandomForest')

# plot: bagging vs. RandomForest
par(mfrow=c(1,2))
plot(td[,1:2],col=ba_pred,main='bagging')
plot(td[,1:2],col=rf_pred,main='RandomForest')


# Vowel data
library(mlbench)  # Vowel data
data(Vowel)
d <- data.frame(x=Vowel[,-11], y=as.factor(Vowel[,11]))
dim(d)

idx <- sample(nrow(Vowel),600)  # index of training data 

# random forest
rf <- randomForest(y~., d[idx,], ntree=100, na.action=na.omit)
pred <- predict(rf,d[-idx,])       # predict
mean(pred!=d$y[-idx],na.rm=TRUE)   # test error

# rpart
rp <- rpart(y~., d[idx,], na.action=na.omit)
pred <- predict(rp,d[-idx,],type='class') # predict
mean(pred!=d$y[-idx],na.rm=TRUE)          # test eror

# bagging
ba <- bagging(y~., d[idx,], na.action=na.omit,mfinal=100)
pred <- predict(ba,d[-idx,])       # predict
mean(pred!=d$y[-idx],na.rm=TRUE)   # test error


##############################
## R seminar 5-4: boosting
############################## 

# library
require(rpart)        # rpart
require(ipred)        # bagging
require(randomForest) # randomForest
require(xgboost)      # xgboost
require(kernlab)      # spam data


# read spam data
data(spam)
dim(spam)

# data setup
x <- spam[,-58]; y <- spam[,58]; y <- as.integer(y)-1
# index of training data 
idx <- sample(nrow(spam),3000)

tr <- list(x=as.matrix(x[ idx,]),y=y[ idx]) # training data
te <- list(x=as.matrix(x[-idx,]),y=y[-idx]) # test data 
# data frame
d <- data.frame(tr$x, y=as.factor(tr$y))


# ensemble learning: setting
T <- 500     # num. of trees

# XGBoost
xgb  <- xgboost(data=tr$x, label=tr$y, 
                nround=T, 
                objective="binary:logistic", 
                verbose=0)

xgb_pred <- predict(xgb, te$x) > 1/2   # predict
mean(xgb_pred != te$y)                 # test error


# rpart
rp <- rpart(y~., d) 
rp_pred <- predict(rp,data.frame(te$x), type='class') # predict
mean(rp_pred != te$y)                                 # test error 

# bagging
ba <- bagging(y~., d, mfinal=T) 
ba_pred <- predict(ba,te$x,type='class') # prediction
mean(ba_pred != te$y)                    # test error

# random forest
rf <- randomForest(y~., d, ntree=T) 
rf_pred <- predict(rf,te$x,type='class') # predict
mean(rf_pred != te$y)                    # test error 

# cross validation for xgboost
K <- 5
xcv <- xgb.cv(data=tr$x, label=tr$y, 
              nfold=K,    # K-cross validation
              nround=T,   # T rounds
              metrics="error", 
              objective="binary:logistic",
              verbose=F)

# plot: training error, cross validation error
ylim <- range(c(xcv[[4]]$train_error_mean, xcv[[4]]$test_error_mean))
par(mfrow=c(1,1), ps=14)
 plot(xcv[[4]]$train_error_mean, lwd=2, type='l',log='x',ylim=ylim,xlab='round',ylab='error')
lines(xcv[[4]]$test_error_mean,  lwd=2, col=2) 
lines(rep(min(xcv[[4]]$test_error_mean),T), lty=2)
legend('topright',legend=c("cv error","training error"), col=c(2,1), lty=c(1,1))


# XGBoost with optimal T
xgb  <- xgboost(data=tr$x, label=tr$y, 
                nround=which.min(xcv[[4]]$test_error_mean),
                objective="binary:logistic", 
                verbose=0)

xgb_pred <- predict(xgb, te$x) > 1/2  # predict
mean(xgb_pred != te$y)                # test error
