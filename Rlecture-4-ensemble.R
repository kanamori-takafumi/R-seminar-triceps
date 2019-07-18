##############################
## R seminar 4-1: decision tree
##############################
# setting  
par(family="mono")

require(rpart)      # rpart
require(rpart.plot) # rpart.plot

# stagec data (stage C prostate cancer)
d <- data.frame(x=stagec[3:8], y=as.factor(stagec$pgstat)); dim(d)
par(mfrow=c(1,1)); plot(d)

# rpart: cp=0.01 (default)
rp <- rpart(y~.,d) # rpart
rp$control$cp      # regularization par.
# training error
cat("rpart: training error =",mean(predict(rp,type='class') != d$y))
par(mfrow=c(1,1)); rpart.plot(rp, roundint=FALSE)  # plot


# rpart: cp=0.1
rq <- rpart(y~.,d,control=rpart.control(cp=0.1))
rq$control$cp # regularization par.
# training error
cat("rpart: training error =",mean(predict(rq,type='class')!=d$y))
par(mfrow=c(1,1)); rpart.plot(rq, roundint=FALSE) # plot


# rpart with variables: x.grade, x.g2
rpf <- rpart(y ~ x.grade + x.g2, d) # rpart
rpf$control$cp # regularization par.
# training error
cat("rpart: training error =",mean(predict(rpf,type='class') != d$y)) 
par(mfrow=c(1,1)); rpart.plot(rpf, roundint=FALSE) # plot

# exercise
# rpf <- rpart(y ~ x.grade + x.g2, d, control=rpart.control(cp=0.1)) # rpart
# par(mfrow=c(1,1)); rpart.plot(rpf, roundint=FALSE) # plot


##############################
## R seminar 4-2: bagging
##############################
# spiral data
require(mlbench) # mlbench.spirals
d <- data.frame(mlbench.spirals(200, 1.3,sd=0.1)) # training data
td <- data.frame(mlbench.spirals(1000,1.3,sd=0.1)) # test data
par(mfrow=c(1,1)); plot(d[1:2],col=d$c,lwd=2,main='training data')

# learning
require(rpart) # rpart
require(ipred) # bagging

# rpart
rp <- rpart(classes~., d)
# bagging with 100 trees
ba <- bagging(classes~., d, mfinal=100)

# rpart
rp_pred <- predict(rp,td,type='class') # prediction
cat('rpart: test error =',mean(rp_pred != td$c),"\n")  # test error

# bagging
ba_pred <- predict(ba,td,type='class') # prediction
cat('bagging: test error =',mean(ba_pred != td$c),"\n")    # test error

# plot
par(mfrow=c(1,2));
plot(td[,1:2],col=rp_pred,main='rpart'); plot(td[,1:2],col=ba_pred,main='bagging')



##############################
## R seminar 4-3: random forest
############################## 
# libraries
require(randomForest) # randomForest
require(rpart)        # stagec data

# stagec data
d <- data.frame(x=stagec[3:8],y=as.factor(stagec$pgstat))
dim(d)

# random forest
rf <- randomForest(y~., d, ntree=1000, importance=TRUE, na.action=na.omit)
# importance of variables
importance(rf)[,3:4]

varImpPlot(rf)  # plot

# iris data
data(iris)
d <- data.frame(x=iris[,-5], y=as.factor(iris[,5]))
dim(d)

# random forest
rf <- randomForest(y~., d, ntree=1000,importance=TRUE, na.action=na.omit)
# importance of variables
importance(rf)[,4:5]

varImpPlot(rf) # plot


##############################
## R seminar 4-4: boosting
############################## 
# library
require(randomForest) # randomForest
require(xgboost)      # xgboost
require(kernlab)      # spam data

# read spam data
data(spam); dim(spam)
# data setup
x <- spam[,-58]; y <- spam[,58]; y <- as.integer(y)-1
idx <- sample(nrow(spam),3000) # index of training data 

tr <- list(x=as.matrix(x[ idx,]),y=y[ idx]) # training data
te <- list(x=as.matrix(x[-idx,]),y=y[-idx]) # test data 
d <- data.frame(tr$x, y=as.factor(tr$y))  # data frame

# ensemble learning: setting
T <- 1000     # num. of trees

# XGBoost
xgb  <- xgboost(data=tr$x, label=tr$y, 
                nround=T, 
                objective="binary:logistic", 
                verbose=0)

xgb_pred <- predict(xgb, te$x) > 1/2   # predict
cat("xgboost: test error =",mean(xgb_pred != te$y))     # test error

# random forest
rf <- randomForest(y~., d, ntree=T)
rf_pred <- predict(rf,te$x,type='class') # predict
cat("random forest: test error =",mean(rf_pred != te$y)) # test error 

# cross validation for xgboost
xcv <- xgb.cv(data=tr$x, label=tr$y, 
              nfold=5,    # 5-cross validation
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
cat("xgboost: test error =",mean(xgb_pred != te$y))   # test error

