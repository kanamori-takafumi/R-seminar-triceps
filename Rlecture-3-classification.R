##############################
## R seminar 3-1: binary SVM
##############################
# setting  
par(family="mono")
  
# data
require(mlbench)
dat <- mlbench.2dnormals(200,cl=2,sd=1)  # data
par(mfrow=c(1,1)); plot(dat,lwd=2)

# linear svm
require(kernlab)  # ksvm
linsvm <- ksvm(dat$x,dat$c,type="C-svc", kernel="vanilladot")
linsvm    

# prediction
tdat <- mlbench.2dnormals(1000,cl=2,sd=1)  # test data
predy <- predict(linsvm,tdat$x) # prediction
mean(predy != tdat$c)           # test error

# plot decision boundary
source('Rscripts.r')
par(mfrow=c(1,1)); PlotDB(linsvm,dat,main="linear SVM")



##############################
## R seminar 3-2: kernel-SVM
##############################
  
# data
require(mlbench)
# training data
dat  <- mlbench.spirals(300, cycles=1,sd=0.15) 
# test data
tdat <- mlbench.spirals(1000,cycles=1,sd=0.15) 

# plot training data
par(mfrow=c(1,1)); plot(dat, lwd=2)

require(kernlab)  # ksvm

# kernel svm(poly kernel)
# poly-kernel: degree=2
sv2 <- ksvm(dat$x,dat$c,kernel="polydot", kpar=list(degree=2))  
sv2

# test error
mean(predict(sv2,tdat$x)!=tdat$c)   

# kernel svm(poly kernel)
# poly-kernel: degree=3
sv3 <- ksvm(dat$x,dat$c,kernel="polydot",kpar=list(degree=3))
sv3

# test error
mean(predict(sv3,tdat$x)!=tdat$c)  

# plot decision boundary
source('Rscripts.r'); par(mfrow=c(1,2), ps=14)
PlotDB(sv2,dat,len=300,main="poly: degree=2")
PlotDB(sv3,dat,len=300,main="poly: degree=3")

# cross validation
# degree=2
sv <- ksvm(dat$x,dat$c, cross=10, kernel="polydot",kpar=list(degree=2))
cross(sv)

# degree=3
sv <- ksvm(dat$x,dat$c, cross=10, kernel="polydot",kpar=list(degree=3))
cross(sv)

# cross validation
# degree=4
sv <- ksvm(dat$x,dat$c, cross=10, kernel="polydot",kpar=list(degree=4))
cross(sv)

# degree=5
sv <- ksvm(dat$x,dat$c, cross=10, kernel="polydot",kpar=list(degree=5))
cross(sv)



##############################
## R seminar 3-3: model parameter setting
############################## 

# data
require(mlbench)
dat <- mlbench.spirals(200, cycles=1.2,sd=0.16) # train
td  <- mlbench.spirals(1000,cycles=1.2,sd=0.16) # test

# plot
par(mfrow=c(1,1)); plot(dat,lwd=2)

# require
require(kernlab)               # ksvm
require(doParallel)            # foreach

# cross validation for sigma in Gauss kernel
# sigma candidates
sc <- exp(seq(log(0.01),log(100),l=30))  
sc

# cross validation error and test error
err <- foreach(s=sc,.combine=rbind)%do%{
  # cross validation for each sigma
  kcv <- ksvm(dat$x,dat$c,type="C-svc",
              kernel='rbfdot',
              cross=10,                # 10-fold CV
              kpar=list(sigma=s), C=1) # model par.
  # error
  data.frame(cv=cross(kcv),
             test=mean(predict(kcv,td$x)!=td$c))
}

# optimal sigma
opts <- sc[which.min(err$cv)]
opts

# plot cv error
par(mfrow=c(1,1))
plot(sc, err$cv, ylim=range(unlist(err)),log='x')
lines(sc,err$test,col=2,lwd=2)



##############################
## R seminar 3-4: multiclass
##############################  

# data
require(kernlab)
require(mlbench)
G <- 8         # 8 classes
dat  <- mlbench.2dnormals(500, cl=G,sd=0.8) # train
tdat <- mlbench.2dnormals(1000,cl=G,sd=0.8) # test

# plot
par(mfrow=c(1,1)); plot(dat,lwd=2)

# linear kernel
# default: one-vs-one
linsv <- ksvm(dat$x,dat$c,kernel='vanilladot')
mean(predict(linsv,tdat$x)!=tdat$c)  # test error

# Gauss kernel
# default: one-vs-one
rbfsv <- ksvm(dat$x,dat$c,kernel='rbfdot')
mean(predict(rbfsv,tdat$x)!=tdat$c) # test eror

# plot decision boundary
source('Rscripts.r'); par(mfrow=c(1,2),ps=14)
PlotDB(linsv,dat,main="linear")
PlotDB(rbfsv,dat,main="Gauss")
