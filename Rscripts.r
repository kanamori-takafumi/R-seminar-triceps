
## ## (negative) log likelihood
## ## theta <- c(thetaA,thetaB); n <- c(nA,nB,nAB,nO). 
## nlikelihood <- function(theta,n){# log likelihood
##   a <- theta[1]; b <- theta[2]; o <- 1-a-b
##   l <- log(c(a^2+2*a*o, b^2+2*b*o, 2*a*b, o^2))
##   -sum(n*l)
## }
## ## caclulate mle
## ## n <- c(nA,nB,nAB,nO). 
## mle <- function(n){
##   op <- optim(c(1/3,1/3),nlikelihood,n=n) 
##   list(a=op$par[1],b=op$par[2],o=1-op$par[1]-op$par[2])
## } 


## ## ## ## ## ## 
## # K: num. of components
## # x: (n,d) matrix 
## EM_mixBernoulli <- function(x,K=5,maxitr=1000,tol=1e-5){
##   d <- ncol(x); n <- nrow(x)         # 次元dとデータ数n
##   eps <- .Machine$double.eps         
##   # コンポーネント初期設定
##   mu <- mean(x)                      
##   p <- matrix(rbeta(K*d,shape1=mu,shape2=(1-mu)),K)
##   q <- rep(1/K,K)                    # 混合確率の初期値
##   ul <- Inf
##   for(itr in 1:maxitr){              # EMアルゴリズム
##     # 多次元ベルヌーイ分布の確率を計算
##     mp <- exp(log(p)%*%t(x)+log(1-p)%*%t(1-x))*q
##     # γ, q, p 更新．pmin, pmax で発散を防ぐ．
##     gmm <- pmin(pmax(t(t(mp)/colSums(mp)),eps),1-eps)
##     q <- pmin(pmax(rowSums(gmm)/n, eps),1-eps)
##     p <- pmin(pmax((gmm%*%x)/(n*q),eps),1-eps)
##     # 負の対数尤度の上界
##     uln <- -sum(gmm*((log(p)%*%t(x)+log(1-p)%*%t(1-x)+log(q))-log(gmm)))
##     if(abs(ul-uln)<tol){            # 停止条件
##       break
##     }
##     ul <- uln
##   }
##   BIC <- ul+0.5*(d*K+(K-1))*log(n) # BIC
##   list(p=p,q=q,gamma=gmm,BIC=BIC)
## }

## ## plot data
## numplot <- function(i,img){
##   m <- 16-matrix(img[i,],8,8)[,8:1]
##   a <- b <- 1:8
##   image(a,b,m,col=gray((0:16)/16))
## }




PlotDB <- function(sv,dat,len=500,main=NULL){
  x <- dat$x; cl <- as.numeric(dat$cl)
  r1 <- range(x[,1]); r2 <- range(x[,2])
  ## トレーニングデータのプロット  
  plot(data.frame(x),col=cl,pch=cl,xlim=r1,ylim=r2,lwd=2,cex=1.1,axes=TRUE,main=main)
  ## 格子点上でラベル予測
  x1 <- seq(r1[1],r1[2],l=len); x2 <- seq(r2[1],r2[2],l=len)
  X <- expand.grid(x1,x2)                                
  Y <- as.numeric(predict(sv,X)); Y <- matrix(Y,length(x1))  
  for(i in 1:max(Y)){  # 判別境界のプロット
    contour(x1,x2,Y==i,levels=0.5,lwd=2,lty=1,col=4,drawlabels=FALSE,add=TRUE)
  }
}

