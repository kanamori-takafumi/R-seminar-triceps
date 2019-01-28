
PlotDB <- function(sv,dat,len=400,main=NULL){
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

