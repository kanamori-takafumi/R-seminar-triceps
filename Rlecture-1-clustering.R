##############################
## R seminar 1-1: k-means
##############################
par(family="mono")

# 2dim data
require(mlbench) # mlbench.2dnormals
dat <- mlbench.2dnormals(100,cl=2,sd=0.6)
x <- dat$x; y <- dat$c
par(mfrow=c(1,1)); plot(x, lwd=2)

# k-means: k=2
km <- kmeans(x,2,nstart=10); km$cluster  # cluster

# plot: k=2
par(mfrow=c(1,2), ps=14)  
plot(x,col=y,main="label") 
plot(x,col=km$cl,main="k-means")

# k-means: k=3
km <- kmeans(x,3,nstart=10); km$cluster # cluster

# plot: k=3
par(mfrow=c(1,2), ps=14)
plot(x,col=y,lwd=2,main="label")
plot(x,col=km$cl,lwd=2,main="k-means")

# wine data 
require(rattle.data) # wine data
data(wine)           # read data
x <- wine[,-1]       # data point 
x <- scale(x)        # scaling
class <- wine[,1]    # label: type of wine
dim(x)

# k-means (k=3)
km <- kmeans(x,3,nstart=10);km$cluster # cluster

# plot
par(mfrow=c(1,2), ps=14); pca <- prcomp(x)  # PCA
plot(pca$x[,1:2],col=class,main="type of wine",lwd=2) 
plot(pca$x[,1:2],col=km$cl,main="k-means",lwd=2) 




##############################
## R seminar 1-2: spectral clustering
##############################
# spiral data
require(mlbench)  # mlbench.spirals
data <- mlbench.spirals(300, cycles=1, sd=0.05)
par(mfrow=c(1,1), ps=14); plot(data,main="data")

# spectral clustering
require(kernlab)  # specc
sc <- specc(data$x, centers=2)  # spectral clustering
km <- kmeans(data$x,2)          # kmeans

# plot
par(mfrow=c(1,2), ps=14)
plot(data$x, col=sc,main='spectral clustering')
plot(data$x, col=km$cl, main="kmeans")


##############################
## R seminar 1-4: hierarchical cluster
##############################
require(rattle.data)         # wine data

idx <- sample(nrow(wine),60) # random resampling
x <- scale(wine[idx,-1])     # wine data
class <- wine[idx,1]         # type of wine

# hclust: max
d <- dist(x)     # distance matrix
hc <- hclust(d)  # hclust
par(mfrow=c(1,1), ps=14); plot(hc,label=class,main='max')  

# hclust: Ward
d <- dist(x)     # distance matrix
hc <- hclust(d, method='ward.D')  # hclust
par(mfrow=c(1,1), ps=14); plot(hc,label=class,main='Ward')

