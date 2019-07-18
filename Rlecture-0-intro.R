##############################
## R seminar 0-1: introduction
##############################

# Arithmetic
-10+3.43
x <- 3*1.23

y <- x/3
y

x / y

# square root, power
sqrt(2)
sqrt(2)^2


# Infinity, pi
1/Inf                # Inf: Infinity
pi


# Comparison
1==1
1!=1
-Inf<1 & 1< 3    # AND 
2<1 | 3<1        # OR


# vector/array
x <- c(4,-1)      # 2-dim vector
x

y <- -3:4         # from -3 to 4
y

c(x,y)


# sum/mean
a <- c(3,-2,4,1,9)                   
3*a

sum(a)                              # sum
mean(a)                             # mean


# elements of vectors
v <- rep(1:5,3)           # vector v
v

v[3]                      # 3rd element
v[3:8]                    # v[3], v[4],...,v[8]
which(v==2)               # v[2],v[7],v[12] = 2
v[which(v==2)]


# matrix
X <- rbind(c(1,2),c(3,4)) 
X

cbind(c(1,2),c(3,4))


#
Y <- rbind(X,c(-1,-2))
Y

cbind(Y,c(1.1, -2.2, 3))


# matrix generate
A <- matrix(1:12,3,4)       
A

dim(A)                      # size of A


# elements of matrix
A[2,]                            # 2nd row
A[,4]                            # 4th column
A[,c(2,4)]                       # 2nd,4th column
A[2,4]                           # (2,4) element


# linear algebra
A <- matrix(1:12,3,4)
A

B <- A %*% t(A)/100
B


# 
diag(B) <- 1        # Diagonal elements = 1
B

d <- c(1,0,-1)      # generate vector d 
d

solve(B,d)          # solve B x = d 


# for 
s <-0; for(i in 1:8){  # sum from 1 to 8
  s <- s+i
  print(s)
}

sum(1:8)           # same result


# parity function
parity <- function(x){
  if(x%%2==0){        # if
    print('even number')
  }else if(x%%2==1){  # else
    print('odd number')
  }else{
    print('non integer')
  }
}

parity(2)

parity(pi)

# plot
# random data
x <- rnorm(100); y <- x^2 + rnorm(100,sd=1)
plot(x,y)                    



# exercise: mysign function
# mysign <- function(x){
#   if(????){       
#     print('positive')
#   }else if(????){
#     print('negative')
#   }else{
#     print('null')
#   }
# }


