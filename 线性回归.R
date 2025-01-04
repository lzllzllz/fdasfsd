x=c(2,2,3,4,5,4.5,5.5,7.5,8,9,10,11)
y=c(30,35,40,45,50,55,66,75,85,100,110,120)
plot(x,y)
lm <- lm(y~x)

summary(lm)

beta0=9.5077
beta1=9.7470
x0 <- 4.2
y0 <- beta0+beta1*x0


yhat <- beta0 + beta1*x
res <- y-predict(lm)
sigma <- sqrt(sum(res^2)/(n-2))
SST <- sum((y-mean(y))^2)
SSE <- sum(res^2)
SSR <- SST - SSE
r2=SSR/SST
anova(lm(y~1), lm(y~x))
anova(lm(y~x))

beta0=9.5077
beta1=9.7470
x0 <- 4.2
y0 <- beta0+beta1*x0

alpha <- 0.05
talpha <- qt(1-0.5*alpha,n-2)
h00 <- 1/n+(x0-mean(x))^2/lxx
y0iv <- c(y0-talpha*sigma*sqrt(1+h00), y0+talpha*sigma*sqrt(1+h00))

y0 <- 9.508+9.747*4.2
y0

asd=talpha*sigma/sqrt(lxx)
asd*asd/lxx
confint(lm)


data<-read.csv("C:/Users/asus/Desktop/9.9.csv",header=T,fileEncoding="GBK")[,-1]
n <- length(data$y)
x <- as.matrix(data[,-7])
y <- data$y
lm9.9 <- lm(y~., data=data)
plot(lm9.9)

