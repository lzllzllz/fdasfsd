##  决策树的生成，以鸢尾花数据集为例

## 加载包
install.packages('party')
library(party)
library(grid)
library(mvtnorm)
library(modeltools)
library(stats4)
library(strucchange)
library(zoo)

## 鸢尾花数据集
iris

##  构建训练集和测试集
train_ord <- sample(nrow(iris),0.7*nrow(iris),replace = FALSE)
train <- iris[train_ord,]
test <- iris[-train_ord,]

#删掉某一列
library(dplyr)
test1<-select(iris,-'Species')


##  构建树
tree <- ctree(Species ~., data = train)

##检测预测值
preTable <- table(predict(tree), train$Species)

##准确率----0.9583333
accurary <- sum(diag(preTable))/sum(preTable)

##画决策树
plot(tree)
plot(tree, type="simple")  ## 简化决策树
## 在测试集上测试决策树

testPred <- predict(tree, newdata = test)
preTatable1<-table(testPred, test$Species)

##准确率------0.9666667
accurary<-sum(diag(preTatable1))/sum(preTatable1)






install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)
train_ord <- sample(nrow(iris),0.8*nrow(iris),replace = FALSE)
train <- iris[train_ord,]
test <- iris[-train_ord,]
fit <- rpart(Species ~ ., data=train,)

train.pred <- predict(fit, train)
table(train$Species == train.pred)['TRUE'] / length(train.pred)
test.pred <- predict(fit, test, type="class") 
table(test$Species == test.pred)['TRUE'] / length(test.pred)

length(predict(fit))

preTable <- table(predict(fit), train$Species)
##准确率----0.9583333
accurary <- sum(diag(preTable))/sum(preTable)
##画决策树
plot(fit)
plot(tree, type="simple")  ## 简化决策树


