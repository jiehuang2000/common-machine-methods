---
title: "Wine Quality Prediction"
author: "Jie Huang"
date: "11/9/2018"
output: html_document
---

```{r setup, include = FALSE}
knitr::opts_chunk$set(echo = TRUE, cache = TRUE, warning = F, message = F)
options(width = 100)
```

--------------

I am using this wine quality data to demostrate the application of some common Machine Learning tools. The dataset is downloaded from the UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/index.php>. The original dataset is published in the paper: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

The description of this dataset can be obtained from the UCI website. I am copying it here as well:

* Input variables (based on physicochemical tests):

     1 - fixed acidity
   
     2 - volatile acidity
   
     3 - citric acid
   
     4 - residual sugar
   
     5 - chlorides
   
     6 - free sulfur dioxide
   
     7 - total sulfur dioxide
   
     8 - density
   
     9 - pH
   
     10 - sulphates
   
     11 - alcohol
 
* Output variable (based on sensory data): 

     12 - quality (score between 0 and 10)

--------------

There are two questions in this analysis. The first question is to predict wine quality using the input variables. The second question is to specifically classify wines with excellent qualities, which is defined as wines with quality >= 7. 

To solve the first question, I will use the wine quality variable as a continuous variable and build models such as linear regression, random forest, support vector machine to predict wine quality. This part is displayed in Session 1 of this analysis.

To solve the second question, I will transform the quality variable into a binary variable, and build models such as logistic regression, lasso to make classification. This part is displayed in Session 2 of this analysis.

--------------

#Analysis Outlines:

[R environment preparation and customized functions](#envi)

#### [Session 1. For the purpose of prediction:](#sess1)

- [1. Data importing and processing](#data)

- [2. Linear Regression](#lm)

- [3. Outlier/influential point detection](#outlier)

- [4. Polynomial Regression](#pr)

- [5. Categorize continuous covariants(ANOVA)](#cat)

- [6. Variable interaction and variable selection](#varselect)

- [7. Random Forest](#rf)

- [8. Support Vector Machine](#svm)

- [9. Regression Tree](#rt)

- [10. Neural Network](#nn)

- [11. Categorical Y variable](#catY)

#### [Session 2. For the purpose of classification:](#sess2)

- [1. Data Processing and visualization](#data2)

- [2. Logistic Regression](#glm)

- [3. LASSO with Cross Validation](#lass0)

- [4. Decision Tree](#dt)

- [5. Random Forest](#rfclass)

- [6. Support Vector Machine](#svm2)

#### [Conclusion](#conclusion)

#### [Limitations and Discussion](#limit)

--------------

# R environment preparation and customized functions {#envi}
```{r environment preparation}
rm(list = ls())
packages = c("tidyverse", "RCurl", "psych", "stats", 
             "randomForest", "glmnet", "caret","kernlab", 
             "rpart", "rpart.plot", "neuralnet", "C50",
             "doParallel", "AUC")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
invisible(lapply(packages, require, character.only = TRUE))

# customized function to evaluate model performance for continuous predictors
eval = function(pred, true, plot = F, title = "") {
  rmse = sqrt(mean((pred - true)^2))
  mae = mean(abs(pred - true))
  cor = cor(pred, true)
  if (plot == TRUE) {
    par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
    diff = pred - true
    plot(jitter(true, factor = 1), 
         jitter(pred, factor = 0.5), #jitter so that we can see overlapped dots
         pch = 3, asp = 1,
         xlab = "Truth", ylab = "Predicted") 
    abline(0,1, lty = 2)
    hist(diff, breaks = 20, main = NULL)
    mtext(paste0(title, " predicted vs. true using test set"), outer = TRUE)
    par(mfrow = c(1,1))}
  return(list(rmse = rmse,
              mae = mae,
              cor = cor))
}
# customized function to evaluate model performance for binary predictors
eval_class = function(prob, true, plot = F, title = "") {
    # find cutoff with the best kappa
    cuts = seq(0.01, 0.99, by=0.01)
    kappa = c()
    for (cut in cuts){
      cat = as.factor(ifelse(prob >= cut, 1, 0))
      cm = confusionMatrix(cat, true, positive = "1")
      kappa = c(kappa, cm$overall[["Kappa"]])
    }
    opt.cut = cuts[which.max(kappa)]
    
    # make predictions based on best kappa
    pred = as.factor(ifelse(prob >= opt.cut, 1, 0))
    confM = confusionMatrix(pred, true, positive = "1")
    
    # calculate AUC
    roc = roc(as.vector(prob), as.factor(true))
    auc = round(AUC::auc(roc),3)
    
    if (plot==T){
      # plot area under the curve
      par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
      plot(roc, main = "AUC curve"); abline(0,1)
      text(0.8, 0.2, paste0("AUC = ", auc))
      
      # plot confusion matrix
      tab = table(true, pred)
      plot(tab,
           xlab = "Truth",
           ylab = "Predicted",
           main = "Confusion Matrix")
      text(0.9, 0.9, paste0('FN:', tab[2,1]))
      text(0.9, 0.05, paste0('TP:', tab[2,2]))
      text(0.1, 0.9, paste0('TN:', tab[1,1]))
      text(0.1, 0.05, paste0('FP:', tab[1,2]))
      mtext(paste0(title, " predicted vs. true using test set"), outer = TRUE)
      par(mfrow = c(1,1))
      }
    return(list(auc=auc, 
                confusionMatrix = confM))
}
```
--------------

# Session 1. For the purpose of prediction {#sess1}

# 1. Data importing and processing {#data}

```{r data import and processing}
myfile = getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
raw = read.csv(textConnection(myfile), header = T, sep = ";")
n = nrow(raw); p = ncol(raw); dim(raw)
head(raw) # check the first few lines
str(raw) # check the general structure
summary(raw) # check the summary
```

It seems that the dataset is very clean, with no missing data and clear structure. All variables are numeric. The range of independent variables varies greatly, so when building the model I will normalize them to be within the same range.

Next step I will check the pairwise relationship of each variable. As we can see from the below figure, there is not a clear linear relationship between the quality variable and other covariants, indicating that a simple linear regression might not work. In addition, there is some collinearity between covariants. These observations are against the assumption of a linear model.

```{r pairs plot, fig.height = 10, fig.width = 10}
pairs.panels(raw)
```


I will split the dataset into a training and testing set, and normalize each set separately.

```{r data splitting}
set.seed(1)
idx = sample(n, 0.9*n)
train = raw[idx,]; dim(train)
test = raw[-idx,]; dim(test)

# normalize train set so that the range is 0 ~ 1
normalize_train = function(x) (x - min(x))/(max(x) - min(x))
train.norm = data.frame(apply(train[,-p], 2, normalize_train), 
                        quality = train[,p])
summary(train.norm)
# normalize test set using the values from train set to make prediction comparable
train.min = apply(train[,-p], 2, min)
train.max = apply(train[,-p], 2, max)
test.norm = data.frame(sweep(test, 2, c(train.min, 0)) %>% 
                         sweep(2, c(train.max-train.min, 1), FUN = "/"))
summary(test.norm) # test.norm might have data out of range 0~1, since it's normalized against the training set.
```

--------------

# 2. Linear Regression {#lm}

This is not always the best model, but it's okay to start with, so that we have a basic sense of the relationship between the independent variable and dependent variable.

First, I will check the normality of the dependent variable using the Shapiro-Wilk test.

```{r linear regression}
hist(raw$quality)  
shapiro.test(raw$quality) #Didn't pass normality test, so linear model may have a problem
```

The dependent variable doesn't pass the normality test, so one assumption of linear regression is not met. In addition, as we see from the pairwise plot, the relationship among independent variables and dependent variables are not entirely linear. There is also some collinearity among independent variables. Any of those could sabotage the performance of the linear model. 

Then I will apply this linear model to the test set, and visualize the predicted value against true value. I will also evaluate the model performance based on 3 measures: RMSE (root mean square error), MAE (mean absolute error) and cor (correlation). Smaller RMSE, MAE and larger cor are indicators of a good prediction. 

```{r}
tr.lm = lm(quality~., data = train.norm)
summary(tr.lm)
tr.lm.pred = predict(tr.lm, test.norm[,-p])
tr.lm.eval = eval(tr.lm.pred, test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)
```

As we can see above in the model summary, the $R^2$ of the model is only 0.28, so this model doesn't explain much of the variance component. The evaluation of comparison between predicted quality and true quality, RMSE is `r round(tr.lm.eval$rmse,2)` and MAE is `r round(tr.lm.eval$mae,2)`, considering that the quality range is from 0 to 10, this level of difference is not too bad. As we can see from the plot above, the model performs worse in predicting extreme cases, i.e., wines with especially high or low qualities.

To remedy this issue, there are a few thoughts: 

1. I could check if there are many outliers or influential points.

2. For those variables having a non-linear relationship with the y variable, I could try higher order regression, such as quadratic regression. I could also break them into a few intervals and make them categorical variables.

3. I could test the interaction between each pair of the independent variables, especially those pairs with high correlations from the pairs plot.

4. I could use non-linear models, such as random forest (rf), support vector machine (SVM), regression trees (rt), neural network (NN). 

5. For wines with especially high or low qualities, they are relatively rare, so a common model would not put enough weight on them. So I could boost their weight in training the model.

I will try some of these ideas in the following part.

--------------

# 3. Outlier/influential point detection {#outlier}

```{r outlier, fig.height = 10, fig.width = 10}
par(mfrow=c(2,3))
lapply(1:6, function(x) plot(tr.lm, which=x)) %>% invisible()
par(mfrow=c(1,1))
```

From the plots above, the observations #2782, #4746 and #1932 appears as outliers in most of the plots. So I will remove them in the following analysis.

```{r}
rm = c(2782, 4746, 1932)
removed = train.norm[rm, ]
train.norm = train.norm[-rm, ]
tr.lm.rmoutlier = lm(quality~., data = train.norm)
summary(tr.lm.rmoutlier)
```

By only removing 3 observations from more than 4000 observations, the $R^2$ increases from 0.2858 to 0.2862, so the effects of these outliers on the model are considered significant.

# 4. Polynomial Regression {#pr}

I will fit a quadratic regression model by feeding $x + x^2$ into the model for each independent variables. As we can see from the model summary, comparing with the linear model the $R^2$ increases, and RMSE MAE both decreases, but $R^2 = 0.325$ is still not very satisfying.  

``` {r polynomial reg}
# 2nd order regression (quadratic model)
tr.qm = lm(quality~ poly(fixed.acidity, 2) + 
                     poly(volatile.acidity,2) + 
                     poly(citric.acid,2) + 
                     poly(residual.sugar,2) +  
                     poly(chlorides,2) + 
                     poly(free.sulfur.dioxide,2) +
                     poly(total.sulfur.dioxide,2) + 
                     poly(density,2) + 
                     poly(pH,2) + 
                     poly(sulphates,2) + 
                     poly(alcohol,2), 
           data = train.norm)
summary(tr.qm)
tr.qm.pred = predict(tr.qm, test.norm[,-p])
tr.qm.eval = eval(tr.qm.pred, test.norm$quality, plot=T, title="quadratic model: ");unlist(tr.qm.eval)
```

--------------

# 5. Categorize continuous covariants (ANOVA) {#cat}

I will categorize each independent variable according to their value quantiles. I will split each variable into 6 categories:

0:  below 10%
1:  10% to 25%
2:  25% to median
3:  median to 75%
4:  75% to 90%
5:  above 90%

I further fit a multi-level ANOVA model using the new categorical variables. From the model summary below, the model performance is very similar to the quadratic model.

``` {r Categorize continuous covariants}
# categorize covariates by cutoff in the quantiles of c(0.1, 0.25, 0.5, 0.75, 0.9)
low10pct = apply(train.norm, 2, function(x) quantile(x, 0.1))
q1 = apply(train.norm, 2, function(x) quantile(x, 0.25))
q2 = apply(train.norm, 2, function(x) quantile(x, 0.5))
q3 = apply(train.norm, 2, function(x) quantile(x, 0.75))
top10pct = apply(train.norm, 2, function(x) quantile(x, 0.9))

categorize = function(dataset = train.norm) {
  df.cat = dataset
  for (i in 1:(p-1)){
    col = dataset[,i]
    cat = case_when(col<low10pct[i]              ~ "0",
                    col>=low10pct[i] & col<q1[i] ~ "1",
                    col>=q1[i] & col<q2[i]       ~ "2",
                    col>=q2[i] & col<q3[i]       ~ "3",
                    col>=q3[i] & col<top10pct[i] ~ "4",
                    col>=top10pct[i]             ~ "5")
    df.cat[,i] = cat
    }
  return(df.cat)
}
train.cat = categorize(train.norm)
test.cat = categorize(test.norm)
head(train.cat)

tr.cat.lm = lm(quality~., data = train.cat)
summary(tr.cat.lm)
tr.cat.lm.pred = predict(tr.cat.lm, test.cat[,-p])
tr.cat.lm.eval = eval(tr.cat.lm.pred, test.cat$quality, plot=T, title="ANOVA: ");unlist(tr.cat.lm.eval)
```

--------------

# 6. Variable interaction and variable selection {#varselect}

I will further examine the pairwise interactions between independent variables. Considering the relatively large training size and relatively small number of covariants, I can examine their interaction all at once. After feeding all interaction pairs into the model, I will perform variable selection using the stepwise method.

```{r variable interaction}
tr.lm.interract = lm(quality~ .^2, data = train.norm)
summary(tr.lm.interract)

# variable selection using stepwise methods
lm0 = lm(quality ~ 1, data = train.norm)
tr.lm.interract.step = step(lm0, ~ (fixed.acidity + volatile.acidity + 
                      citric.acid + residual.sugar +  chlorides + free.sulfur.dioxide +
                      total.sulfur.dioxide + density + pH + sulphates + alcohol)^2, 
                    direction = "both", trace = 0)

tr.lm.interract.step.pred = predict(tr.lm.interract.step, test.norm[,-p])
tr.lm.interract.step.eval = eval(tr.lm.interract.step.pred, test.norm$quality, plot=T, title="lm wiht interaction and var selection: ");unlist(tr.lm.interract.step.eval)
```

--------------

# 7. Random Forest {#rf}

I will build a random forest (rf) model using the `randomForest` function from the `randomForest` package. This model will ensemble 1000 decision trees, each tree with sqrt(p) = `r round(sqrt(p))` variables selected.

```{r random forest}
tr.rf = randomForest(quality~., data = train.norm, ntree = 1000, mtry = sqrt(p))
tr.rf
tr.rf.pred = predict(tr.rf, test.norm[,-p])
tr.rf.eval = eval(tr.rf.pred, test.norm$quality, plot = T, title = "Random Forest: "); tr.rf.eval
```


I will use the `train` function in the `caret` package to automatically optimize model hyperparameters. Here I am using a 10-fold cross-validation (cv) with 2 repeats, including 2, 4 or 6 variables respectively at each tree level. I am using RMSE to compare model performance. Due to the limitation of computation power, I only choose a few combinations. The available methods in the `train` function can be obtained by typing `names(getModelInfo())`. 

Since this step will be computationally expensive, I will use parallel computation from the `doParallel` package.

```{r rf with cv}
ct = trainControl(method = "repeatedcv", number = 10, repeats = 2)
grid_rf = expand.grid(.mtry = c(2, 3, 6))

set.seed(1)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
tr.cvrf = train(quality~., data = train.norm,
                method = 'rf',
                metric = "RMSE",
                trControl = ct,
                tuneGrid = grid_rf)
stopCluster(cl)
save(tr.cvrf, file = "~/Downloads/wine_train_cvrf.RData")
load(file = "~/Downloads/wine_train_cvrf.RData"); tr.cvrf
tr.cvrf.pred = predict(tr.cvrf, test.norm[,-p])
tr.cvrf.eval = eval(tr.cvrf.pred, test.norm$quality, plot = T, title = "Random Forest with CV: "); unlist(tr.cvrf.eval)
```

A comparison of linear regression, quadratic regression, ANOVA and the random forest is shown in the table below. We can tell that the random forest model performs much better than other models:

```{r results = 'asis'}
knitr::kable(cbind(lm = unlist(tr.lm.eval),
                   quadratic = unlist(tr.qm.eval),
                   ANOVA = unlist(tr.cat.lm.eval), 
                   rf = unlist(tr.rf.eval),
                   rf.cv = unlist(tr.cvrf.eval)) %>% round(3),
             title = "Comparing linear regression, quatratic regression, ANOVA and random forest")
```

--------------

# 8. Support Vector Machine {#svm}

Next I will apply the Support Vector Machine (svm) model using the `ksvm` function from the `kernlab` package. I will use the default `rbfdot` Radial Basis kernel (Gaussian kernel). Other available kernels options can be obtained by typing the `?ksvm` command. 

```{r svm}
tr.svm = ksvm(quality ~ ., 
              data = train.norm, 
              scaled = F,
              kernel = "rbfdot", 
              C = 1)
tr.svm.pred = predict(tr.svm, test.norm[,-p])
tr.svm.eval = eval(tr.svm.pred, test.norm$quality, plot = T, title = "SVM: "); unlist(tr.svm.eval)
```

Similarly, I will use the `train` function to optimize model hyperparameters. This step will be computationally expensive.

```{r svm train}
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
tr <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
set.seed(1)
tr.svmRadial <- train(quality ~.,
                    data = train.norm,
                    method = "svmRadial",
                    trControl=tr,
                    preProcess = NULL,
                    tuneLength = 10)
stopCluster(cl)
save(tr.svmRadial, file="~/Downloads/wine_train_cv_svmRadial.RData")
load(file = "~/Downloads/wine_train_cv_svmRadial.RData"); tr.svmRadial
tr.svmRadial.pred = predict(tr.svmRadial, test.norm[,-p])
tr.svmRadial.eval = eval(tr.svmRadial.pred, test.norm$quality, plot = T, title = "SVM with CV: "); unlist(tr.svmRadial.eval)
```

It seems that svm model with Gaussian kernel doesn't perform as well as the random forest model.

```{r results = 'asis'}
 knitr::kable(cbind(lm = unlist(tr.lm.eval),
                   quadratic = unlist(tr.qm.eval),
                   ANOVA = unlist(tr.cat.lm.eval), 
                   rf = unlist(tr.rf.eval),
                   rf.cv = unlist(tr.cvrf.eval),
                   svm = unlist(tr.svm.eval),
                  svm.cv = unlist(tr.svmRadial.eval)) %>% round(3),
              caption = "")
```

--------------

# 9. Regression Tree {#rt}

Next I will try running a regression using the `rpart` function from the `rpart` package. 

```{r regression tree}
tr.rpart = rpart(quality~., data=train.norm)
rpart.plot(tr.rpart)  
tr.rpart.pred = predict(tr.rpart, test.norm[,-p])
tr.rpart.eval = eval(tr.rpart.pred, test.norm$quality, plot=T); unlist(tr.rpart.eval)
```

--------------

# 10. Neural Network {#nn}

Due to the small training size, I will train a simple neural network with 1 hidden layer and with sigmoid activation function. I am using the `neuralnet` function from the `neuralnet` package.

```{r neural network, fig.height = 10, fig.width = 10}
tr.nn = neuralnet(quality ~ fixed.acidity + volatile.acidity + citric.acid +
                      residual.sugar +  chlorides + free.sulfur.dioxide +
                      total.sulfur.dioxide + density + pH + sulphates + alcohol,
                    data=train.norm,
                    hidden = 2,
                    act.fct = "logistic")
save(tr.nn, file="~/Downloads/wine_train_nn.RData")
load(file="~/Downloads/wine_train_nn.RData")
plot(tr.nn) # for some reason, the plot doesn't show up here, so I am plotting it separately below.
```

```{r nn plot, echo=FALSE, fig.cap="NeuralNet Plot", out.width = '120%'}
knitr::include_graphics("/Users/JieHuang1/Downloads/NN_Rplot.png")
```


```{r nn pred}
tr.nn.pred = compute(tr.nn, test.norm[,-p])
tr.nn.pred = tr.nn.pred$net.result
tr.nn.eval = eval(tr.nn.pred, test.norm$quality, plot=T); unlist(tr.nn.eval)
```


-------------

# 11. Categorical Y variable {#catY}

Considering that the y variable type is `integer` with only less than 10 choices, I will try to treat it as a categorical variable, and fit a random forest model. For simplicity, I will not normalize the datasets.

```{r catY, fig.height = 10, fig.width = 10}
df = raw %>%
  mutate(quality = as.factor(quality))

set.seed(1) #split into train & test set 
idx = createDataPartition(df$quality, p = 0.9, list=F)
train = df[idx,]; table(train$quality)
test = df[-idx,]; table(test$quality)

# Visualizing relationship of the Y variable (quality) and other covariants 
long = gather(train, key = "features", value = "value", 1:(p-1)) 
ggplot(long, aes(x=quality, y= value)) + 
  geom_boxplot(aes(fill = quality)) + 
  coord_flip() +
  facet_wrap(~ features, ncol = 3, scales = "free") +
  labs(title = "Visualizing relationship of the Y variable (quality) and other covariants")

# fit a random forest model
tr.rf.fac= randomForest(quality~., data=train, ntree=1000, mtry=sqrt(p))
tr.rf.fac

# evaluate model performance using the test set
tr.rf.fac.prob = predict(tr.rf.fac, test[,-p])
confM = confusionMatrix(tr.rf.fac.prob, test[,p]); confM
```

```{r Visualize the confusion matrix table}
# Visualize the confusion matrix table
long = confM$table %>% tbl_df()
ggplot(long, aes(Prediction, Reference)) + 
  geom_tile(aes(fill = n), color="black") + 
  scale_fill_gradient(low = "white", high = "steelblue")
```

From the confusion table above, although the overall accuracy is only around 70%, most of the misclassified observations are in a fairly close range with the true quality level. Again, the model doesn't do well when predicting extreme cases: for wines with quality 8, only 6 out of (5 + 6 + 6) = 17 observations are classified correctly; for the 2 wines with quality 3, both of them are classified as quality 6. There are no wines with quality 9 in the test set, due to the rarity of this wine kind, but I could have assigned a few observations with quality 9 in the test dataset to check the performance of the model. 

-------------

Putting all results together in the following table, it seems that the random forest model has the best performance. The best MAE is 0.37 and the best RMSE is 0.51. Considering that wine quality score ranges from 0 to 10, this performance is not acceptable. However, none of these models performs well for those *extreme cases*, i.e., wines with especially high or low quality. To be able to specifically classify wines with superior quality, we will train classification models in the next session. 

```{r results = 'asis'}
 knitr::kable(cbind(lm = unlist(tr.lm.eval),
                    quadratic = unlist(tr.qm.eval),
                    ANOVA = unlist(tr.cat.lm.eval),
                    lm.interac = unlist(tr.lm.interract.step.eval),
                    rf = unlist(tr.rf.eval),
                    rf.cv = unlist(tr.cvrf.eval),
                    svm = unlist(tr.svm.eval),
                    svm.cv = unlist(tr.svmRadial.eval),
                    regression.tree = unlist(tr.rpart.eval),
                    neural.net = unlist(tr.nn.eval)) ,
              caption = "Comparing all models")
```

--------------


# Session 2. For the purpose of classification {#sess2}

# 1. Data Processing and visualization {#data2}

Suppose that my clients have a special requirement to classify wines with quality >=7, defined as *excellent wines*. In this situation, I will build a classification model to identify excellent wines.  

I will make a new binary y variable called "excellent". Since this label is not balanced, i.e., only about 1/5 of the wines are excellent, I will using the `createDataPartition` function to partition the data while balancing the label proportion in the training and testing set. For simplicity, I will not normalize the datasets in this session.

```{r data processing}
df = raw %>% 
      mutate(excellent = ifelse(quality >= 7, "1", "0") %>% as.factor()) %>%
      select(-quality); dim(df)
table(df$excellent)

set.seed(1)
idx = createDataPartition(df$excellent, p = 0.9, list=F)

train.x = df[idx, -p] %>% as.matrix(); dim(train.x)
train.y = df[idx, p]; table(train.y)
train = data.frame(train.x, excellent=train.y)

test.x = df[-idx, -p] %>% as.matrix(); dim(test.x)
test.y = df[-idx, p]; table(test.y)
test = data.frame(test.x, excellent=test.y)
```

The following plot shows the relationship of covariants and the binary Y variables. As we can see from the plots, excellent wines appear to have a smaller variation and fewer outliers for their covariants.

```{r visua2, fig.height = 10, fig.width = 10}
long = gather(train, key = "features", value = "value", 1:(p-1)) 
ggplot(long, aes(x=excellent, y= value)) + 
  geom_boxplot(aes(fill = excellent)) + 
  coord_flip() +
  facet_wrap(~ features, ncol = 3, scales = "free") +
  labs(title = "Visualizing relationship of the Y variable (excellent) and other covariants",
       subtitle = "Note: Wine quality >= 7 is defined to be excellent wine")
```

--------------

# 2. Logistic Regression {#glm}

I will start with a basic logistic regression (glm). 

```{r glm, fig.height = 6, fig.width = 10}
tr.glm = glm(excellent ~. , data = train, family = binomial)
summary(tr.glm)
tr.glm.prob = predict(tr.glm, data.frame(test.x), type="response")
tr.glm.eval = eval_class(tr.glm.prob, test.y, plot=T, title="Logistic Reg: ");tr.glm.eval
```

As we can see from the above model summary and plots, the AUC is `r tr.glm.eval$auc`, which not bad. I chose the cutoff so that the kappa statistic is maximized. The model has an overall accuracy of `r round(tr.glm.eval$confusionMatrix$overall[["Accuracy"]],2)`, sensitivity of `r round(tr.glm.eval$confusionMatrix$byClass[["Sensitivity"]],2)`, precision (ppv) of `r round(tr.glm.eval$confusionMatrix$byClass[["Pos Pred Value"]],2)`, kappa of `r round(tr.glm.eval$confusionMatrix$overall[["Kappa"]],2)`

Note: When choosing the cutoff, I didn't necessarily need to use the cutoff that maximizes the Kappa statistic. The cutoff choice also largely depends on the clients' requirement. For example, suppose clients care more about precision(PPV) than sensitivity, i.e., comparing with missing classifying some excellent wines, classifying non-excellent wines to be excellent wines cost much higher, because clients will waste all the preparation process and end up with a wine type with non-excellent quality. In that case, I will be more stringent, i.e., increase the model cutoff so that I sacrifice sensitivity for the gain of higher precision. 


--------------

# 3. LASSO with Cross Validation {#lasso}

Next step I will build a LASSO model with 10-fold cv using the `cv.glmnet` function from the `glmnet` package. As we can see from the summary below, the LASSO model doesn't out-perform the logistic model. 

```{r lasso, fig.height = 6, fig.width = 10}
set.seed(1)
tr.cvlasso = cv.glmnet(train.x, train.y, 
                       family = "binomial",
                       type.measure = "auc")
coef(tr.cvlasso, s=tr.cvlasso$lambda.1se) # show model coefficients

tr.cvlasso.prob = predict(tr.cvlasso, 
                          test.x, 
                          type="response", 
                          s=tr.cvlasso$lambda.1se)
tr.cvlasso.eval = eval_class(tr.cvlasso.prob, test.y, plot = T, title = "LASSO with CV");tr.cvlasso.eval
```

--------------

# 4. Decision Tree {#dt}

I will build a decision tree model, first a single decision tree, then using boosting. The boosting decision tree out-performs all my pervious models, with a Kappa of 0.59.

```{r decision tree}
#decision tree
tr.dt = C5.0(train.x, train.y)
tr.dt
tr.dt.pred = predict(tr.dt, test.x)
confMat = confusionMatrix(tr.dt.pred, test.y, positive="1")
tr.dt.eval = list(auc = NA, confusionMatrix = confMat); tr.dt.eval
```


```{r decision with boosting}
# using boosting
tr.dtboost = C5.0(train.x, train.y, trials = 10)
tr.dtboost
tr.dtboost.pred = predict(tr.dtboost, test.x)
confMat = confusionMatrix(tr.dtboost.pred, test.y, positive="1")
tr.dtboost.eval = list(auc = NA, confusionMatrix = confMat); tr.dtboost.eval
```

--------------

# 5. Random Forest {#rfclass}

Similar with session 1, I will build a random forest model, and use the `train` function to choose best hyper-parameters. The best kappa from this model reaches 0.64, and overall accuracy of 0.89, which is an improvement compared with the previous models.

```{r rf2}
tr.rfclass = randomForest(excellent~., data=train, ntree=1000, mtry=sqrt(p))
tr.rfclass

tr.rfclass.prob = predict(tr.rfclass, test.x)
confMat = confusionMatrix(tr.rfclass.prob, test.y, positive="1")
tr.rfclass.eval = list(auc = NA,
                       confusionMatrix = confMat); tr.rfclass.eval
```


```{r rfcv2}
ct = trainControl(method = "repeatedcv", number = 10, repeats = 2)
grid_rf = expand.grid(.mtry = c(2, 3, 6))
set.seed(1)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
tr.cvrfclass = train(excellent~., data = train,
                method = 'rf',
                metric = "Kappa",
                trControl = ct,
                tuneGrid = grid_rf)
stopCluster(cl)
save(tr.cvrfclass, file = "~/Downloads/wine_train_cvrfclass.RData")
load(file = "~/Downloads/wine_train_cvrfclass.RData"); tr.cvrfclass
tr.cvrfclass.pred = predict(tr.cvrfclass, test.x)
confMat = confusionMatrix(tr.cvrfclass.pred, test.y, positive="1")
tr.cvrfclass.eval = list(auc = NA,
                         confusionMatrix = confMat); tr.cvrfclass.eval
```

--------------

# 6. Support Vector Machine {#svm2}

SVM model is built in a similar way as in session 1. The model here doesn't perform as well as the random forest model, however, by choosing a different kernel and other hyper-parameters, the model could potentially reach a better performance.

```{r svm2, fig.height = 6, fig.width = 10}
tr.svmclass = ksvm(excellent ~ . , 
                data=train,
                kernel='rbfdot',
                prob.model=T,
                C=1)

tr.svm.pred = predict(tr.svmclass, test.x, type="probabilities")[,2]
tr.svm.eval = eval_class(tr.svm.pred, test.y, plot = T, title = "SVM: "); tr.svm.eval
```

-------------


In the end, I will put all the classification model performance measures in the following table. The same with my conclusion in session 1, the random forest model out-performs all other models, with the highest accuracy of 0.89, and highest kappa of 0.64.

```{r results = 'asis'}
glm = c(auc = tr.glm.eval$auc, 
        tr.glm.eval$confusionMatrix$overall, 
        tr.glm.eval$confusionMatrix$byClass)
cv.lasso = c(auc = tr.cvlasso.eval$auc, 
             tr.cvlasso.eval$confusionMatrix$overall, 
             tr.cvlasso.eval$confusionMatrix$byClass)
decision.tree = c(auc = tr.dt.eval$auc, 
                 tr.dt.eval$confusionMatrix$overall, 
                 tr.dt.eval$confusionMatrix$byClass)
decision.tree.boost = c(auc = tr.dtboost.eval$auc, 
                         tr.dtboost.eval$confusionMatrix$overall, 
                         tr.dtboost.eval$confusionMatrix$byClass)
rf = c(auc = tr.rfclass.eval$auc, 
       tr.rfclass.eval$confusionMatrix$overall, 
       tr.rfclass.eval$confusionMatrix$byClass)
cv.rf = c(auc = tr.cvrfclass.eval$auc, 
          tr.cvrfclass.eval$confusionMatrix$overall, 
          tr.cvrfclass.eval$confusionMatrix$byClass)
svm = c(auc = tr.svm.eval$auc, 
        tr.svm.eval$confusionMatrix$overall, 
        tr.svm.eval$confusionMatrix$byClass)

all = cbind(glm, cv.lasso, 
            decision.tree, decision.tree.boost,
            rf, cv.rf, svm) %>% data.frame()

knitr::kable(all %>% round(3),
             caption = "comparing all models")

all$evaluate = rownames(all)
long = gather(all, key="method", value = "measure", 1:7)
ggplot(long, aes(method, evaluate)) + 
  geom_tile(aes(fill = measure), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1)) + 
  labs(caption = "Visualizing model performance measures")
```


---------

# Conclusion {#conclusion}

After trying many models and optimizing their hyper-parameters, this study reaches a conclusion that random forest model performs the best for prediction and for classification. For predicting wine quality, RF model gives the best MAE of 0.37 and RMSE of 0.51, this is not bad considering that the wine quality could range from 0 to 10. For classifying high-quality wine, RF model gives the best overall accuracy of 0.89, kappa of 0.64, sensitivity of 0.62 and precision of 0.82. 

---------

# Limitations and Discussion {#limit}

The random forest models build in this study performs the best, however, it is still far from perfect. To further improve my model performance, I could further try the following steps.

1. Learn some wine quality control knowledge to have a better sense of the prior knowledge in the wine producing business, so that I might be able to properly transform some features or interpret the interaction between features 

2. Communicate with my clients to know more clearly about their goal: whether to predicting wine quality or to pick up wines with superior quality. Also, try to understand whether they have more tolerance for the type I error or the type II error, so that I can properly set the model cutoff value.

3. To have a closer look at misclassified observations from the confusion matrix (in the prediction case, check observations with large differences between predicted quality and real quality), and try to understand why these wines are classified wrongly.

4. Ask clients if they have more data available, especially for the underrepresented classes, such as wines with high or low quality. Using more data, I could build a more complicated model, such as a neural network with more layers.

5. Ask clients if they have other related features. The model could be improved by adding these features in.

6. If computing power is a limit, I could ask my manager if there is a way to gain more computing power, by either using on-demand cloud computing, or upgrade my computer, or using company computer cluster.


