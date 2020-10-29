library(caret)
library(raster)
library(Metrics)
library(keras)
library(reticulate)
library(vip)

#Reading Plot AGB data along with related predictor variables
mydata<-read.csv('DL_var_imp_plot_data.csv')

#Sub-setting the data for model training and model testing
frac<-createDataPartition(mydata$BM, p = .7, list = FALSE, groups = 10)
training<-mydata[frac,]
testing<-mydata[-frac,]

#Separating the training and testing data to response (Y) and predictor variables (x)
X_train<-as.matrix(training[,c(2:4)])
Y_train<-as.matrix(training[,1])
X_test<-as.matrix(testing[,c(2:4)])
Y_test<-as.matrix(testing[,1])

#Deep Learning Model
model = keras_model_sequential() %>% 
  layer_dense(units=8, activation = "relu", input_shape=3) %>% 
  layer_dense(units=4, activation = "relu") %>%
  layer_dense(units=1, activation="linear")

model %>% compile(
  loss = "mse",
  optimizer =  "adam", 
  metrics = list("mean_absolute_error")
)

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 40)

model %>% summary()

history<-model %>% fit(X_train,
                       Y_train,
                       epochs = 750,
                       batch_size = 25,
                       validation_data = list(X_test,Y_test),
                       callbacks = list(early_stop),
                       verbose = 0
)
options(scipen=5)

#Plotting the model and validation loss
loss1<-sqrt(history$metrics$loss)
plot(loss1, main="Model Loss", xlab = "epoch", ylab="loss(RMSE)",  ylim = c(0,400), col="blue", lwd=2, type="l")

val_loss1<-sqrt(history$metrics$val_loss)
lines(val_loss1, col="red", lwd=2, type="l")

legend("topright", c("train","test"), col=c("blue", "red"), lwd=2, lty=c(1,1))

scores = model %>% evaluate(X_train, Y_train, verbose = 0)
print(scores)

#Variable importance plot
pred <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

p1 <- vip(
  model,                     
  method = "permute",                
  num_features = ncol(X_test),       
  pred_wrapper = pred,         
  train = as.data.frame(X_test) ,   
  target = Y_test,                  
  metric = "RMSE"               
  progress = "text"               
)

# display plot
print(p1)  

#New data prediction and model validation
Y_pred = model %>% predict(X_test)
rmse(Y_pred,Y_test)
result = lm(Y_pred~Y_test)
summary(result)

plot(Y_test,Y_pred,pch = 16,cex.axis = 1.2, cex.lab = 1.2, col = "blue",main = "Field vs Model Predicted Biomass\n (29 November-05 December)\n", xlim = c(0,700), ylim = c(0,700), xlab = "Field Biomass (t/ha)", ylab = "Predicted Biomass (t/ha)")
abline(1.123e-15, 1)

#Predicting AGB map
bhk <- as.matrix(rasterToPoints(stack('BHK_Stack_17_23D.tif')))
bhk_pred = bhk[,3:5]
B <- model %>% predict(bhk_pred)
B_n<-cbind.data.frame(bhk[,1],bhk[,2],B)
BM_raster<-rasterFromXYZ(B_n,crs='+proj=utm +zone=45')
writeRaster(BM_raster,'Biomass_DL_17_23D.tif',overwrite=TRUE)