devtools::install_github("rstudio/keras")
library(keras)
library(reticulate)
library(Metrics)
library(caret)

bm <- read.csv('nls_keras.csv', header = TRUE)

frac<-createDataPartition(bm$BIOMASS, p = .7, list = FALSE, groups = 10)
training<-bm[frac,]
testing<-bm[-frac,]

training<-read.csv('training.csv')
testing<-read.csv('testing.csv')

X_train<-as.matrix(training[,c(2:45)])
Y_train<-as.matrix(training[,1])
X_test<-as.matrix(testing[,c(2:45)])
Y_test<-as.matrix(testing[,1])

model = keras_model_sequential() %>% 
  layer_dense(units=128, activation = "relu", input_shape=44) %>% 
  layer_dense(units=64, activation = "relu") %>%
  layer_dense(units=64, activation = "relu") %>%
  layer_dense(units=1, activation="linear")

model %>% compile(
  loss = "mse",
  optimizer =  "adam", 
  metrics = list("mean_absolute_error")
)

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 40)

model %>% summary()

history<-model %>% fit(X_train,
                       Y_train,
                       epochs = 750,
                       batch_size = 25,
                       validation_data = list(X_test,Y_test),
                       callbacks = list(early_stop, print_dot_callback)
                       )

scores = model %>% evaluate(X_train, Y_train, verbose = 0)
print(scores)

Y_pred = model %>% predict(X_test)
rmse(Y_pred,Y_test)
result = lm(Y_pred~Y_test)
summary(result)

plot(Y_test,Y_pred,pch = 16,cex.axis = 1.2, cex.lab = 1.2, col = "blue",main = "Field vs Model Predicted Biomass\n (SAR and Biophysical Variables)\n", xlim = c(0,700), ylim = c(0,700), xlab = "Field Biomass (t/ha)", ylab = "Predicted Biomass (t/ha)")
abline(1.123e-15, 1)


X_axes = seq(1:length(Y_pred))
plot(X_axes, Y_test, col="red", type="l")
lines(X_axes, Y_pred, col="blue")
legend("topleft", legend=c("y-original", "y-predicted"),
       col=c("red", "blue"), lty=1,cex=0.8)

