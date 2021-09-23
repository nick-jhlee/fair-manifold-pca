dirloc  <- 'pca'
setwd(dirloc)
a0 = read.csv( file='exp_vars_test.csv', sep=',', header=F)
b0 = read.csv( file='mmds_test.csv', sep=',', header=F)

dirloc  <- '../fpca'
setwd(dirloc)
a1 = read.csv( file='exp_vars_test.csv', sep=',',header=F)
b1 = read.csv( file='mmds_test.csv', sep=',',header=F)
c1 = read.csv( file='runtimes.csv', sep=',',header=F)

dirloc  <- '../mbfpca_4'
setwd(dirloc)
a2 = read.csv( file='exp_vars_test.csv', sep=',',header=F)
b2 = read.csv( file='mmds_test.csv', sep=',',header=F)
c2 = read.csv( file='runtimes.csv', sep=',',header=F)

dirloc <- '../'
setwd(dirloc)


WData = matrix(1:(10*9*3*3),10*9*3,3)
WData[,1] = rep(c('PCA','FPCA','MbF-PCA'), each=90)
WData[,2] = rep( rep(c(2:10)*10,10), 3)
WData[,3] = c(as.vector(unlist(a0[,1:9])), 
              as.vector(unlist(a1[,1:9])), 
              as.vector(unlist(a2[,1:9])))
df = data.frame(Var.exp=as.numeric(WData[,3]),dimension=WData[,2],
                Label=WData[,1])
df$Label <- factor(df$Label, levels=c('PCA','FPCA','MbF-PCA'))
# df[,2] = factor(df[,2], as.character(df[1:9,2]))


require(ggplot2)
p_varexp = ggplot(data = df, aes(x=dimension, y=Var.exp)) +
  geom_boxplot(aes(fill=Label)) +
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 14),
    axis.title.y = element_text(size = 16))


WData = matrix(1:(10*9*3*3),10*9*3,3)
WData[,1] = rep(c('PCA','FPCA','MbF-PCA'), each=90)
WData[,2] = rep( rep(c(2:10)*10,10), 3)
WData[,3] = c(as.vector(unlist(b0[,1:9])), 
              as.vector(unlist(b1[,1:9])), 
              as.vector(unlist(b2[,1:9])))
df = data.frame(Squared_MMD=as.numeric(WData[,3]),dimension=WData[,2],
                Label=WData[,1])
df$Label <- factor(df$Label, levels=c('PCA','FPCA','MbF-PCA'))

df[,2] = factor(df[,2], as.character(df[1:9,2]))
require(ggplot2)
p_mmd = ggplot(data = df, aes(x=dimension, y=Squared_MMD)) +
  geom_boxplot(aes(fill=Label)) +
  theme(
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 14),
    axis.title.y = element_text(size = 16))


dims <- c(20, 30, 40, 50, 60, 70, 80, 90, 100)
fpca_time <- colMeans(c1)
stfpca_time <- colMeans(c2)
stfpca_time <- stfpca_time[1:9]
plot(dims, fpca_time, frame=FALSE, pch = 19, type="b", col="red", xlab="dimension", ylab="runtime", cex=1.3, cex.lab=1.5)
lines(dims, stfpca_time, pch=19, col="blue", type="b", cex=1.3)
legend("topleft", legend=c("FPCA", "MbF-PCA"), col=c("red", "blue"), lty=1:2)
