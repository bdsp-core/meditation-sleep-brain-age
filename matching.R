library(readxl)
library(MatchIt)
library(lmtest) #coeftest
library(sandwich) #vcovCL
library(diagis)   # for weighted_se
library(ggplot2)  # for plotting

## load datasets, make sure that all of them contain the following columns
## * SID: subject identifier, "sXX" for meditators, MRN for MGH, User for Dreem
## * Dataset
## * Treat = 1 is meditators; Treat = 0 is others
## * BAI (year)
## * Age (year)
## * Sex (1 is male; 0 is female)
## to do ###### NO for now * AHI_group:0 is AHI < 5 or no sleep apnea; 1 is 15 > AHI > 5, or yes to sleep apnea (???? what about AHI>15)
## to do ######* BMI (kg/m2): actual BMI for control; actual or average BMI for meditators (???? also consider group BMI?)

# load meditation set
# to do # Why it is under "Control" folder? on my computer it is under Dropbox/Code stuff/BA_results_FNRbalanced_filtered
meditation_set <- read_csv("~/Desktop/Control/BA_Dreem_FNRbalanced (3).csv")
meditation_set$Dataset <- 'meditation Dreem3'  # every dataset has a Dataset column to indicate where it comes from
meditation_set$Treat <- 1
names(meditation_set)[names(meditation_set) == "Sex"] <- "Sex_letters"  # need to rename column so we can add correct "Sex column"
meditation_set$Sex <- ifelse(grepl("F", meditation_set$Sex_letters), 0, 1)
# every dataset has a different BAI name, unify them to "BAI", but also make sure don't duplicate BAI
meditation_set <- subset(meditation_set, select = -BAI ) # remove the existing "BAI" first which is the old version
meditation_set <- subset(meditation_set, select = -BA ) # remove the existing "BA" first which is the old version
names(meditation_set)[names(meditation_set) == "robustBAI"] <- "BAI" # rename robustBAI to BAI
names(meditation_set)[names(meditation_set) == "robustBA"] <- "BA" 
#to do ##meditation_set$AHI_group -> as.factor(meditation_set$AHI_group) # make AHI_group into factor


# load MGH_control set
#### TO DO: find correct file and there should be a column for bai and bai_f
control_set_MGH <- read_csv("~/Desktop/Control/BAI_control_loose.csv")
control_set_MGH <- subset(control_set_MGH, select = -BAI_old ) # remove the "BAI_old" first which is the old version
#TODO (??? already done?) limit to Asian only, otherwise have to match on race in the analysis below
names(control_set_MGH)[names(control_set_MGH) == "MRN"] <- "SID"  # rename MRN to SID
control_set_MGH$Dataset <- 'HealthyControl MGH_PSG'
control_set_MGH$Treat <- 0
control_set_MGH <- subset(control_set_MGH, select = -BA ) # remove the existing "BA" first which is the old version
control_set_MGH <- subset(control_set_MGH, select = -BAI ) # remove the existing "BAI" first which is the old version
names(control_set_MGH)[names(control_set_MGH) == "BA_F"] <- "BA"  # rename BA_F to BA
names(control_set_MGH)[names(control_set_MGH) == "BAI_F"] <- "BAI"  # rename BAI_F to BAI
### To do ##control_set_MGH$AHI_group <-  as.factor(control_set_MGH$AHI) # make AHI_group into factor


# load Dreem control set
control_set_Dreem <- read_csv("~/Desktop/Control/BAI_Dreem3K_lr.csv")
names(control_set_Dreem)[names(control_set_Dreem) == "User"] <- "SID"  # rename User to SID
control_set_Dreem$Dataset <- 'ControlDreem3K Dreem2'
control_set_Dreem$Treat <- 0
control_set_Dreem$Sex <- ifelse(grepl("Female",control_set_Dreem$Gender), 0, 1)
# BAI: control_set_Dreem's BAI column is what we want
#control_set_Dreem$AHI_group -> as.factor(control_set_Dreem$AHI_group) # make AHI_group into factor
##### only the insomnia_aid

# load symptomatic/MCI/dementia set
# in your code it is reading from MCI_Dementia_info.csv, which I'm not sure where it comes from.
# the file where I added disease information is BAI_MGH_lr.csv
disease_set <- read_csv("~/Desktop/Control/BAI_MGH_lr.csv")
#TODO limit to Asian only, otherwise have to match on race in the analysis below
names(disease_set)[names(disease_set) == "MRN"] <- "SID"  # rename MRN to SID
disease_set$Dataset <- 'Disease MGH_PSG'
disease_set$Treat <- 0
disease_set <- subset(disease_set, select = -BA ) # remove the existing "BA" first which is the old version
disease_set <- subset(disease_set, select = -BAI ) # remove the existing "BAI" first which is the old version
names(disease_set)[names(disease_set) == "BA_F"] <- "BA"  # rename BA_F to BA
names(disease_set)[names(disease_set) == "BAI_F"] <- "BAI"  # rename BAI_F to BAI
#### to do ##  disease_set$AHI_group -> as.factor(disease_set$AHI_group) # make AHI_group into factor


## concatenate datasets, from here `df` is the dataset you deal with

columns <- c('SID', 'Dataset', 'Treat', 'BAI','BA', 'Age', 'Sex') 
##columns <- c('SID', 'Dataset', 'Treat', 'BAI', 'Age', 'Sex', 'AHI_group', 'BMI') ##### to do fix this
df1 <- rbind(meditation_set[,columns], control_set_MGH[,columns])   # for comparing meditator vs. MGH healthy control
df2 <- rbind(meditation_set[,columns], control_set_Dreem[,columns])  # for comparing meditator vs. Dreem control



## Analysis

# specify estimand here, which is VERY important
# ATE (average treatment effect) means the change in average BAI if everyone was non-meditator vs. everyone was meditator
# ATT (average treatment effect on the treated) means the change in average BAI if all meditators became non-meditators (assuming treat=1 for meditators)
# ATC (average treatment effect on the control) means the change in average BAI if all non-meditators became meditator (assuming treat=1 for meditators)
estimand <- "ATE"
#estimand <- 'ATT'

# Compare meditator vs. MGH healthy control -----------------------------------------------------

#change this
df <- df1

#Check balance prior to matching
myFunction0 <- function(df) {
  m.out0 <- matchit(Treat ~ Age + Sex, data = df,
                  method = NULL, distance = "glm")
  return(m.out0) # pay attention to `Std. Mean Diff.`, are they qualitatively close to 0?
}

#1:1 nearest neighbor matching
myFunction1 <- function(df) {
  m.out1 <- matchit(
    Treat ~ Age + Sex, data = df,
    method = 'nearest', distance = "glm",
    estimand = "ATT")
  return(m.out1)
}

#Full matching
myFunction2 <- function(df, estimand='ATE') {
  m.out2 <- matchit(Treat ~ Age + Sex, data = df, method = "full", 
                    distance = "glm",estimand = estimand)
  return(m.out2)
}

#estimating treatment effect
myFunctionWeights <- function(m.data) {
  fit <- lm(BAI ~ Treat + Age + Sex, data=m.data, weights=weights)
  return(coeftest(fit, vcov. = vcovCL, cluster = ~subclass))
}


m.out0 <- myFunction0(df1, estimand=estimand)
m.out1 <- myFunction1(df1, estimand=estimand)
m.out2 <- myFunction2(df1, estimand=estimand)

plot(m.out2, type = "jitter", interactive = FALSE)
plot(m.out2, type = "qq", interactive = FALSE)
plot(summary(m.out2))

m.data <- match.data(m.out2)
myFunctionWeights(m.data)

#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat0 <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wmean_treat1 <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat0  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat1  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

# visualize result as a bar plot
df_plot <- data.frame(
  groups=c('Healthy controls', 'Meditators'),
  means=c(wmean_treat0, wmean_treat1),
  sems=c(wsem_treat0, wsem_treat1))

gg_plot <- ggplot(df_plot, aes(groups, means)) +  
  geom_bar(stat="identity")+
  geom_errorbar(aes(x=groups, ymin=means, ymax=means+sems))
gg_plot

png(file="~/Desktop/Control/Healthy controls vs Meditators.png",
    width=600, height=350)
gg_plot
dev.off()




# Compare Healthy controls (strict) MGH vs meditatpors --------------------

# load MGH_control set
#### TO DO: find correct file and there should be a column for bai and bai_f
control_set_MGHs <- read_csv("~/Desktop/Control/BAI_control_strict.csv")
control_set_MGHs <- subset(control_set_MGHs, select = -BAI_old ) # remove the "BAI_old" first which is the old version
#TODO (??? already done?) limit to Asian only, otherwise have to match on race in the analysis below
names(control_set_MGHs)[names(control_set_MGHs) == "MRN"] <- "SID"  # rename MRN to SID
control_set_MGHs$Dataset <- 'HealthyControlStrict MGH_PSG'
control_set_MGHs$Treat <- 0
control_set_MGHs <- subset(control_set_MGHs, select = -BA ) # remove the existing "BA" first which is the old version
control_set_MGHs <- subset(control_set_MGHs, select = -BAI ) # remove the existing "BAI" first which is the old version
names(control_set_MGHs)[names(control_set_MGHs) == "BA_F"] <- "BA"  # rename BA_F to BA
names(control_set_MGHs)[names(control_set_MGHs) == "BAI_F"] <- "BAI"  # rename BAI_F to BAI
### To do ##control_set_MGH$AHI_group <-  as.factor(control_set_MGH$AHI) # make AHI_group into factor

df1s <- rbind(meditation_set[,columns], control_set_MGHs[,columns])   # for comparing meditator vs. MGH healthy control


m.out0 <- myFunction0(df1s, estimand=estimand)
m.out1 <- myFunction1(df1s, estimand=estimand)
m.out2 <- myFunction2(df1s, estimand=estimand)

plot(m.out2, type = "jitter", interactive = FALSE)
plot(m.out2, type = "qq", interactive = FALSE)
plot(summary(m.out2))

m.data <- match.data(m.out2)
myFunctionWeights(m.data)

#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat0 <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wmean_treat1 <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat0  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat1  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

# visualize result as a bar plot
df_plot <- data.frame(
  groups=c('MGH Control (STRICT)', 'Meditators'),
  means=c(wmean_treat0, wmean_treat1),
  sems=c(wsem_treat0, wsem_treat1))

gg_plot <- ggplot(df_plot, aes(groups, means)) +  
  geom_bar(stat="identity")+
  geom_errorbar(aes(x=groups, ymin=means, ymax=means+sems))
gg_plot


png(file="~/Desktop/Control/MGH Control (STRICT) vs Meditators.png",
    width=600, height=350)
gg_plot
dev.off()

# Compare meditator vs. Dreem control -------------------------------------------------------------------
#TODO do exactly the same as above, but use df2 instead of df1
#TODO maybe it's better to write the analysis code above as a function, to reduce repeated code


#average ### TO DO

m.out0 <- myFunction0(df2, estimand=estimand)
m.out1 <- myFunction1(df2, estimand=estimand)
m.out2 <- myFunction2(df2, estimand=estimand)

plot(m.out2, type = "jitter", interactive = FALSE)
plot(m.out2, type = "qq", interactive = FALSE)
plot(summary(m.out2))

m.data <- match.data(m.out2)
myFunctionWeights(m.data)

#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat0 <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wmean_treat1 <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat0  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat1  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

# visualize result as a bar plot
df_plot <- data.frame(
  groups=c('Controls: Dreem', 'Meditators'),
  means=c(wmean_treat0, wmean_treat1),
  sems=c(wsem_treat0, wsem_treat1))

gg_plot <- ggplot(df_plot, aes(groups, means)) +  
  geom_bar(stat="identity")+
  geom_errorbar(aes(x=groups, ymin=means, ymax=means+sems))
gg_plot


png(file="~/Desktop/Control/Dreem Controls vs Meditators.png",
    width=600, height=350)
gg_plot
dev.off()

# Compare meditator vs. MGH healthy control vs. MGH no dementia/Sym/MCI/Dementia ------------------------
# df3 contains 6 conditions: meditator, healthy control, no dementia, symptomatic, MCI, dementia

# we can choose a reference condition, e.g. meditator, and then compare
# * meditator vs. healthy control [done above]
# * meditator vs. no dementia
# * meditator vs. symptomatic
# * meditator vs. MCI
# * meditator vs. dementia

# or we can choose no dementia as the reference condition, andd then compare
# * no dementia (treat=0) vs. meditator (treat=1)
# * no dementia (treat=0) vs. healthy control (treat=1)
# * no dementia vs. symptomatic
# * no dementia vs. MCI
# * no dementia vs. dementia

# I think choosing no dementia as the reference condition makes more sense because they represent the most people
# (note that no dementia means they can still be unhealthy otherwise,
# but healthy control means they are healthy in terms of the strictness of inclusion criteria.)

# In this case, we can consider no dementia as treat=0, and the other one as treat=1
# and we consider the estimand to be ATC, which means:
# the change in average BAI if all no-dementia people became meditator/healthy control/symptomatic/MCI/dementia
# the reason we consider ATC, instead of ATE, is that we can only do matching for 2 groups,
# when we use ATE for each comparison, the population is different across comparisons, therefore the ATEs are not comparable
# for ATC, the population of interest is the same across comparisons, therefoer the ATEs are comparable,
# so that all 7 comparisons can be put in one barplot
estimand <- 'ATC'

#for each comparison, change their treat variable
# for example, for no dementia vs. healthy control
disease_set2 <- data.frame(disease_set[,columns]) # first copy the dataset
disease_set2 <- disease_set2[disease_set2$DementiaStage=="No Dementia",] # then get the no dementia subset
control_set_MGH2 <- data.frame(control_set_MGH[,columns]) # first copy the dataset
row.has.na <- apply(disease_set2, 1, function(x){any(is.na(x))})
disease_set2 <- disease_set2[!row.has.na,]

disease_set2$Treat <- 0 # then set treat to 0 (looks like already 0, but still do it)
control_set_MGH$Treat <- 1 # then set treat to 1
df3 <- rbind(disease_set2, control_set_MGH) # combine MGH healthy control and no symptomatic

# get wmean and wsem for no dementia and healthy control from MGH
m.out2 <- myFunction2(df3, estimand=estimand)
m.data <- match.data(m.out2)
#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat_no_dementia <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wmean_treat_healthy_control <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat_no_dementia  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat_healthy_control  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])


columns <- c('SID', 'Dataset', 'Treat', 'BA', 'Age', 'Sex', 'DementiaStage')
#TODO do exactly the same as above, but use df3 instead of df1
disease_set3 <- data.frame(disease_set[,columns]) # first copy the dataset
disease_set3 <- disease_set3[disease_set3$DementiaStage=='Symptomatic',] # symptomatic data
row.has.na <- apply(disease_set3, 1, function(x){any(is.na(x))})
disease_set3 <- disease_set3[!row.has.na,]

disease_set2$Treat <- 0 # then set treat to 0 (looks like already 0, but still do it)
disease_set3$Treat <- 1 # then set treat to 1
df4 <- rbind(disease_set2, disease_set3) # combine no dementia and symptomatic

# get wmean and wsem for no dementia and symptomatic
m.out2 <- myFunction2(df4, estimand=estimand)
m.data <- match.data(m.out2)
#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat_no_dementia_new <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
# check wmean_treat_no_dementia_new == wmean_treat_no_dementia, at least numerically
# if they are equal, this approach is working, 
# so we do not need to assign ..._new, we can either not do it multiple times, or overwrite every time
wmean_treat_symptomatic <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat_no_dementia_new  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat_symptomatic  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

#MCI
disease_set4 <- data.frame(disease_set[,columns]) # first copy the dataset
disease_set4 <- disease_set4[disease_set4$DementiaStage=='MCI',] # MCI
row.has.na <- apply(disease_set4, 1, function(x){any(is.na(x))})
disease_set4 <- disease_set4[!row.has.na,]

disease_set2$Treat <- 0 # then set treat to 0 (looks like already 0, but still do it)
disease_set4$Treat <- 1 # then set treat to 1
df5 <- rbind(disease_set2, disease_set4) # combine no dementia and MCI

# get wmean and wsem for no dementia and MCI
m.out2 <- myFunction2(df5, estimand=estimand)
m.data <- match.data(m.out2)
#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat_no_dementia_anew <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
# check wmean_treat_no_dementia_new == wmean_treat_no_dementia, at least numerically
# if they are equal, this approach is working, 
# so we do not need to assign ..._new, we can either not do it multiple times, or overwrite every time
wmean_treat_MCI <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat_no_dementia_anew  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat_MCI  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])


#Dementia
disease_set5 <- data.frame(disease_set[,columns]) # first copy the dataset
disease_set5 <- disease_set5[disease_set5$DementiaStage=='Dementia',] # dementia
row.has.na <- apply(disease_set5, 1, function(x){any(is.na(x))})
disease_set5 <- disease_set5[!row.has.na,]

disease_set2$Treat <- 0 # then set treat to 0 (looks like already 0, but still do it)
disease_set5$Treat <- 1 # then set treat to 1
df6 <- rbind(disease_set2, disease_set5) # combine no dementia and MCI

# get wmean and wsem for no dementia and MCI
m.out2 <- myFunction2(df6, estimand=estimand)
m.data <- match.data(m.out2)
#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat_no_dementia_aanew <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
# check wmean_treat_no_dementia_new == wmean_treat_no_dementia, at least numerically
# if they are equal, this approach is working, 
# so we do not need to assign ..._new, we can either not do it multiple times, or overwrite every time
wmean_treat_dementia <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat_no_dementia_aanew  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat_dementia  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

#meditator
disease_set2$Treat <- 0 # then set treat to 0 (looks like already 0, but still do it)
meditation_set2$Treat <- 1 # then set treat to 1
df7 <- rbind(disease_set2, meditation_set2) # combine no dementia and MCI

# get wmean and wsem for no dementia and MCI
m.out2 <- myFunction2(df7, estimand=estimand)
m.data <- match.data(m.out2)
#SEM = standard deviation of the mean (Standard Error of Measurement)
wmean_treat_no_dementia_aaanew <- weighted.mean(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
# check wmean_treat_no_dementia_new == wmean_treat_no_dementia, at least numerically
# if they are equal, this approach is working, 
# so we do not need to assign ..._new, we can either not do it multiple times, or overwrite every time
wmean_treat_meditators <- weighted.mean(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])
wsem_treat_no_dementia_aaanew  <- weighted_se(m.data$BAI[m.data$Treat==0], m.data$weights[m.data$Treat==0])
wsem_treat_meditators  <- weighted_se(m.data$BAI[m.data$Treat==1], m.data$weights[m.data$Treat==1])

# visualize result as a bar plot
df_plot <- data.frame(
  groups=c("Meditators","Healthy Control:MGH","No Dementia", "Symptomatic", "MCI","Dementia"),
  means=c(wmean_treat_meditators, wmean_treat_healthy_control, wmean_treat_no_dementia_new, wmean_treat_symptomatic, wmean_treat_MCI,wmean_treat_dementia),
  sems=c(wsem_treat_meditators, wsem_treat_healthy_control,wsem_treat_no_dementia_new, wsem_treat_symptomatic, wsem_treat_MCI,wsem_treat_dementia))

gg_plot <- ggplot(df_plot, aes(groups, means)) +  
  geom_bar(stat="identity")+
  geom_errorbar(aes(x=groups, ymin=means, ymax=means+sems))
gg_plot


#TODO maybe it's better to write the analysis code above as a function, to reduce repeated code

#####final to do is graph
x <- control_set_MGH$Age
y <- control_set_MGH$BA

x1 <- meditation_set$Age
y1 <- meditation_set$BA

x2 <- disease_set2$Age
y2 <- disease_set2$BA

x3 <- disease_set3$Age
y3 <- disease_set3$BA

x4 <- disease_set4$Age
y4 <- disease_set4$BA

x5 <- disease_set5$Age
y5 <- disease_set5$BA
##needs work!!

plot(x, y, col = rgb(0, 0, 0, 0.50), xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main ='BA versus CA: MGH controls')
par(new=T)
plot(x1, y1, pch = 19, col = 'mediumorchid1', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA: meditator')
par(new=T)
plot(x2, y2, pch = 1, xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA: No dementia')
par(new=T)
plot(x3, y3, col = 'darkgreen', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA: Symptomatic')

plot(x4, y4, col = 'orange', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA: MCI')

plot(x5, y5, col = 'red', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA: Dementia')



plot(x, y, col = rgb(0, 0, 0, 0.50), xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main ='BA versus CA')
par(new=T)
plot(x1, y1, pch = 19, col = 'mediumorchid1', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA')
par(new=T)
plot(x2, y2, pch = 1, xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA')
par(new=T)
plot(x3, y3, col = 'darkgreen', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA')
par(new=T)
plot(x4, y4, col = 'orange', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA')
par(new=T)
plot(x5, y5, col = 'red', xlim=c(0,100), ylim=c(0,100), xlab='CA', ylab='BA',main='BA versus CA')

columns <- c('SID', 'Dataset', 'Treat', 'BA', 'Age', 'BAI', 'DementiaStage')
disease_set <- data.frame(disease_set[,columns]) 
meditation_set$DementiaStage <- "Meditatior"
meditation_set<- data.frame(meditation_set[,columns])
datas <- rbind(disease_set,meditation_set)

library("ggplot2")
library("ggsci")



# Change area fill color. JCO palette
ggplot(datas, aes(DementiaStage, BAI)) +
  geom_boxplot(aes(fill = DementiaStage)) +
  scale_fill_jco()+
  theme_classic() +
  theme(legend.position = "top")


# Change point color and the confidence band fill color. 
# Use tron palette on dark theme
ggplot(datas, aes(Age, BA)) +
  geom_point(aes(color = DementiaStage)) +
  geom_smooth(aes(color = DementiaStage, fill = DementiaStage)) + 
  scale_color_tron()+
  scale_fill_tron()+
  theme_dark() +
  theme(
    legend.position = "top",
    panel.background = element_rect(fill = "#2D2D2D"),
    legend.key = element_rect(fill = "#2D2D2D")
  )

