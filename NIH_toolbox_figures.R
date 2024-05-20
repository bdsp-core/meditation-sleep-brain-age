library(readxl)
library(tidyverse)
library(gghalves)

#read files
#named dataset as "sample"
#sample <- read_excel("C:/Users/hibi9/OneDrive - Beth Israel Lahey Health/NIH_toolbox_MED_pre_post_fixed.xlsx")
#sample <- read_excel("C:/Users/horui/OneDrive/OneDrive - Beth Israel Lahey Health/NIH_toolbox_MED_pre_post_fixed.xlsx")
sample <- read_excel("C:/Users/shariri1/Desktop/BAI Graph generation/NIH_toolbox_MED_pre_post_fixed.xlsx")


#filter data and named as sample 2
sample2 <- sample %>% filter(Inst == "Cognition Crystallized Composite v1.1") 
sample2 <- sample %>% filter(Inst == "NIH Toolbox List Sorting Working Memory Test Age 7+ v2.1") 
sample2 <- sample %>% filter(Inst == "NIH Toolbox Oral Reading Recognition Test Age 3+ v2.1")
#S30 pre has Int name "NIH Toolbox Oral Reading Recognition Test Age 3+ Instructions v2.1", that you might want to rename it in your excel


sample2 <- sample %>% filter(Inst == "NIH Toolbox Picture Sequence Memory Test Age 8+ Form A v2.1")
#sample2 <- sample %>% filter(Inst == "NIH Toolbox Picture Sequence Memory Test Age 8+ Form B v2.1")


#S14 has Post has Int name "NIH Toolbox Picture Vocabulary Test Age 3+ Practice v2.1" that you may need to rename it in your excel
sample2 <- sample %>% filter(Inst == "NIH Toolbox Picture Vocabulary Test Age 3+ v2.1")


sample2 <- sample %>% filter(Inst == "NIH Toolbox Emotional Support FF Age 18+ v2.0") 
sample2 <- sample %>% filter(Inst == "NIH Toolbox Friendship FF Age 18+ v2.0") 
sample2 <- sample %>% filter(Inst == "NIH Toolbox General Life Satisfaction CAT Age 18+ v2.0") 
sample2 <- sample %>% filter(Inst == "NIH Toolbox Instrumental Support FF Age 18+ v2.0") 
#sample2 <- sample %>% filter(Inst == "NIH Toolbox Positive Affect CAT Age 18+ v2.0")
#sample2 <- sample %>% filter(Inst == "NIH Toolbox Positive Affect FF Age 18+ v2.0")
sample2 <- sample %>% filter(Inst == "NIH Toolbox Positive Affect")
sample2 <- sample %>% filter(Inst == "NIH Toolbox Loneliness FF Age 18+ v2.0")
sample2 <- sample %>% filter(Inst == "NIH Toolbox Perceived Stress FF Age 18+ v2.0")






#setting order of assessment name (pre -> post)
sample2$`Assessment Name` <- factor(sample2$`Assessment Name`, levels = c("Pre", "Post"))

#keep ggplot statement
ggplot(sample2, aes(x = `Assessment Name`, y = `Fully-Corrected T-score`)) +
  stat_boxplot(geom = "errorbar", width=0.1, size=1) + # to add whiskers
  
  geom_boxplot(aes(fill = `Assessment Name`), alpha = 1, width=0.3, size = 1) +
  
  #scale_fill_manual(values = c("#0099f8", "#e74c3c")) + to set determined colors
  
  # #line connecting same participants' IDs
  #  geom_line(aes(group = SID), color = "black", linetype = "dashed", size = 0.75) +
  
  #  #dot plot for Pre
  #  geom_point(
  #    data = sample2 %>% filter(`Assessment Name` == "Pre"),
  #    aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), color = "#F8766D", size = 2
  #  ) +

# #dot plot for Post
#  geom_point(
#    data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
#    aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), color = "#00BFC4", size = 2
#  ) +

#violin plot for Pre
#side for left "l" and right "r", position_nudge to adjust position (min -1 to max 1), 
#fill for color, alpha for opacity
# geom_half_violin(
#  data = sample2 %>% filter(sample2$`Assessment Name` == "Pre"),
#   aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), side = "l",
#   position = position_nudge(x = -.5), fill = "#F8766D", alpha = .3
# ) +

#violin plot for Post
# geom_half_violin(
#   data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
#   aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), side = "r",
#   position = position_nudge(x = .5), fill = "#00BFC4", alpha = .3
# ) +

#remove grey background
theme_classic() + 
  
  #adding fixed limits in graph, changing ticks, changing titles of y
  #you can do scale_x_discrete for x variable
  scale_y_continuous(name = "Fully-Corrected T-score",
                     limits=c(0, 100),
                     breaks =  c(0,20,40,60,80,100)) +
  
  #adding xlab
  xlab(sample2$Inst) + 
  
  ##adding title \n to change line
  #ggtitle("TITLE\nTITLE") + 
  
  #changing legend titles
  labs(fill = "PRE vs. POST", face="bold") + 
  
  
  #changing the color, the size, and face (bold/italic) of labels
  theme(
    
    #for title
    plot.title = element_text(color="black", size=14, face="bold.italic"),
    
    #for x lab
    axis.title.x = element_text(color="black", size=14, face="bold"),
    
    #for x tick 
    axis.text.x = element_text(color="black", size=14, face="bold"),
    
    #for y lab
    axis.title.y = element_text(color="black", size=14, face="bold"),
    
    #for y tick 
    axis.text.y = element_text(color="black", size=14, face="bold"),
    
    #panel border
    panel.border = element_rect(colour = "black", fill= NA, size= 1)
  )











#setting order of assessment name (pre -> post)
sample2$`Assessment Name` <- factor(sample2$`Assessment Name`, levels = c("Pre", "Post"))

#keep ggplot statement
ggplot(sample2, aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`)) +
 stat_boxplot(geom = "errorbar", width=0.1, size=1) + # to add whiskers
  
  geom_boxplot(aes(fill = `Assessment Name`), alpha = 1, width=0.3, size = 1) +
  
  #scale_fill_manual(values = c("#0099f8", "#e74c3c")) + to set determined colors
  
 # #line connecting same participants' IDs
#  geom_line(aes(group = SID), color = "black", linetype = "dashed", size = 0.75) +
  
#  #dot plot for Pre
#  geom_point(
#    data = sample2 %>% filter(`Assessment Name` == "Pre"),
#    aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), color = "#F8766D", size = 2
#  ) +
  
 # #dot plot for Post
#  geom_point(
#    data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
#    aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), color = "#00BFC4", size = 2
#  ) +
  
  #violin plot for Pre
  #side for left "l" and right "r", position_nudge to adjust position (min -1 to max 1), 
  #fill for color, alpha for opacity
 # geom_half_violin(
 #  data = sample2 %>% filter(sample2$`Assessment Name` == "Pre"),
 #   aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), side = "l",
 #   position = position_nudge(x = -.5), fill = "#F8766D", alpha = .3
 # ) +
  
  #violin plot for Post
 # geom_half_violin(
 #   data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
 #   aes(x = `Assessment Name`, y = `Age-Corrected Standard Score`), side = "r",
 #   position = position_nudge(x = .5), fill = "#00BFC4", alpha = .3
 # ) +
  
  #remove grey background
  theme_classic() + 
  
  #adding fixed limits in graph, changing ticks, changing titles of y
  #you can do scale_x_discrete for x variable
  scale_y_continuous(name = "Age-Corrected Standard Score",
                    limits=c(75, 150),
                    breaks =  c(75,100,125,150)) +

  #adding xlab
  xlab(sample2$Inst) + 
  
  ##adding title \n to change line
  #ggtitle("TITLE\nTITLE") + 
  
  #changing legend titles
  labs(fill = "PRE vs. POST", face="bold") + 
  
  
  #changing the color, the size, and face (bold/italic) of labels
  theme(
    
    #for title
    plot.title = element_text(color="black", size=14, face="bold.italic"),
    
    #for x lab
    axis.title.x = element_text(color="black", size=14, face="bold"),
    
    #for x tick 
    axis.text.x = element_text(color="black", size=14, face="bold"),
    
    #for y lab
    axis.title.y = element_text(color="black", size=14, face="bold"),
    
    #for y tick 
    axis.text.y = element_text(color="black", size=14, face="bold"),
    
    #panel border
    panel.border = element_rect(colour = "black", fill= NA, size= 1)
  )













###displaying Tscore

#setting order of assessment name (pre -> post)
sample2$`Assessment Name` <- factor(sample2$`Assessment Name`, levels = c("Pre", "Post"))

#keep ggplot statement
ggplot(sample2, aes(x = `Assessment Name`, y = `TScore`)) +
  stat_boxplot(geom = "errorbar", width=0.1, size=1) + # to add whiskers
  
  geom_boxplot(aes(fill = `Assessment Name`), alpha = 1, width=0.3, size = 1) +
  
  
 # #line connecting same participants' IDs
 # geom_line(aes(group = SID), color = "black", linetype = "dashed", size = 0.75) +
  
 # #dot plot for Pre
 # geom_point(
#    data = sample2 %>% filter(`Assessment Name` == "Pre"),
#    aes(x = `Assessment Name`, y = `TScore`), color = "#F8766D", size = 2
 # ) +
  
#  #dot plot for Post
#  geom_point(
#    data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
#    aes(x = `Assessment Name`, y = `TScore`), color = "#00BFC4", size = 2
#  ) +
  
  #violin plot for Pre
  #side for left "l" and right "r", position_nudge to adjust position (min -1 to max 1), 
  #fill for color, alpha for opacity
 # geom_half_violin(
#  data = sample2 %>% filter(sample2$`Assessment Name` == "Pre"),
#    aes(x = `Assessment Name`, y = `TScore`), side = "l",
#    position = position_nudge(x = -.5), fill = "#F8766D", alpha = .3
#  ) +
  
 # #violin plot for Post
#  geom_half_violin(
#    data = sample2 %>% filter(sample2$`Assessment Name` == "Post"),
#    aes(x = `Assessment Name`, y = `TScore`), side = "r",
#    position = position_nudge(x = .5), fill = "#00BFC4", alpha = .3
#  ) +
  
  #remove grey background
  theme_classic() + 
  
  #adding fixed limits in graph, changing ticks, changing titles of y
  #you can do scale_x_discrete for x variable
  scale_y_continuous(name = "TScore", 
                     limits=c(20, 80),
               breaks =  c(10,20,30,40,50,60,70,80)) +
  
  #adding xlab
  xlab(sample2$Inst) + 
  
  #adding title \n to change line
  #ggtitle("TITLE\nTITLE") + 
  
  #changing legend titles
  labs(fill = "PRE vs. POST") + 
  
  
  
  #changing the color, the size, and face (bold/italic) of labels
  theme(
    
    #for title
    plot.title = element_text(color="black", size=14, face="bold.italic"),
    
    #for x lab
    axis.title.x = element_text(color="black", size=14, face="bold"),
    
    #for x tick 
    axis.text.x = element_text(color="black", size=14, face="bold"),
    
    #for y lab
    axis.title.y = element_text(color="black", size=14, face="bold"),
    
    #for y tick 
    axis.text.y = element_text(color="black", size=14, face="bold"),
    
    #panel border
    panel.border = element_rect(colour = "black", fill= NA, size= 1)
  )





#you may want to refer this website for some visual examples (www.sthda.com is good resource)
#http://www.sthda.com/english/wiki/ggplot2-title-main-axis-and-legend-titles
#http://www.sthda.com/english/wiki/ggplot2-axis-ticks-a-guide-to-customize-tick-marks-and-labels#customize-a-continuous-axis

