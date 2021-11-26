# All necessary libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)

# Read model inferences using csv format
df <- read.csv('bigbirvtrans')

# Compare two models using rouge metrics
tscore <- qt(0.025/3,length(df$X)/2,lower.tail=F) # T Score with Bonferroni Correction
result <- df%>%group_by(block.sparsity)%>%summarise(mean(rouge1),mean(rouge2),
  mean(rougeL),tscore*sd(rouge1)/sqrt(length(rouge1)),
  tscore*sd(rouge2)/sqrt(length(rouge2)),tscore*sd(rougeL)/sqrt(length(rougeL)))

# Rename
names(result) <- c('Big Bird', 'ROUGE1','ROUGE2','ROUGEL','CI1','CI2','CIL')

# Melt
result <- gather(result, key = 'metric',value = 'mean',factor_key = T, 
  c(ROUGE1:ROUGEL,CI1:CIL))

# Take CI subset out and merge it on left column
result <- cbind(result[0:6,],result[7:12,3])
names(result)<-c('Big Bird', 'metric', 'mean', 'ci')

# Upper and Lower CI
result <- result%>%mutate(upper_ci = mean+ci)%>%mutate(lower_ci = mean-ci)
result$`Big Bird` <- as.character(result$`Big Bird`) # coerce into character for plot
limits <- aes(ymax = upper_ci, ymin = lower_ci)
dodge <- position_dodge(width = 0.9)


jpeg('myplot.jpeg',width = 600, height = 600)
ggplot(result, aes(fill=`Big Bird`, y=mean, x=metric)) + 
  geom_bar(position=dodge, stat="identity", color = 'black')+
  geom_errorbar(limits, position = dodge, width = 0.25)+
  scale_fill_manual(values = c('black','white'))+
  scale_linetype_manual(values = c('black'))+
  theme(panel.background = element_rect(fill = '#ffd966'),
        panel.grid.minor = element_line(color = '#ffd966'),
        panel.grid.major = element_line(color = '#ffd966'),
        plot.background = element_rect(fill = '#ffd966'),
        axis.title.y = element_text(vjust = .5, color='black',size=12),
        legend.position="right",
        legend.text = element_text(color='black', size=12),
        legend.background = element_rect(fill = '#ffd966'),
        axis.text = element_text(colour = 'black',size=12))+
  ylab('Average metric (95% CI with Bonferroni correction)')+
  xlab('')
dev.off()

# Summarize timer
notbigbird<-df%>%select(time.in.seconds, block.sparsity)%>%filter(block.sparsity==0)
compr<-cbind(df%>%select(time.in.seconds, block.sparsity)%>%filter(block.sparsity==1),
  notbigbird)[,c(1,3)]
names(compr)<-c('Big Bird','Transformers (XLNET)')
summary(compr)

# Timer violin plot
forviolin <- df%>%select(time.in.seconds, block.sparsity)
forviolin$block.sparsity <- ifelse(forviolin$block.sparsity == 1,"Big Bird", 
  "Transformers (XLNET)")

jpeg('myplot1.jpeg',width = 600, height = 600)
ggplot(forviolin, aes(block.sparsity, time.in.seconds))+
  geom_violin(fill = 'black')+
  theme(panel.background = element_rect(fill = '#ffd966'),
        panel.grid.minor = element_line(color = '#ffd966'),
        panel.grid.major = element_line(color = '#ffd966'),
        plot.background = element_rect(fill = '#ffd966'),
        axis.title.y = element_text(vjust = .5, color='black',size=12),
        axis.text = element_text(colour = 'black',size=12),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'))+
  ylab('Time per text summarization prediction (in seconds)')+
  xlab('')
dev.off()

# Timer geom point plot
bigbird <- df%>%select(article.word.counts,time.in.seconds,
  block.sparsity)%>%filter(block.sparsity == 1)


jpeg('myplot2.jpeg',width = 600, height = 600)
ggplot(bigbird,aes(x = article.word.counts, y = time.in.seconds))+
  geom_point()+
  geom_smooth(method="auto", se=TRUE, fullrange=FALSE, level=0.95,
        color = 'black', fill = 'black')+
  theme(panel.background = element_rect(fill = '#ffd966'),
        panel.grid.minor = element_line(color = '#ffd966'),
        panel.grid.major = element_line(color = '#ffd966'),
        plot.background = element_rect(fill = '#ffd966'),
        axis.text = element_text(colour = 'black',size=12),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5,size = 16))+
  xlab('Word counts (in article text)')+
  ylab('Time per text summarization prediction (in seconds)')+
  ggtitle('Big Bird')
dev.off()

notbigbird <- df%>%select(article.word.counts,time.in.seconds,
                       block.sparsity)%>%filter(block.sparsity == 0)

jpeg('myplot3.jpeg',width = 600, height = 600)
ggplot(notbigbird,aes(x = article.word.counts, y = time.in.seconds))+
  geom_point()+
  geom_smooth(method="auto", se=TRUE, fullrange=FALSE, level=0.95, 
        color = 'black', fill = 'black')+
  theme(panel.background = element_rect(fill = '#ffd966'),
        panel.grid.minor = element_line(color = '#ffd966'),
        panel.grid.major = element_line(color = '#ffd966'),
        plot.background = element_rect(fill = '#ffd966'),
        axis.text = element_text(colour = 'black',size=12),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5,size = 16))+
  xlab('Word counts (in article text)')+
  ylab('Time per text summarization prediction (in seconds)')+
  ggtitle('Transformers (XLNET)')
dev.off()

