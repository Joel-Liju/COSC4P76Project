

stats_file = read.csv('stats.csv')


labels = c('Comp', 'Comp2019', 'Comp2020', 'Trump', 'Trump2019', 'Trump2020','Tru', 'Tru2019',
           'Tru2020', 'EMA', 'EMA2019', 'EMA2020')

boxplot(accuracy~file, data = stats_file, notch = TRUE, names=labels,
        cex.lab=1.8, cex.axis=1.4, cex.main=1.6, cex.sub=1.6,
        xlab = 'Experiment', ylab = 'Accuracy')

shapiro.test(stats_file$accuracy)

kruskal.test(accuracy~as.factor(file), data = stats_file)

pairwise.wilcox.test(stats_file$accuracy, stats_file$file,
                     p.adjust.method = "BH")


aggregate(stats_file$accuracy, by=list(stats_file$file), FUN=median)




first_group = stats_file[stats_file$file == "complete.csv" 
                      | stats_file$file == "complete2019.csv"
                      | stats_file$file == "complete2020.csv"
                      | stats_file$file == "NoTrudeau.csv"
                      | stats_file$file == "NoTrudeau2019.csv"
                      | stats_file$file == "NoTrudeau2020.csv"
                      | stats_file$file == "NoTrump.csv",] 

kruskal.test(accuracy~as.factor(file), data = first_group)

                      