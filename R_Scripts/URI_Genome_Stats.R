# Loading tsv files

list_High = read_tsv('ncbi_labels_higher_performance.tsv', col_names=FALSE)

list_Low = read_tsv('ncbi_labels_lower_performance.tsv', col_names=FALSE)

list_num_reads = read_tsv('num_reads.tsv', col_names=FALSE)

dfGenome = read_tsv('Number_Genomes.tsv')


# Necessary Variables

dict_high = c(list_High)
dict_low = c(list_Low)

num_high_genomes = dfGenome$Num_Genomes[dfGenome$Species_label 
                                        %in% c(as.numeric(unlist(list_High)))]
num_low_genomes = dfGenome$Num_Genomes[dfGenome$Species_label 
                                       %in% c(as.numeric(unlist(list_Low)))]
# Initial statistics

mean_high = mean(num_high_genomes)
mean_low = mean(num_low_genomes)

min_low = min(num_low_genomes)
min_high = min(num_high_genomes)

max_high = max(num_high_genomes)
max_low = max(num_low_genomes)

# Create a temporary dataframe 

plot_high_df = data.frame(
  performance = "high",
  num_genomes = num_high_genomes
  
)

plot_low_df = data.frame(
  performance = "low",
  num_genomes = num_low_genomes
  
)


# create Boxplot using ggplot2

plot_df = rbind(plot_low_df, plot_high_df)


png(file = "boxplot.png")

ggplot(plot_df, aes(x= performance, y= num_genomes)) +
geom_boxplot() +
labs(title= "High vs Low Performance",
     subtitle= "TL-TODA vs Kraken2",
     x= "Performance",
     y = "Number of Genomes") 


dev.off()

sum(list_num_reads[3])

