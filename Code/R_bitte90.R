query = "/scratch/vstrobel/drivers/1/1.csv"
template = "/scratch/vstrobel/drivers/1/2.csv"


library(dtw)

alignment <- dtw(query, template, keep = TRUE);