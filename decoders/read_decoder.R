setwd("~/Dropbox/00 Oxford/29_piVAE_decoders/R")

# Load the jsonlite package
library(jsonlite)

# Read the JSON string from the file
json_str = readLines("../pi-vae/decoders/gp1d_ls03.json")

# Convert the JSON string to a list
stan_data = fromJSON(json_str)

str(stan_data)

