#thoughts on the data

##problems
1. data doesn't have same length, need to fill it? (unfortunate because it limits use of pandas/makes it inconvenient)
2. data is randomly rotated -> x,y coordinates are basically useless
3. means are of limited use - slow highway driver has same speed mean as fast "landstraﬂen" driver
4. Dataset is medium-sized/small but home laptops might take significant time to read them in - we sould come up with sanity-check-sets

## solutions
3.: combined means might be of interest, since acceleration and "mean acceleration in curve" and speed are likely to be different for different kinds of drivers.