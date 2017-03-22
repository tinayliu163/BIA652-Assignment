

# Dow Jones data analysis
# input data
dji <- read.csv(file = "Users/admin/Desktop/DJA.csv", header = T, skip = 4)
dji$Date <- as.Date(dji$Date, "%m/%d/%Y") # show date as month/day/yea

# pick up first day of each year
dji_yearly <- NULL
date <- dji$Date
for(i in 1:(length(date)-1))
{
    if(format(date[i],"%Y") != format(date[i + 1],"%Y"))
    {
        dji_yearly <- rbind(dji_yearly, dji[i + 1,])
    }
}

# compute change
dji.end <- dji_yearly[-1,2]
dji.start <- dji_yearly[-nrow(dji_yearly),2]
dji.change <- dji.end/dji.start
head(dji.change)

# mean & standard deviation
mean_dji.change <- mean(dji.change)
mean_dji.change

sd_dji.change <- sd(dji.change)
sd_dji.change

# all year over 1,2,3 standard deviation 
year <- format(dji_yearly$Date,"%Y")
year <- year[-1]
dji.change <- data.frame(dji.change)
dji.change <- cbind(year, dji.change)

# over 1 sd
# "l" for lower boundary, "u" for upper boundary
first_dji_l <- subset(dji.change, dji.change <= mean_dji.change - sd_dji.change)
first_dji_u <- subset(dji.change, dji.change >= mean_dji.change + sd_dji.change)
first_dji <- rbind(first_dji_l, first_dji_u)
first_dji <- first_dji[order(first_dji$year),]
first_dji

# over 2 sd
second_dji_l <- subset(dji.change, dji.change <= mean_dji.change - 2 * sd_dji.change)
second_dji_u <- subset(dji.change, dji.change >= mean_dji.change + 2 * sd_dji.change)
second_dji <- rbind(second_dji_l, second_dji_u)
second_dji <- second_dji[order(second_dji$year),]
second_dji


# over 3 sd
third_dji_l <- subset(dji.change, dji.change <= mean_dji.change - 3 * sd_dji.change)
third_dji_u <- subset(dji.change, dji.change >= mean_dji.change + 3 * sd_dji.change)
third_dji <- rbind(third_dji_l, third_dji_u)
third_dji <- third_dji[order(third_dji$year),]
third_dji

# normal probability plot of the sequence of yearly change for Dow index
# This is the plot of norm CDF

hx2	<- pnorm (dji. change $ dji.change, mean_dji. change, sd_dji. change)
plot (dji. change $ dji.change, hx2, xlab =	"",	ylab = "")
# the sequence looks like from a normal distribution

 

