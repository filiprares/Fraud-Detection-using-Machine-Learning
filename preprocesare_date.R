library(dplyr)
library(lubridate)

# importul setului de date inițial
data <- read.csv("~/Desktop/licenta/csv_licenta/fraud1.csv")


data$trans_date_trans_time <- ymd_hms(data$trans_date_trans_time)


# grupare pe ore
data$hour_group <- hour(data$trans_date_trans_time) %/% 6 + 1

data$dob <- ymd(data$dob)
data$age <- as.numeric(difftime(Sys.Date(), data$dob, units = "weeks")) %/% 52
data$age_group <- cut(data$age, breaks = c(0, 18, 35, 50, 65, Inf), labels = c(1, 2, 3, 4, 5), right = FALSE)

data$age_group <- as.numeric(data$age_group)
data$hour_group <- as.numeric(data$hour_group)

# folosim haversine pt a calcula distanta
haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371  # raza pamant
  dlat <- (lat2 - lat1) * pi / 180
  dlon <- (lon2 - lon1) * pi / 180
  a <- sin(dlat / 2) * sin(dlat / 2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dlon / 2) * sin(dlon / 2)
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R * c
}

data$distance_to_merchant <- mapply(haversine, data$lat, data$long, data$merch_lat, data$merch_long)

# grupare dupa 4 categorii de populatie
data$city_pop_category <- cut(data$city_pop, breaks = c(0, 10000, 50000, 100000, Inf), labels = c(1, 2, 3, 4), right = FALSE)
data$city_pop_category <- as.numeric(data$city_pop_category)

# adaugam weekday ca numar
data$weekday <- wday(data$trans_date_trans_time, label = TRUE)
data$weekday <- as.numeric(data$weekday)

data$category <- as.numeric(as.factor(data$category))
data$gender <- as.numeric(as.factor(data$gender))

# logaritmarea amt pt o mai buna distributie
data$amt <- log(data$amt + 1)

# selectam coloane finale
ds_fraud <- data %>%
  select(category, amt, weekday, gender, distance_to_merchant, is_fraud, age_group, hour_group, city_pop_category)

# exportam setul
write.csv(ds_fraud, "~/Desktop/proiect big_data/fraud.csv", row.names = FALSE)
