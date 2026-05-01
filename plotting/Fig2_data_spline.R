library(mgcv)
library(readxl)
library(dplyr)

df_raw <- read_excel(
  "/Volumes/UCL/论文工作/空气污染/清华城市健康指数-健康服务.xlsx",
  sheet = "earlypeak_NZ_CL"
)

df <- df_raw %>%
  filter(HS_type != 0) %>% #change HS_type to HS_type2 to delete the cities withou HS score ()
  select(HS, delta_mort, delta_mortratio) %>%
  mutate(across(everything(), as.numeric)) %>%
  filter(!is.na(HS))

run_gam <- function(data, yvar, sample_label) {
  m  <- gam(reformulate("s(HS, k=20)", response = yvar),
            data = data, method = "REML")
  xg <- seq(min(data$HS), max(data$HS), length.out = 300)
  pr <- predict(m, newdata = data.frame(HS = xg), se.fit = TRUE)
  st <- summary(m)$s.table
  data.frame(
    HS      = xg,
    fit     = as.numeric(pr$fit),
    lo      = as.numeric(pr$fit - 1.96 * pr$se.fit),
    hi      = as.numeric(pr$fit + 1.96 * pr$se.fit),
    edf     = round(st[1, "edf"], 2),
    p_val   = st[1, "p-value"],
    yvar    = yvar,
    sample  = sample_label
  )
}

out <- bind_rows(
  run_gam(df,                    "delta_mort",      "full"),
  run_gam(df %>% filter(HS<=90), "delta_mort",      "sub"),
  run_gam(df,                    "delta_mortratio",  "full"),
  run_gam(df %>% filter(HS<=90), "delta_mortratio",  "sub")
)

write.csv(out,          "/Users/shirley/Desktop/plots_V2/gam_pred.csv",  row.names=FALSE)
write.csv(df["HS"],     "/Users/shirley/Desktop/plots_V2/gam_rug.csv",   row.names=FALSE)
cat("Done\n")