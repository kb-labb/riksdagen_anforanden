library(dplyr)
library(ggplot2)
library(zoo)

df <- arrow::read_parquet("df_visualization.parquet")
df2 <- arrow::read_parquet("data/df_inference_bleu_eval.parquet")

df <- df2 %>% 
  select(dokid, anforande_nummer, start, duration, debatedate, bleu_score, start_diff, end_diff, start_segment_diff) %>%
  full_join(df, by = c("dokid", "anforande_nummer", "start", "duration"))


df <- df %>%
  mutate(debatedate = coalesce(debatedate.x, debatedate.y)) %>%
  select(-debatedate.x, -debatedate.y)

df <- df %>%
  arrange(debatedate, dokid, anforande_nummer) %>%
  mutate(year = lubridate::year(debatedate),
         month = lubridate::month(debatedate),
         day = lubridate::day(debatedate),
         week = lubridate::week(debatedate))

df <- df %>%
  mutate(bleu_ma_kb = zoo::rollmean(bleu_score.x, k=10, fill=NA, align="right"),
         bleu_ma_riks = zoo::rollmean(bleu_score.y, k=10, fill=NA, align="right"),
         start_diff_ma = zoo::rollmean(start_segment_diff, k=10, fill=NA, align="right"))

df_month <- df %>%
  group_by(year, month) %>%
  summarise(bleu_kb = mean(bleu_score.x, na.rm=TRUE),
            bleu_riks = mean(bleu_score.y, na.rm=TRUE),
            start_diff_kb = mean(start_segment_diff, na.rm=TRUE),
            start_kb = mean(start_diff, na.rm=TRUE),
            end_kb = mean(end_diff, na.rm=TRUE))

df_month$date <- lubridate::make_date(year=df_month$year, month=df_month$month)

start <- lubridate::ymd("2004-01-01")
end <- lubridate::ymd("2022-01-01")
dates <- seq(start, end, by="2 years")

df_month_bleu <- df_month %>%
  tidyr::pivot_longer(cols = starts_with("bleu"), 
                      names_to = "source",
                      values_to = "bleu")

p <- ggplot(df_month_bleu, aes(x=date, y=bleu, color=source)) +
  geom_line(size=0.4, alpha=0.7) +
  # geom_point(color="black", alpha=0.6, stroke=1, size=1.05, shape=21, fill="white") +
  geom_point(alpha=0.6, size=0.6) +
  scale_color_manual(labels = c("KBLab", "The Riksdag"), values=c("firebrick3", "steelblue")) +
  scale_x_date(date_labels = "%Y", breaks=dates, 
               limits = as.Date(c("2003-06-01", "2023-03-01"))) +
  # scale_x_date(guide=guide_axis(n.dodge=2)) +
  theme_minimal(base_size=8) +
  labs(x = "Date",
       y = "BLEU Score",
       title = "Average monthly BLEU score in the Riksdag's speeches",
       subtitle = "We segment and cut the audio files of debates in to speeches based on available metadata.\nAutomatic Speech Recognition (ASR) is performed on each speech. BLEU score here\nis a measure of how well transcripts as predicted by ASR overlap with official transcripts.",
       color = "Metadata source") +
  theme(panel.grid.major.y = element_line(colour="grey60", size=0.3, linetype=2),
        panel.grid.major.x = element_line(colour="grey60", size=0.1, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=5.5))

ggsave("/home/faton/projects/web/kb-labb.github.io/_posts/2023-02-15-finding-speeches-in-the-riksdags-debates/monthly_bleu.jpg", 
       plot = p, width=1900, height=1000, units="px", dpi=300)




df2$debatedate2 <- as.Date(df2$debatedate) 

p <- df2 %>%
  group_by(dokid, debatedate2) %>%
  summarise(end_diff_mean = mean(end_diff),
            start_diff_mean = mean(start_diff)) %>%
  ggplot(aes(x=debatedate2, y=start_diff_mean)) +
  geom_hline(yintercept=0, alpha=0.9, color="palegreen4", linetype=2) +
  geom_point(color="black", alpha=0.8, stroke=0.8, size=0.8, shape=21) + 
  geom_point(size=0.7, alpha=0.7, color="steelblue") + 
  scale_x_date(date_labels = "%Y", breaks=dates, 
               limits = as.Date(c("2003-06-01", "2023-03-01"))) +
  scale_y_continuous(limits = c(-200, 200)) +
  # scale_color_viridis_c(limits = c(-200, 200), option="magma") + 
  # scale_color_distiller(palette = "Spectral", limits=c(-200, 200)) +
  # scale_color_gradient2(limits=c(-200, 200)) + 
  # scale_x_date(guide=guide_axis(n.dodge=2)) +
  theme_minimal(base_size=8) +
  labs(x = "Date",
       y = "Time difference (seconds)",
       # title = "End of speech time difference between KBLab metadata and Riksdag metadata",
       title = "Speech start time difference between KBLab metadata and Riksdag metadata",
       subtitle = "Points represent debates and show the mean difference in a debate.") +
  theme(# panel.background = element_rect(fill = "grey93", linewidth = 0), 
        # plot.background = element_rect(fill = "grey95"),
        panel.grid.major.y = element_line(colour=ifelse(c(-200, -100, 0, 100, 200) == 0, "transparent", "grey60"), 
                                          size=0.3, linetype=2),
        panel.grid.major.x = element_line(colour="grey60", size=0.1, linetype=1),
        panel.grid.minor.x = element_blank(),
        text=element_text(family="Palatino"),
        plot.subtitle=element_text(size=6.5),
        axis.line.y = element_line(arrow = grid::arrow(length = unit(0.25, "cm"), 
                                                       ends = "both")))
p

ggsave("/home/faton/projects/web/kb-labb.github.io/_posts/2023-02-15-finding-speeches-in-the-riksdags-debates/start_diff.jpg", 
       plot = p, width=1600, height=1010, units="px", dpi=300)

