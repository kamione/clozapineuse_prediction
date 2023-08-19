# Environment ------------------------------------------------------------------
library(tidyverse)
library(here)
library(haven)
library(flextable)


# Data I/O ---------------------------------------------------------------------
hosp_cluster = c("HK+KE", "KW", "NTW", "NTE")

losocv_performacne_baseline_df <- here(
    "outputs", "tables", "losocv_performacne_baseline.csv"
) %>% 
    read_csv(show_col_types = FALSE)
losocv_performacne_12m_df <- here(
    "outputs", "tables", "losocv_performacne_12m.csv"
) %>% 
    read_csv(show_col_types = FALSE)
losocv_performacne_24m_df <- here(
    "outputs", "tables", "losocv_performacne_24m.csv"
) %>% 
    read_csv(show_col_types = FALSE)
losocv_performacne_36m_df <- here(
    "outputs", "tables", "losocv_performacne_36m.csv"
) %>% 
    read_csv(show_col_types = FALSE)


# replace hospital cluster code for number
losocv_performacne_baseline_df$Site <- hosp_cluster
losocv_performacne_12m_df$Site <- hosp_cluster
losocv_performacne_24m_df$Site <- hosp_cluster
losocv_performacne_36m_df$Site <- hosp_cluster

# 
combined_df <- losocv_performacne_baseline_df %>% 
    bind_cols(losocv_performacne_12m_df %>% select(-Site)) %>% 
    bind_cols(losocv_performacne_24m_df %>% select(-Site)) %>% 
    bind_cols(losocv_performacne_36m_df %>% select(-Site)) 
    
overall_performmance <- combined_df %>%
    select(-Site) %>% 
    colMeans() %>% 
    t() %>% 
    as_tibble() %>% 
    mutate(Site = "Overall", .before = "AUROC...2")

combined_df %>% 
    bind_rows(overall_performmance) %>% 
    mutate(across(-Site, round, 2)) %>% 
    flextable() %>% 
    autofit() %>% 
    bold(part = "header") %>% 
    align(align = "center", part = "all") %>% 
    add_header_row(values = c("Site", "Baseline", "Baseline", "12 Months",
                              "12 Months", "24 Months", "24 Months",
                              "36 Months", "36 Months")) %>% 
    merge_h(part = "header") %>%
    merge_v(part = "header") %>% 
    save_as_docx(path = here("outputs", "tables", "losocv_performacne.docx"))
    
    
