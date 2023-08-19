# Environment ------------------------------------------------------------------
library(here)
library(tidyverse)
library(ggpubr)
library(ggthemes)
library(flextable)
library(patchwork)

# load internal functions
source(here("src", "R", "stats.R"))


# Data I/O ---------------------------------------------------------------------
auroc_df <- here('outputs', 'tables', 'performance-type_auroc.csv') %>% 
    read_csv(show_col_types = FALSE)
bs_df <- here('outputs', 'tables', 'performance-type_brierscore.csv') %>% 
    read_csv(show_col_types = FALSE)

auroc_top10_df <- here('outputs', 'tables', 'performance-type_auroc-desc_top10.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 10")
auroc_top15_df <- here('outputs', 'tables', 'performance-type_auroc-desc_top15.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 15")
auroc_top20_df <- here('outputs', 'tables', 'performance-type_auroc-desc_top20.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 20")

bs_top10_df <- here('outputs', 'tables', 'performance-type_brierscore-desc_top10.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 10")
bs_top15_df <- here('outputs', 'tables', 'performance-type_brierscore-desc_top15.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 15")
bs_top20_df <- here('outputs', 'tables', 'performance-type_brierscore-desc_top20.csv') %>% 
    read_csv(show_col_types = FALSE) %>% 
    mutate(feature = "Top 20")


performance_cutoff_baseline <- here(
    'outputs', 'tables', 'performance_riskcutoff_baseline.csv'
) %>%
    read_csv(show_col_types = FALSE) %>% 
    mutate(group = "Baseline")
performance_cutoff_12m <- here(
    'outputs', 'tables', 'performance_riskcutoff_12m.csv'
) %>%
    read_csv(show_col_types = FALSE) %>% 
    mutate(group = "12 Months")
performance_cutoff_24m <- here(
    'outputs', 'tables', 'performance_riskcutoff_24m.csv'
) %>%
    read_csv(show_col_types = FALSE) %>% 
    mutate(group = "24 Months")
performance_cutoff_36m <- here(
    'outputs', 'tables', 'performance_riskcutoff_36m.csv'
) %>%
    read_csv(show_col_types = FALSE) %>% 
    mutate(group = "36 Months")


# AUROC Violin Plot ------------------------------------------------------------
auroc_violinplot <- auroc_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "AUROC") %>% 
    mutate(model = factor(
        model, levels = c("Baseline", "12-month", "24-month", "36-month"))
    ) %>% 
    ggplot(aes(x = model, y = AUROC)) +
        geom_violin(width = 0.8, color = NA, fill = "grey70") +
        stat_summary(
            fun.data = "mean_sd", geom = "pointrange", colour = "grey20"
        ) +
        stat_compare_means(label.x = 0.8, size = 6) +
        labs(x = "", y = "Area under the ROC Curve (AUROC)") +
        theme_pander() +
        scale_y_continuous(limits = c(0.575, 0.86), breaks = seq(0.55, 0.9, 0.05)) +
        theme(
            plot.margin = margin(5, 5, 5, 5, "mm"),
            axis.title.y = element_text(size = 18),
            axis.text = element_text(size = 16)
        )
auroc_violinplot
ggsave(
    plot = auroc_violinplot,
    filename = here("outputs", "figs", "auroc_violin.pdf"),
    width = 8, 
    height = 5
)

auroc_df_long <- auroc_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "AUROC") 

# get mean, SD, and 95% CI of all models
auroc_df_long %>% 
    group_by(model) %>% 
    summarise(
        mean = mean(AUROC),
        sd = sd(AUROC),
        ci_lower = confidence_interval(AUROC)[1],
        ci_upper = confidence_interval(AUROC)[2]
    )

kruskal.test(AUROC ~ model, data = auroc_df_long)

# get pairwise t test for all models
auroc_df_long %>% 
    rstatix::pairwise_t_test(AUROC ~ model, p.adjust.method = "fdr")


# Brier Violin Plot ------------------------------------------------------------
brier_violinplot <- bs_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "BS") %>% 
    mutate(model = factor(
        model, levels = c("Baseline", "12-month", "24-month", "36-month"))
    ) %>% 
    ggplot(aes(x = model, y = BS)) +
    geom_violin(width = 0.8, color = NA, fill = "grey70") +
    stat_summary(
        fun.data = "mean_sd", geom = "pointrange", colour = "grey20"
    ) +
    stat_compare_means(label.x = 0.8, label.y = 0.126, size = 6) +
    labs(x = "", y = "Brier Score") +
    theme_pander() +
    theme(
        plot.margin = margin(5, 5, 5, 5, "mm"),
        axis.title.y = element_text(size = 18),
        axis.text = element_text(size = 16)
    )
brier_violinplot
ggsave(
    plot = brier_violinplot,
    filename = here("outputs", "figs", "brier_violin.pdf"),
    width = 8, 
    height = 5
)

bs_df_long <- bs_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "Brier")

# get mean, SD, and 95% CI of all models
bs_df_long %>% 
    group_by(model) %>% 
    summarise(
        mean = mean(Brier),
        sd = sd(Brier),
        ci_lower = confidence_interval(Brier)[1],
        ci_upper = confidence_interval(Brier)[2]
    )

kruskal.test(Brier ~ model, data = bs_df_long)

# get pairwise t test for all models
auroc_df_long %>% 
    rstatix::pairwise_t_test(AUROC ~ model, p.adjust.method = "fdr")


# AUROC Violin Plot Feature Variation ------------------------------------------
auroc_feature_df <- auroc_df %>% 
    mutate(feature = "All") %>% 
    bind_rows(auroc_top10_df) %>% 
    bind_rows(auroc_top15_df) %>% 
    bind_rows(auroc_top20_df) %>% 
    pivot_longer(cols = -feature, names_to = "model", values_to = "AUROC") %>% 
    mutate(model = factor(
        model, levels = c("Baseline", "12-month", "24-month", "36-month"))
    ) %>% 
    mutate(feature = factor(
        feature, levels = c("All", "Top 10", "Top 15", "Top 20"))
    )
auroc_feature_violinplot <- auroc_feature_df %>% 
    ggplot(aes(x = feature, y = AUROC)) +
        geom_violin(width = 0.8, color = NA, fill = "grey70") +
        stat_summary(
            fun.data = "mean_sd", geom = "pointrange", colour = "grey20"
        ) +
        facet_wrap(vars(model), ncol = 4) +
        stat_compare_means(label.y = 0.87) +
        labs(x = "", y = "Area under the ROC Curve (AUROC)") +
        theme_pander() +
        scale_y_continuous(limits = c(0.575, 0.9), breaks = seq(0.55, 1, 0.05)) +
        theme(
            plot.margin = margin(5, 5, 5, 5, "mm"),
            legend.position = "none",
            axis.text.x = element_text(angle = 45)
        )

brier_feature_df <- bs_df %>% 
    mutate(feature = "All") %>% 
    bind_rows(bs_top10_df) %>% 
    bind_rows(bs_top15_df) %>% 
    bind_rows(bs_top20_df) %>% 
    pivot_longer(cols = -feature, names_to = "model", values_to = "BS") %>% 
    mutate(model = factor(
        model, levels = c("Baseline", "12-month", "24-month", "36-month"))
    ) %>% 
    mutate(feature = factor(
        feature, levels = c("All", "Top 10", "Top 15", "Top 20"))
    )
brier_feature_violinplot <- brier_feature_df %>% 
    ggplot(aes(x = feature, y = BS)) +
        geom_violin(width = 0.8, color = NA, fill = "grey70") +
        stat_summary(
            fun.data = "mean_sd", geom = "pointrange", colour = "grey20"
        ) +
        facet_wrap(vars(model), ncol = 4) +
        stat_compare_means(label.y = 0.125) +
        labs(x = "Number of Features", y = "Brier Score") +
        theme_pander() +
        scale_y_continuous(limits = c(0.075, 0.13), breaks = seq(0.06, 0.12, 0.01)) +
        theme(
            plot.margin = margin(5, 5, 5, 5, "mm"),
            legend.position = "none",
            axis.text.x = element_text(angle = 45)
        )

feature_violinplot <- auroc_feature_violinplot / brier_feature_violinplot +
    plot_annotation(tag_levels = "A")
feature_violinplot
ggsave(
    plot = feature_violinplot,
    filename = here("outputs", "figs", "feature_violin.pdf"),
    width = 11, 
    height = 9
)

# save scores to tables
auroc_feature_df %>% 
    group_by(model) %>% 
    rstatix::t_test(
        AUROC ~ feature, 
        #detailed = TRUE,
        p.adjust.method = "fdr"
    ) %>% 
    select(model, group1, group2, statistic, p, p.adj, p.adj.signif) %>% 
    flextable() %>% 
    bold(part = "header") %>% 
    autofit() %>% 
    save_as_docx(path = here("outputs", "tables", "table_auroc_feature_comparisons.docx"))

brier_feature_df %>% 
    group_by(model) %>% 
    rstatix::t_test(
        BS ~ feature, 
        #detailed = TRUE,
        p.adjust.method = "fdr"
    ) %>% 
    select(model, group1, group2, statistic, p, p.adj, p.adj.signif) %>% 
    flextable() %>% 
    bold(part = "header") %>% 
    autofit() %>% 
    save_as_docx(path = here("outputs", "tables", "table_bs_feature_comparisons.docx"))



# Performance at Cutoff Table --------------------------------------------------
performance_cutoff_baseline %>% 
    bind_rows(performance_cutoff_12m) %>% 
    bind_rows(performance_cutoff_24m) %>% 
    bind_rows(performance_cutoff_36m) %>% 
    mutate(across(-c('Risk Cutoff', group), round, 2)) %>% 
    unite("Sensitivity", 2:3, sep = " ± ", remove = TRUE) %>% 
    unite("Specificity", 3:4, sep = " ± ", remove = TRUE) %>% 
    unite("PPV", 4:5, sep = " ± ", remove = TRUE) %>% 
    unite("NPV", 5:6, sep = " ± ", remove = TRUE) %>% 
    unite("NB", 6:7, sep = " ± ", remove = TRUE) %>% 
    unite("sNB", 7:8, sep = " ± ", remove = TRUE) %>% 
    as_grouped_data(groups = c("group"), columns = NULL) %>% 
    flextable() %>% 
    autofit() %>% 
    bold(part = "header") %>% 
    save_as_docx(path = here("outputs", "tables", "table_riskcutoff_performance.docx"))

# print AUROC mean, sd, and 95% CI
mean(auroc_df %>% pull(1))
sd(auroc_df %>% pull(1))
confidence_interval(auroc_df %>% pull(1))
mean(auroc_df %>% pull(2))
sd(auroc_df %>% pull(2))
confidence_interval(auroc_df %>% pull(2))
mean(auroc_df %>% pull(3))
sd(auroc_df %>% pull(3))
confidence_interval(auroc_df %>% pull(3))
mean(auroc_df %>% pull(4))
sd(auroc_df %>% pull(4))
confidence_interval(auroc_df %>% pull(4))

# print Brier Score mean and 95% CI
mean(bs_df %>% pull(1))
sd(bs_df %>% pull(1))
confidence_interval(bs_df %>% pull(1))
mean(bs_df %>% pull(2))
sd(bs_df %>% pull(2))
confidence_interval(bs_df %>% pull(2))
mean(bs_df %>% pull(3))
sd(bs_df %>% pull(3))
confidence_interval(bs_df %>% pull(3))
mean(bs_df %>% pull(4))
sd(bs_df %>% pull(4))
confidence_interval(bs_df %>% pull(4))


# generate reports
auroc_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "auroc") %>%
    kruskal.test(auroc ~ model, data = .)

bs_df %>% 
    pivot_longer(cols = everything(), names_to = "model", values_to = "Brier") %>%
    kruskal.test(Brier ~ model, data = .)
