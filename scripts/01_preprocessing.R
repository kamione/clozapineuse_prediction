# Environment ------------------------------------------------------------------
library(tidyverse)
library(here)
library(haven)
library(gtsummary)
library(glue)
library(flextable)
library(lubridate)


# Data I/O ---------------------------------------------------------------------
df1 <- here("data", "raw", "FULL_1400_TRS.sav") %>% 
    read_sav() %>% 
    #as_factor() %>% 
    zap_labels() %>% 
    zap_label() %>% 
    zap_missing() %>% 
    zap_widths() %>% 
    zap_formats() %>% 
    rename("HCS_code" = "HCS3ID")

df2 <- here("data", "raw", "Bkgd, PreDUP, DUP23 050814_partial_20230407.sav") %>% 
    read_sav() %>% 
    #as_factor() %>% 
    zap_labels() %>% 
    zap_label() %>% 
    zap_missing() %>% 
    zap_widths() %>% 
    zap_formats() %>% 
    # duplicated variables
    select(-c(Age_onset, Sex, Smoker, Yrs_edu, life_event1, Dx_cat, Dx, 
              onsetmode, Occup_impair, life_event2, ddd_36mth_mean))

# 192 TRS before 01-Jul-2015
df3 <- here("data", "raw", "1400_clozapine status_042023.xlsx") %>% 
    readxl::read_excel(
        col_types = c("text", "logical", "date"),
        na = c("888", "999")
    ) %>% 
    rename(
        "HCS_code" = "HCS code",
        "Clozintake_YN" = "Clozapine intake_YN",
        "Clozintake_date" = "clozapine_start_date_2022"
    ) %>% 
    mutate(Clozintake_date = as_date(Clozintake_date)) %>% 
    mutate(Clozintake_date_2 = if_else(
        Clozintake_date < ymd(20150701), 
        Clozintake_date,
        NA
    )) %>% 
    mutate(Clozintake_YN_2 = if_else(
        is.na(Clozintake_date_2) == TRUE, 
        FALSE,
        TRUE
    ))


df <- df1 %>% 
    left_join(df2, by = "HCS_code") %>% 
    left_join(df3, by = "HCS_code") %>% 
    mutate(
        Date_1stpre = dmy(Date_1stpre),
        DOB = ymd(DOB),
    ) %>% 
    mutate(
        age_at_1stpre = interval(DOB, Date_1stpre) / years(1), 
        age_cloz = interval(DOB, Clozintake_date_2) / years(1),
        length_cloz_intake = interval(Date_1stpre, Clozintake_date_2) / months(1)
    ) %>% 
    drop_na(Clozintake_YN) %>% 
    filter(length_cloz_intake > 0 | is.na(length_cloz_intake))


# Preprocessing ----------------------------------------------------------------
df %>% 
    select(length_cloz_intake) %>% 
    drop_na() %>% 
    summarize(
        max = max(length_cloz_intake),
        min = min(length_cloz_intake),
        mean = mean(length_cloz_intake),
        median = median(length_cloz_intake),
        sd = sd(length_cloz_intake)
    )

# create baseline dataframe
preprocessed_baseline_df <- df %>% 
    # remove participants who start clozapine within first month
    filter(is.na(length_cloz_intake) | length_cloz_intake > 1) %>%
    select(
        # baseline
        Clozintake_YN_2, Hosp_cluster, age_at_1stpre, Age_onset, Sex, Yrs_edu, 
        DUP_d, Filter, Smoker2, Dx_schiz, Dx_recode1, life_event1, EP1_d, 
        DUP_SS_His, DUP_DSH_His, M1hosp_days,
        # comorbidity
        co_ocd, co_anx, co_dep, co_sa, co_oth, 
        # longitudinal clinical
        M1_Poscf, M1_Negcf, M1_OPD,
        M1_Aff, M1_compliance, 
        M1_SOFAScf, 
        # clinically related  
        M1_SS, M1_dsh, M1_ae, M1_IP, M1_sa,
        M1_default,
        # longitudinal medication
        M1_ddd, M1_AC, M1_AD, M1_ect,
        M1_MS, M1_BZ, Poly_M01_r
    ) %>% 
    mutate_at(vars(contains('co_')), list(~ factor(.))) %>% 
    mutate_at(vars(contains('_sa')), list(~ if_else(. == "888", 0, 1))) %>% 
    mutate(
        is_cloz = factor(Clozintake_YN_2),
        is_sczdx = factor(if_else(Dx_schiz == 1, 1, 0)),
        is_affective = factor(if_else(Dx_recode1 == 1, 1, 0)),
        is_ei = factor(Filter),
        DUP_d_log = log(DUP_d), 
        has_lifeevent = factor(if_else(life_event1 == 1, 0, 1)),
        is_smoker = factor(Smoker2),
        DUP_SS_His = factor(DUP_SS_His),
        DUP_DSH_His = factor(DUP_DSH_His),
        .keep = "unused"
    ) %>% 
    rowwise() %>% 
    # get first 1 months clinical information
    mutate(
        # symptoms and functioning
        pos_mean = M1_Poscf, 
        neg_mean = M1_Negcf, 
        dep_mean = M1_Aff, 
        ma_mean = M1_compliance,
        sofas_mean = M1_SOFAScf,
        # clinically related
        ss_sum = M1_SS, 
        dsh_sum = M1_dsh,
        sa_sum = M1_sa,
        ae_sum = M1_ae,
        opd_sum = M1_OPD, 
        ip_sum = M1_IP,
        default_sum = M1_default,
        # medication
        ddd_mean = M1_ddd,
        ac_sum = M1_AC,
        ad_sum = M1_AD,
        bz_sum = M1_BZ,
        ms_sum = M1_MS,
        poly_sum = Poly_M01_r,
        ect_sum = M1_ect,
        .keep = "unused"
    ) %>% 
    ungroup() %>% 
    select(is_cloz, age_at_1stpre:M1hosp_days, is_sczdx:is_smoker, everything())

write_csv(
    preprocessed_baseline_df, 
    here("data", "processed", glue("cohort_trs-n_{dim(preprocessed_baseline_df)[1]}-desc_baseline.csv"))
)

preprocessed_baseline_df %>% 
    select(-Hosp_cluster) %>% 
    tbl_summary(
        by = is_cloz,
        type = where(is.numeric) ~ "continuous",
        statistic = list(all_continuous() ~ "{mean} ({sd})"),
        digits = all_continuous() ~ 2,
        missing = "no"
    ) %>% 
    add_p() %>% 
    add_q() %>% 
    bold_p(q = TRUE)


# create 12 month dataframe
preprocessed_12m_df <- df %>% 
    # remove participants who start clozapine within first 1 years
    filter(is.na(length_cloz_intake) | length_cloz_intake > 12) %>%
    select(
        # baseline
        Clozintake_YN_2, Hosp_cluster, age_at_1stpre, Age_onset, Sex, Yrs_edu, DUP_d, Filter, Smoker2,
        Dx_schiz, Dx_recode1, life_event1, EP1_d, DUP_SS_His, DUP_DSH_His, 
        Relapse_Y1, M1hosp_days,
        # comorbidity
        co_ocd, co_anx, co_dep, co_sa, co_oth, 
        # longitudinal clinical
        M1_Poscf:M12_Poscf, M1_Negcf:M12_Negcf, M1_OPD:M12_OPD,
        M1_Aff:M12_Aff, M1_compliance:M12_compliance, 
        M1_SOFAScf:M12_SOFAScf, 
        # clinically related  
        M1_SS:M12_SS, M1_dsh:M12_dsh, M1_ae:M12_ae, M1_IP:M12_IP, M1_sa:M12_sa,
        M1_default:M12_default,
        # longitudinal medication
        M1_ddd:M12_ddd, M1_AC:M12_AC, M1_AD:M12_AD, M1_ect:M12_ect,
        M1_MS:M12_MS, M1_BZ:M12_BZ, Poly_M01_r:Poly_M12_r
    ) %>% 
    mutate_at(vars(contains('co_')), list(~ factor(.))) %>% 
    mutate_at(vars(contains('_sa')), list(~ if_else(. == "888", 0, 1))) %>% 
    mutate(
        is_cloz = factor(Clozintake_YN_2),
        is_sczdx = factor(if_else(Dx_schiz == 1, 1, 0)),
        is_affective = factor(if_else(Dx_recode1 == 1, 1, 0)),
        is_ei = factor(Filter),
        DUP_d_log = log(DUP_d), 
        has_lifeevent = factor(if_else(life_event1 == 1, 0, 1)),
        is_smoker = factor(Smoker2),
        DUP_SS_His = factor(DUP_SS_His),
        DUP_DSH_His = factor(DUP_DSH_His),
        .keep = "unused"
    ) %>% 
    rowwise() %>% 
    # get first 12 months clinical information
    mutate(
        # symptoms and functioning
        pos_mean = mean(c_across(M1_Poscf:M12_Poscf), na.rm = TRUE), 
        neg_mean = mean(c_across(M1_Negcf:M12_Negcf), na.rm = TRUE), 
        dep_mean = mean(c_across(M1_Aff:M12_Aff), na.rm = TRUE), 
        ma_mean = mean(c_across(M1_compliance:M12_compliance), na.rm = TRUE),
        sofas_mean = mean(c_across(M1_SOFAScf:M12_SOFAScf), na.rm = TRUE),
        pos_mssd = psych::mssd(c_across(M1_Poscf:M12_Poscf), na.rm = TRUE), 
        neg_mssd = psych::mssd(c_across(M1_Negcf:M12_Negcf), na.rm = TRUE),
        dep_mssd = psych::mssd(c_across(M1_Aff:M12_Aff), na.rm = TRUE),
        ma_mssd = psych::mssd(c_across(M1_compliance:M12_compliance), na.rm = TRUE), 
        sofas_mssd = psych::mssd(c_across(M1_SOFAScf:M12_SOFAScf), na.rm = TRUE), 
        # clinically related
        ss_sum = sum(c_across(M1_SS:M12_SS), na.rm = TRUE), 
        dsh_sum = sum(c_across(M1_dsh:M12_dsh), na.rm = TRUE),
        sa_sum = sum(c_across(M1_sa:M12_sa), na.rm = TRUE),
        ae_sum = sum(c_across(M1_ae:M12_ae), na.rm = TRUE),
        opd_sum = sum(c_across(M1_OPD:M12_OPD), na.rm = TRUE), 
        ip_sum = sum(c_across(M1_IP:M12_IP), na.rm = TRUE),
        default_sum = sum(c_across(M1_default:M12_default), na.rm = TRUE),
        relapse_sum = Relapse_Y1,
        # medication
        ddd_mean = mean(c_across(M1_ddd:M12_ddd), na.rm = TRUE),
        ac_sum = sum(c_across(M1_AC:M12_AC), na.rm = TRUE),
        ad_sum = sum(c_across(M1_AD:M12_AD), na.rm = TRUE),
        bz_sum = sum(c_across(M1_BZ:M12_BZ), na.rm = TRUE),
        ms_sum = sum(c_across(M1_MS:M12_MS), na.rm = TRUE),
        poly_sum = sum(c_across(Poly_M01_r:Poly_M12_r), na.rm = TRUE),
        ect_sum = sum(c_across(M1_ect:M12_ect), na.rm = TRUE),
        .keep = "unused"
    ) %>% 
    ungroup() %>% 
    select(is_cloz, age_at_1stpre:M1hosp_days, is_sczdx:is_smoker, everything())

write_csv(
    preprocessed_12m_df, 
    here("data", "processed", glue("cohort_trs-n_{dim(preprocessed_12m_df)[1]}-desc_12m.csv"))
)

preprocessed_12m_df %>% 
    select(-Hosp_cluster) %>% 
    tbl_summary(
        by = is_cloz,
        type = where(is.numeric) ~ "continuous",
        statistic = list(all_continuous() ~ "{mean} ({sd})"),
        digits = all_continuous() ~ 2,
        missing = "no"
    ) %>% 
    add_p() %>% 
    add_q() %>% 
    bold_p(q = TRUE)

# create 24m dataframe
preprocessed_24m_df <- df %>% 
    # remove participants who start clozapine within first 3 years
    filter(is.na(length_cloz_intake) | length_cloz_intake > 24) %>%
    select(
        # baseline
        Clozintake_YN_2, Hosp_cluster, age_at_1stpre, Age_onset, Sex, Yrs_edu, DUP_d, Filter, Smoker2,
        Dx_schiz, Dx_recode1, life_event1, EP1_d, DUP_SS_His, DUP_DSH_His, 
        Relapse_Y1, Relapse_Y2, M1hosp_days,
        # comorbidity
        co_ocd, co_anx, co_dep, co_sa, co_oth, 
        # longitudinal clinical
        M1_Poscf:M24_Poscf, M1_Negcf:M24_Negcf, M1_OPD:M24_OPD,
        M1_Aff:M24_Aff, M1_compliance:M24_compliance, 
        M1_SOFAScf:M24_SOFAScf, 
        # clinically related  
        M1_SS:M24_SS, M1_dsh:M24_dsh, M1_ae:M24_ae, M1_IP:M24_IP, M1_sa:M24_sa,
        M1_default:M24_default,
        # longitudinal medication
        M1_ddd:M24_ddd, M1_AC:M24_AC, M1_AD:M24_AD, M1_ect:M24_ect,
        M1_MS:M24_MS, M1_BZ:M24_BZ, Poly_M01_r:Poly_M24_r
    ) %>% 
    mutate_at(vars(contains('co_')), list(~ factor(.))) %>% 
    mutate_at(vars(contains('_sa')), list(~ if_else(. == "888", 0, 1))) %>% 
    mutate(
        is_cloz = factor(Clozintake_YN_2),
        is_sczdx = factor(if_else(Dx_schiz == 1, 1, 0)),
        is_affective = factor(if_else(Dx_recode1 == 1, 1, 0)),
        is_ei = factor(Filter),
        DUP_d_log = log(DUP_d), 
        has_lifeevent = factor(if_else(life_event1 == 1, 0, 1)),
        is_smoker = factor(Smoker2),
        DUP_SS_His = factor(DUP_SS_His),
        DUP_DSH_His = factor(DUP_DSH_His),
        .keep = "unused"
    ) %>% 
    rowwise() %>% 
    # get first 36 months clinical information
    mutate(
        # symptoms and functioning
        pos_mean = mean(c_across(M1_Poscf:M24_Poscf), na.rm = TRUE), 
        neg_mean = mean(c_across(M1_Negcf:M24_Negcf), na.rm = TRUE), 
        dep_mean = mean(c_across(M1_Aff:M24_Aff), na.rm = TRUE), 
        ma_mean = mean(c_across(M1_compliance:M24_compliance), na.rm = TRUE),
        sofas_mean = mean(c_across(M1_SOFAScf:M24_SOFAScf), na.rm = TRUE),
        pos_mssd = psych::mssd(c_across(M1_Poscf:M24_Poscf), na.rm = TRUE), 
        neg_mssd = psych::mssd(c_across(M1_Negcf:M24_Negcf), na.rm = TRUE),
        dep_mssd = psych::mssd(c_across(M1_Aff:M24_Aff), na.rm = TRUE),
        ma_mssd = psych::mssd(c_across(M1_compliance:M24_compliance), na.rm = TRUE), 
        sofas_mssd = psych::mssd(c_across(M1_SOFAScf:M24_SOFAScf), na.rm = TRUE), 
        # clinically related
        ss_sum = sum(c_across(M1_SS:M24_SS), na.rm = TRUE), 
        dsh_sum = sum(c_across(M1_dsh:M24_dsh), na.rm = TRUE),
        sa_sum = sum(c_across(M1_sa:M24_sa), na.rm = TRUE),
        ae_sum = sum(c_across(M1_ae:M24_ae), na.rm = TRUE),
        opd_sum = sum(c_across(M1_OPD:M24_OPD), na.rm = TRUE), 
        ip_sum = sum(c_across(M1_IP:M24_IP), na.rm = TRUE),
        default_sum = sum(c_across(M1_default:M24_default), na.rm = TRUE),
        relapse_sum = Relapse_Y1 + Relapse_Y2,
        # medication
        ddd_mean = mean(c_across(M1_ddd:M24_ddd), na.rm = TRUE),
        ac_sum = sum(c_across(M1_AC:M24_AC), na.rm = TRUE),
        ad_sum = sum(c_across(M1_AD:M24_AD), na.rm = TRUE),
        bz_sum = sum(c_across(M1_BZ:M24_BZ), na.rm = TRUE),
        ms_sum = sum(c_across(M1_MS:M24_MS), na.rm = TRUE),
        poly_sum = sum(c_across(Poly_M01_r:Poly_M24_r), na.rm = TRUE),
        ect_sum = sum(c_across(M1_ect:M24_ect), na.rm = TRUE),
        .keep = "unused"
    ) %>% 
    ungroup() %>% 
    select(is_cloz, age_at_1stpre:M1hosp_days, is_sczdx:is_smoker, everything())

write_csv(
    preprocessed_24m_df, 
    here("data", "processed", glue("cohort_trs-n_{dim(preprocessed_24m_df)[1]}-desc_24m.csv"))
)

preprocessed_24m_df %>% 
    tbl_summary(
        by = is_cloz,
        type = where(is.numeric) ~ "continuous",
        statistic = list(all_continuous() ~ "{mean} ({sd})"),
        digits = all_continuous() ~ 2,
        missing = "no"
    ) %>% 
    add_p() %>% 
    add_q() %>% 
    bold_p(q = TRUE)


# create 36m dataframe
preprocessed_36m_df <- df %>% 
    # remove participants who start clozapine within first 3 years
    filter(is.na(length_cloz_intake) | length_cloz_intake > 36) %>%
    select(
        # baseline
        Clozintake_YN_2, Hosp_cluster, age_at_1stpre, Age_onset, Sex, Yrs_edu, DUP_d, Filter, Smoker2,
        Dx_schiz, Dx_recode1, life_event1, EP1_d, DUP_SS_His, DUP_DSH_His, 
        Relapse_Y1, Relapse_Y2, Relapse_Y3, M1hosp_days,
        # comorbidity
        co_ocd, co_anx, co_dep, co_sa, co_oth, 
        # longitudinal clinical
        M1_Poscf:M36_Poscf, M1_Negcf:M36_Negcf, M1_OPD:M36_OPD,
        M1_Aff:M36_Aff, M1_compliance:M36_compliance, 
        M1_SOFAScf:M36_SOFAScf, 
        # clinically related  
        M1_SS:M36_SS, M1_dsh:M36_dsh, M1_ae:M36_ae, M1_IP:M36_IP, M1_sa:M36_sa,
        M1_default:M36_default,
        # longitudinal medication
        M1_ddd:M36_ddd, M1_AC:M36_AC, M1_AD:M36_AD, M1_ect:M36_ect,
        M1_MS:M36_MS, M1_BZ:M36_BZ, Poly_M01_r:Poly_M36_r
    ) %>% 
    mutate_at(vars(contains('co_')), list(~ factor(.))) %>% 
    mutate_at(vars(contains('_sa')), list(~ if_else(. == "888", 0, 1))) %>% 
    mutate(
        is_cloz = factor(Clozintake_YN_2),
        is_sczdx = factor(if_else(Dx_schiz == 1, 1, 0)),
        is_affective = factor(if_else(Dx_recode1 == 1, 1, 0)),
        is_ei = factor(Filter),
        DUP_d_log = log(DUP_d), 
        has_lifeevent = factor(if_else(life_event1 == 1, 0, 1)),
        is_smoker = factor(Smoker2),
        DUP_SS_His = factor(DUP_SS_His),
        DUP_DSH_His = factor(DUP_DSH_His),
        .keep = "unused"
    ) %>% 
    rowwise() %>% 
    # get first 36 months clinical information
    mutate(
        # symptoms and functioning
        pos_mean = mean(c_across(M1_Poscf:M36_Poscf), na.rm = TRUE), 
        neg_mean = mean(c_across(M1_Negcf:M36_Negcf), na.rm = TRUE), 
        dep_mean = mean(c_across(M1_Aff:M36_Aff), na.rm = TRUE), 
        ma_mean = mean(c_across(M1_compliance:M36_compliance), na.rm = TRUE),
        sofas_mean = mean(c_across(M1_SOFAScf:M36_SOFAScf), na.rm = TRUE),
        pos_mssd = psych::mssd(c_across(M1_Poscf:M36_Poscf), na.rm = TRUE), 
        neg_mssd = psych::mssd(c_across(M1_Negcf:M36_Negcf), na.rm = TRUE),
        dep_mssd = psych::mssd(c_across(M1_Aff:M36_Aff), na.rm = TRUE),
        ma_mssd = psych::mssd(c_across(M1_compliance:M36_compliance), na.rm = TRUE), 
        sofas_mssd = psych::mssd(c_across(M1_SOFAScf:M36_SOFAScf), na.rm = TRUE), 
        # clinically related
        ss_sum = sum(c_across(M1_SS:M36_SS), na.rm = TRUE), 
        dsh_sum = sum(c_across(M1_dsh:M36_dsh), na.rm = TRUE),
        sa_sum = sum(c_across(M1_sa:M36_sa), na.rm = TRUE),
        ae_sum = sum(c_across(M1_ae:M36_ae), na.rm = TRUE),
        opd_sum = sum(c_across(M1_OPD:M36_OPD), na.rm = TRUE), 
        ip_sum = sum(c_across(M1_IP:M36_IP), na.rm = TRUE),
        default_sum = sum(c_across(M1_default:M36_default), na.rm = TRUE),
        relapse_sum = Relapse_Y1 + Relapse_Y2 + Relapse_Y3,
        # medication
        ddd_mean = mean(c_across(M1_ddd:M36_ddd), na.rm = TRUE),
        ac_sum = sum(c_across(M1_AC:M36_AC), na.rm = TRUE),
        ad_sum = sum(c_across(M1_AD:M36_AD), na.rm = TRUE),
        bz_sum = sum(c_across(M1_BZ:M36_BZ), na.rm = TRUE),
        ms_sum = sum(c_across(M1_MS:M36_MS), na.rm = TRUE),
        poly_sum = sum(c_across(Poly_M01_r:Poly_M36_r), na.rm = TRUE),
        ect_sum = sum(c_across(M1_ect:M36_ect), na.rm = TRUE),
        .keep = "unused"
    ) %>% 
    ungroup() %>% 
    select(is_cloz, age_at_1stpre:M1hosp_days, is_sczdx:is_smoker, everything())

write_csv(
    preprocessed_36m_df, 
    here("data", "processed", glue("cohort_trs-n_{dim(preprocessed_36m_df)[1]}-desc_36m.csv"))
)

preprocessed_36m_df %>% 
    select(-Hosp_cluster) %>% 
    tbl_summary(
        by = is_cloz,
        type = where(is.numeric) ~ "continuous",
        statistic = list(all_continuous() ~ "{mean} ({sd})"),
        digits = all_continuous() ~ 2,
        missing = "no"
    ) %>% 
    add_p() %>% 
    add_q() %>% 
    bold_p(q = TRUE)



# Table 1 ----------------------------------------------------------------------
table01 <- preprocessed_baseline_df %>% 
    select(is_cloz, age_at_1stpre, Sex, Yrs_edu, is_sczdx, 
           Age_onset, is_ei, DUP_d_log) %>% 
    mutate(
        is_cloz = factor(
            is_cloz,
            levels = c(FALSE, TRUE),
            labels = c("No", "Yes")
        ),
        Sex = factor(
          Sex,
          levels = c(1, 2),
          labels = c("Male", "Female")
        ),
        is_sczdx = factor(
            is_sczdx,
            levels = c(1, 0),
            labels = c("Schizophrenia", "Other Schizophrenia spectrum")
        ),
        is_ei = factor(
            is_ei,
            levels = c(1, 0),
            labels = c("Early Intervention", "Standard Care")
        )
    ) %>% 
    tbl_summary(
        by = is_cloz,
        type = where(is.numeric) ~ "continuous",
        statistic = list(all_continuous() ~ "{mean} ({sd})"),
        digits = all_continuous() ~ 2,
        label = list(
            "age_at_1stpre" ~ "Age at first service contact",
            "Yrs_edu" ~ "Years of education",
            "is_sczdx" ~ "Diagnosis",
            "Age_onset" ~ "Age of illness onset",
            "is_ei" ~ "Treatment",
            "DUP_d_log" ~ "DUP days (log)"
        ),
        missing = "no"
    ) %>% 
    modify_spanning_header(all_stat_cols() ~ "**Clozapine Use**") %>% 
    add_p(test = all_continuous() ~ "t.test") %>% 
    modify_header(statistic ~ "**Estimate**") %>%
    modify_fmt_fun(statistic ~ style_sigfig) %>% 
    add_q() %>% 
    bold_p(q = TRUE)
table01 
table01 %>% 
    as_flex_table() %>% 
    autofit() %>% 
    save_as_docx(path = here("outputs", "tables", "table01.docx"))





