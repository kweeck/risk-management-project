CREATE VIEW mart.features AS
SELECT
    id,
    is_default,

    -- исходные числовые фичи
    loan_amnt,
    int_rate,
    installment,
    annual_inc,
    dti,
    fico_avg,
    open_acc,
    revol_util,
    total_acc,

    -- производные фичи
    loan_amnt / NULLIF(annual_inc, 0)        AS loan_to_income,
    int_rate * dti                            AS rate_dti_interaction,
    installment / NULLIF(annual_inc / 12, 0) AS payment_to_income,

    -- категориальные
    home_ownership,
    purpose,
    term,

    -- для LGD
    ead,
    total_rec_prncp,
    CASE
        WHEN ead > 0
        THEN 1 - (total_rec_prncp / ead)
        ELSE NULL
    END AS lgd_actual,

    issue_date

FROM staging.loans