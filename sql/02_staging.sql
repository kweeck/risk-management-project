CREATE TABLE staging.loans AS
SELECT
    id,

    -- целевая переменная
    CASE
        WHEN loan_status IN ('Charged Off', 'Default') THEN 1
        ELSE 0
    END AS is_default,

    -- числовые фичи
    loan_amnt,
    installment,
    annual_inc,
    dti,
    fico_range_low,
    fico_range_high,
    (fico_range_low + fico_range_high) / 2.0          AS fico_avg,
    open_acc,
    revol_util,
    total_acc,

    -- категориальные фичи
    home_ownership,
    purpose,
    term,

    -- для LGD модели понадобится
    loan_amnt                                          AS ead,
    total_rec_prncp,
    recoveries,

    -- дата для временного разреза
    STRPTIME(issue_d, '%b-%Y')::DATE AS issue_date

FROM raw.loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
  AND annual_inc > 0 AND annual_inc < 500000
  AND loan_amnt > 0
  AND dti IS NOT NULL AND dti < 60