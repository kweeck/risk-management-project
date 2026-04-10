CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS raw.loans AS
SELECT * FROM read_csv_auto("C:/Users/kweec/python_files/risk-project/data/accepted_2007_to_2018Q4.csv", ignore_errors=true);