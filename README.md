# TASK_4_DATA – Book Store Analytics

This repository contains my solution for **Task 4 (DATA)** from the data-engineering course.

The project:

- Reads raw book-store data from three folders: **DATA1**, **DATA2**, **DATA3**.
- Cleans and normalises the data (dates, numeric fields, duplicates, malformed values).
- Calculates:
  - paid price in USD for every order,
  - daily revenue and top 5 days by revenue,
  - number of real unique users (with aliases merged),
  - number of unique author sets (solo and co-authored),
  - the most popular author set by sold copies,
  - the best customer by total spending (with all their alias `user_id`s).
- Generates a simple **dashboard** with three tabs (one per dataset).

---

## Project structure

```text
task4.py          # Main analysis script
DATA1/            # Raw data for dataset 1  (orders, users, books)
DATA2/            # Raw data for dataset 2
DATA3/            # Raw data for dataset 3
output/
  ├─ DATA1_summary.json
  ├─ DATA2_summary.json
  ├─ DATA3_summary.json
  ├─ all_summaries.json
  ├─ DATA1_daily_revenue.png
  ├─ DATA2_daily_revenue.png
  ├─ DATA3_daily_revenue.png
  └─ dashboard.html          # Static HTML dashboard
