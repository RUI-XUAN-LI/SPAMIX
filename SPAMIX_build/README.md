# SPAMIX Score Calculator

## Overview

This repository contains a feature extraction and scoring system for the SPAMIX blocklist using DNSBL data and email reputation metrics. The system computes multiple behavioral features for IP addresses, domain SLDs, and email senders, then combines them using weighted scoring models to produce unified anomaly scores.

## Features

### Feature Categories

1. **DNSBL IP Features** (per IP address per DNSBL service)
   - `reputation`: Historical reputation score based on blacklist/whitelist records with time decay
   - `flapping`: Frequency of state changes between black and white status
   - `activity_spike`: Short-term vs long-term activity ratio (7-day vs 90-day windows)
   - `active_day`: Ratio of inactive days over the lookback period

2. **DNSBL SLD Features** (per domain SLD per DNSBL service)
   - Same four feature types as IP features, applied to domain-level DNSBL data

3. **Email Features** (per sender target in format `ip##domain`)
   - `reputation`: Weighted spam ratio over historical email data
   - `active_score`: Activity regularity anomaly based on sending patterns
   - `dkim_pass`: DKIM authentication pass rate anomaly
   - `activity_spike`: Email volume spike detection (7-day vs 90-day)

### Weighted Scoring Models

The system supports configurable feature weight sets. The default configuration (`lr_model`) contains 32 feature weights optimized for logistic regression classification.

## Code Structure

```
.
├── DNSBL_IP_Feature_Calculator     # IP-level DNSBL feature computation
├── DNSBL_SLD_Feature_Calculator    # Domain SLD-level DNSBL feature computation  
├── Email_Feature_Calculator         # Email-based feature computation
├── calculate_daily_score()          # Single target scoring
├── compute_daily_scores()           # Multi-threaded daily scoring
└── main()                           # Main execution pipeline
```

## Input Data Format

All input data is read from CSV files located in the `./data/` directory.

### 1. `data/dnsbl_ip_records.csv`
IP address DNSBL records.

| Column       | Type     | Description                             |
| ------------ | -------- | --------------------------------------- |
| ip           | string   | IP address                              |
| dnsbl        | string   | DNSBL service name (e.g., spamhaus.org) |
| flag         | string   | "black" or "white"                      |
| start_time   | datetime | Start of listing period                 |
| end_time     | datetime | End of listing period                   |
| cnt          | int      | Number of queries during period         |
| client_count | int      | Number of unique clients                |
| date         | date     | Observation date                        |

### 2. `data/dnsbl_sld_records.csv`
Domain SLD DNSBL records.

| Column       | Type     | Description                             |
| ------------ | -------- | --------------------------------------- |
| sld          | string   | Second Level Domain (e.g., example.com) |
| dnsbl        | string   | DNSBL service name                      |
| flag         | string   | "black" or "white"                      |
| start_time   | datetime | Start of listing period                 |
| end_time     | datetime | End of listing period                   |
| cnt          | int      | Number of queries during period         |
| client_count | int      | Number of unique clients                |
| date         | date     | Observation date                        |

### 3. `data/email_reputation.csv`
Email reputation daily statistics.

| Column           | Type   | Description                              |
| ---------------- | ------ | ---------------------------------------- |
| target           | string | Sender identifier (format: `ip##domain`) |
| date             | date   | Date of observation                      |
| spam_cnt         | int    | Number of spam emails sent               |
| normal_cnt       | int    | Number of legitimate emails sent         |
| spam_dkim_pass   | int    | Spam emails with valid DKIM              |
| normal_dkim_pass | int    | Legitimate emails with valid DKIM        |

### 4. `data/email_targets.csv`
Daily aggregated target data (used by main function).

| Column     | Type   | Description                              |
| ---------- | ------ | ---------------------------------------- |
| target     | string | Sender identifier (format: `ip##domain`) |
| date       | date   | Date of observation                      |
| spam_cnt   | int    | Total spam count for the day             |
| normal_cnt | int    | Total legitimate count for the day       |

## Output Format

The main function generates a CSV file containing:

- **Fixed columns**: `target`, `date`, `mail_count`, `spam_cnt`, `normal_cnt`
- **Score columns**: `score_{model_name}` for each weight set
- **Feature columns**: All individual feature values (e.g., `ip_spamhaus.org_reputation`, `email_dkim_pass`)

Missing features are filled with `-1`.

## Usage

### Prerequisites

```bash
pip install numpy pandas scipy tqdm tldextract python-dateutil
```

### Running the Pipeline

```python
python get_score.py
```
