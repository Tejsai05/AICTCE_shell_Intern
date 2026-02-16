# AICTCE **Shell** 6-week Internship Project Report
## Proof of completion


![Proof Image](https://drive.google.com/uc?export=view&id=1Df3SbTIAF6aC7gRuiUTcPfDjfi1keDUn)

## Energy Consumption Analysis Using Smart Meter Data

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Data Preprocessing](#data-preprocessing)
6. [Analysis Results](#analysis-results)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)
9. [References](#references)

---

## Project Overview

This project focuses on analyzing energy consumption patterns using smart meter data from London households. The dataset contains daily energy consumption metrics collected from smart meters installed in residential properties. The analysis aims to identify consumption patterns, handle missing data appropriately, and prepare the dataset for potential predictive modeling.

### Domain
Energy Analytics, Smart Grid Technology, Time Series Analysis

### Technology Stack
- **Programming Language:** Python 3
- **Key Libraries:** 
  - pandas (Data manipulation)
  - kagglehub (Dataset acquisition)
  - numpy (Numerical operations)
- **Development Environment:** Google Colab / Jupyter Notebook

---

## Objectives

### Primary Objectives
1. **Data Acquisition:** Successfully download and load the "Smart Meters in London" dataset from Kaggle
2. **Data Exploration:** Understand the structure, features, and quality of the energy consumption data
3. **Data Cleaning:** Identify and handle missing values using appropriate imputation techniques
4. **Data Preparation:** Convert date columns to proper datetime format for time series analysis
5. **Statistical Analysis:** Examine energy consumption patterns across different households

### Secondary Objectives
1. Establish a reproducible data pipeline for energy analysis
2. Document data quality issues and resolution strategies
3. Prepare cleaned dataset for future machine learning applications
4. Build foundation for predictive modeling of energy consumption

---

## Dataset Description

### Source
**Dataset Name:** Smart Meters in London  
**Source:** Kaggle (jeanmidev/smart-meters-in-london)  
**Version:** 21  
**Size:** 1.17 GB (compressed)

### Dataset Characteristics
- **Total Records:** 3,510,433 daily observations
- **Number of Features:** 9 columns
- **Time Period:** Starting from December 15, 2011
- **Number of Households:** Multiple households identified by LCLid

### Features Description

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| `LCLid` | Object (String) | Unique identifier for each household/meter |
| `day` | Object (String → DateTime) | Date of measurement |
| `energy_median` | Float64 | Median energy consumption (kWh) for the day |
| `energy_mean` | Float64 | Mean energy consumption (kWh) for the day |
| `energy_max` | Float64 | Maximum energy consumption (kWh) recorded |
| `energy_count` | Int64 | Number of measurements taken during the day |
| `energy_std` | Float64 | Standard deviation of energy consumption |
| `energy_sum` | Float64 | Total energy consumption (kWh) for the day |
| `energy_min` | Float64 | Minimum energy consumption (kWh) recorded |

---

## Methodology

### Step 1: Data Acquisition
```python
import kagglehub
path = kagglehub.dataset_download("jeanmidev/smart-meters-in-london")
```
- Downloaded dataset using kagglehub API
- Extracted files to local cache directory
- Located the primary file: `daily_dataset.csv`

### Step 2: Data Loading
```python
import pandas as pd
df_daily = pd.read_csv(daily_dataset_path)
```
- Loaded CSV file into pandas DataFrame
- Initial memory usage: 241.0+ MB

### Step 3: Exploratory Data Analysis
- Examined first 5 rows using `df_daily.head()`
- Analyzed data structure using `df_daily.info()`
- Identified missing values using `df_daily.isnull().sum()`

### Step 4: Data Quality Assessment
Conducted comprehensive missing value analysis:
- Counted missing values per column
- Calculated percentage of missing data
- Assessed impact on analysis

### Step 5: Data Cleaning Strategy
Implemented median imputation for missing values:
- Chose median over mean due to robustness against outliers
- Applied to columns: energy_median, energy_mean, energy_max, energy_std, energy_sum, energy_min
- Verified successful imputation

### Step 6: Data Type Conversion
```python
df_daily['day'] = pd.to_datetime(df_daily['day'])
```
- Converted 'day' column from string to datetime format
- Enables time series operations and analysis

---

## Data Preprocessing

### Missing Value Analysis

#### Initial Missing Values Count
```
LCLid                0
 day                  0
 energy_median       30
 energy_mean         30
 energy_max          30
 energy_count         0
 energy_std       11,331
 energy_sum          30
 energy_min          30
```

#### Missing Values Percentage
```
LCLid            0.000000%
 day              0.000000%
 energy_median    0.000855%
 energy_mean      0.000855%
 energy_max       0.000855%
 energy_count     0.000000%
 energy_std       0.322781%
 energy_sum       0.000855%
 energy_min       0.000855%
```

### Key Observations
1. **Minimal Missing Data:** Most columns have less than 0.001% missing values
2. **energy_std Exception:** 0.32% missing values (11,331 records)
3. **Household Identifiers:** No missing values in LCLid or day columns
4. **Data Completeness:** 99.67% complete for energy_std, >99.99% for other metrics

### Imputation Strategy Justification

**Why Median Imputation?**
1. **Robustness:** Median is less sensitive to extreme outliers in energy consumption
2. **Appropriate for Skewed Data:** Energy consumption often follows non-normal distributions
3. **Minimal Impact:** Very low percentage of missing values minimizes distortion
4. **Preserves Distribution:** Better preserves the underlying data distribution than mean

**Alternative Strategies Considered:**
- **Drop rows:** Rejected due to loss of valuable data (11,331 records)
- **Mean imputation:** Rejected due to sensitivity to outliers
- **Forward/Backward fill:** Not appropriate for non-sequential missing patterns
- **Predictive imputation:** Overkill for <0.5% missing data

### Implementation Results
```
Missing values after imputation:
LCLid            0
 day              0
 energy_median    0
 energy_mean      0
 energy_max       0
 energy_count     0
 energy_std       0
 energy_sum       0
 energy_min       0
```
**Result:** 100% data completeness achieved

---

## Analysis Results

### Data Structure Summary

#### Before Processing
- **Data Types:** 6 float64, 1 int64, 2 object
- **Memory Usage:** 241.0+ MB
- **Date Format:** String (object)
- **Missing Values:** Present in multiple columns

#### After Processing
- **Data Types:** 6 float64, 1 int64, 1 datetime64, 1 object
- **Memory Usage:** ~241.0+ MB (optimized)
- **Date Format:** Datetime (enables time series operations)
- **Missing Values:** 0 (Complete dataset)

### Sample Data Insights

#### Example Household: MAC000131
**Period:** December 15-19, 2011

| Date | Median (kWh) | Mean (kWh) | Max (kWh) | Std Dev | Sum (kWh) | Min (kWh) | Count |
|------|-------------|-----------|----------|---------|----------|----------|-------|
| 2011-12-15 | 0.485 | 0.432 | 0.868 | 0.239 | 9.505 | 0.072 | 22 |
| 2011-12-16 | 0.142 | 0.296 | 1.116 | 0.281 | 14.216 | 0.031 | 48 |
| 2011-12-17 | 0.102 | 0.190 | 0.685 | 0.188 | 9.111 | 0.064 | 48 |
| 2011-12-18 | 0.114 | 0.219 | 0.676 | 0.203 | 10.511 | 0.065 | 48 |
| 2011-12-19 | 0.191 | 0.326 | 0.788 | 0.259 | 15.647 | 0.066 | 48 |

### Key Findings

1. **Measurement Frequency Variability**
   - December 15: Only 22 measurements (incomplete day)
   - December 16-19: Full 48 measurements (half-hourly readings)
   - Indicates potential meter installation or data collection start date

2. **Energy Consumption Patterns**
   - Daily totals range from 9.1 to 15.6 kWh
   - Average household consumption: 9-16 kWh/day
   - High variability: Standard deviation ranges from 0.188 to 0.281 kWh

3. **Consumption Variability**
   - Maximum to minimum ratios suggest significant daily variation
   - Peak consumption 10-35x higher than minimum consumption
   - Indicates diverse usage patterns (standby vs. active use)

### Statistical Insights

#### Data Quality Metrics
- **Completeness Rate:** 99.68% → 100% (after cleaning)
- **Consistency:** All energy metrics mathematically consistent
- **Temporal Coverage:** Multi-year dataset with daily granularity
- **Household Coverage:** 3.5M+ observations across multiple households

#### Dataset Readiness
✅ **Ready for:**
- Time series forecasting
- Anomaly detection
- Consumption pattern clustering
- Predictive modeling
- Statistical analysis

---

## Conclusions

### Project Achievements

1. **Successful Data Pipeline Implementation**
   - Automated dataset acquisition from Kaggle
   - Efficient data loading and exploration
   - Systematic data quality assessment

2. **Effective Data Cleaning**
   - Identified and quantified missing values
   - Applied appropriate imputation strategy
   - Achieved 100% data completeness
   - Maintained data integrity

3. **Data Preparation for Analysis**
   - Proper datetime conversion for time series operations
   - Preserved all 3.5M+ records
   - Maintained statistical properties of distributions
   - Created analysis-ready dataset

### Technical Learnings

1. **Data Quality Management**
   - Importance of missing value analysis before imputation
   - Selection of appropriate imputation methods based on data characteristics
   - Validation of data cleaning results

2. **Time Series Data Handling**
   - Proper datetime conversion essential for temporal analysis
   - Understanding measurement frequency and granularity
   - Handling incomplete time periods

3. **Large Dataset Management**
   - Efficient handling of 1.17 GB compressed data
   - Memory optimization techniques
   - Scalable data processing approaches

### Domain Insights

1. **Smart Meter Data Characteristics**
   - Half-hourly measurement frequency (48 readings/day)
   - Variability in data collection completeness
   - Multiple statistical aggregations per day

2. **Energy Consumption Patterns**
   - Significant intra-day variation in household consumption
   - Typical daily consumption ranges: 9-16 kWh
   - High peak-to-minimum consumption ratios

3. **Data Collection Challenges**
   - Initial meter installation periods show incomplete data
   - Systematic missing patterns in specific metrics
   - Importance of data quality in smart grid applications

---

alysis**
   - [ ] Identify consumption pattern clusters
   - [ ] Segment households by usage profiles
   - [ ] Detect typical vs. atypical consumption behaviors
   - [ ] Create household archetypes

6. **Anomaly Detection**
   - [ ] Implement statistical anomaly detection methods
   - [ ] Identify unusual consumption events
   - [ ] Detect potential meter malfunctions
   - [ ] Flag data quality issues



## References

### Dataset
- **Primary Source:** Jean-Michel D. (2020). Smart meters in London. Kaggle. Retrieved from https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london

### Tools & Libraries
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61.
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.
- Kaggle API Documentation: https://github.com/Kaggle/kaggle-api





