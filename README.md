# Capstone Project – Baseline Product Recommendation (Santander Data)

## 1. Project Overview

This capstone explores how a **recommendation system** can suggest relevant banking products to customers based on their past behavior and profile information.  

For this module, I focus on building and evaluating a **baseline predictive model** using the public **Santander Product Recommendation** dataset from Kaggle. The problem is framed as a **binary classification task**:  
> _Given a customer’s characteristics and product history, how likely is it that they have (or should be offered) a specific banking product?_

The work in this module centers on **EDA, data cleaning, feature engineering, and a first baseline model**, which will later be extended with more advanced recommendation techniques (e.g., collaborative filtering, clustering).

The main analysis is contained in:  
**`rec_sys.ipynb`**

---

## 2. Research Question

**How can a recommendation system help suggest the most relevant banking products to customers based on their past behavior and preferences?**

In this module, the question is operationalized as:

> _Can we predict whether a customer is likely to hold a specific product, using demographic and behavioral features from the Santander dataset, and does this perform better than a naive “recommend to everyone” baseline?_

---

## 3. Data Description

**Dataset:** Santander Product Recommendation (Kaggle)  
**Unit of analysis:** Customer-month record  
**Target variable:** A binary flag indicating whether the customer holds a chosen product (I chose credit card).

### Key Features Used

- **Demographics**
  - `age` – age of customer  
  - `antiguedad` – seniority (months) with the bank  
  - `renta` – estimated income  
  - `segmento` – customer segment (e.g., individuals, VIP, university)  
  - `indrel`, `ind_nuevo` – relationship indicators

- **Behavioral / Engineered**
  - Existing product indicator variables (0/1)  
  - `num_products` – engineered feature: total number of products currently held by the customer.

---

## 4. Data Cleaning & EDA

### Data Cleaning

In the notebook, I applied the following steps:

- **Subset & Sampling**
  - Loaded the Kaggle training file and sampled a subset of rows for faster experimentation.

- **Missing Values**
  - Converted clearly invalid values (e.g., `'NA'` strings) to `NaN`.  
  - Imputed numerical columns (e.g., `age`, `antiguedad`, `renta`) using median values.  
  - Imputed categorical columns (e.g., `segmento`) using the most frequent category.

- **Type Fixes**
  - Cast numeric columns to proper numeric types.  
  - Standardized categorical fields as strings for encoding.

- **Duplicates & Target Quality**
  - Removed obvious duplicate records within the sample.  
  - Ensured the target product column was strictly binary (0/1).

### Exploratory Data Analysis (EDA)

The EDA section includes:

- **Univariate distributions**
  - Histograms of `age`, `renta`, and `antiguedad` to understand typical customer profiles and spot outliers.
- **Product ownership patterns**
  - Distribution of the chosen target product (class balance).  
  - Count of how many products each customer holds (`num_products`), to see if “multi-product” customers look different.
- **Relationships**
  - Boxplots and bar charts comparing product ownership by:
    - Age groups  
    - Income bands  
    - Customer segments (`segmento`)
- **Correlations**
  - Correlation heatmap for numeric features to identify potentially redundant variables.

These visualizations helped to:

- Confirm **class imbalance** (many more customers without the product than with it).  
- See that customers who own the product often have **higher income and seniority** and tend to hold **more products overall**.  
- Understand which demographic groups might be good targets for recommendations.

---

## 5. Modeling Approach

### Problem Framing

- **Type:** Binary classification  
- **Goal:** Predict whether a customer has (or is a good candidate for) one specific product.  
- **Baseline:** A naive **popularity model** that simply recommends the product to every customer.

### Features & Preprocessing

I used a `sklearn` pipeline with:

- **Numerical features**
  - `age`, `antiguedad`, `renta`, `num_products`
- **Categorical features**
  - `segmento`, `indrel`, `ind_nuevo`

Preprocessing steps:

- `SimpleImputer` (median) for numericals  
- `SimpleImputer` (most frequent) + `OneHotEncoder` for categoricals  
- `LogisticRegression` as the baseline ML model

Train/test split (e.g., 80/20) was used to evaluate generalization performance.

### Evaluation Metrics

To keep the model aligned with recommendation quality, I used:

- **Precision** – Of the customers we recommend the product to, how many actually have it?
- **Recall** – Of all customers who truly have the product, how many did we flag?
- **F1 Score** – Harmonic mean of precision and recall; balances the two.

---

## 6. Results

### Popularity Baseline (recommend to everyone)

This baseline assumes “the product is popular, so offer it to all customers”:

- **Precision:** 0.063  
- **Recall:** 1.000  
- **F1 score:** 0.118  

Interpretation:

- The baseline **never misses** a true positive (recall = 1), but it does so by recommending the product to **all customers**, resulting in extremely low precision and poor overall F1.

### Logistic Regression Model

The logistic regression model with demographic, segment, and `num_products` features performed significantly better:

- **Logistic Regression F1:** 0.478  
- **Baseline F1:** 0.118  

Interpretation:

- The logistic regression model achieves roughly a **4× improvement in F1** compared to the naive baseline.
- It is **far more selective** (higher precision) while still capturing a meaningful portion of true positives.
- In practical terms, this means the bank can **target fewer customers** with **more relevant recommendations**, reducing spam while increasing relevance.

### Short Conclusion for This Module

> Predicting product ownership from customer features already provides a useful signal. Even a simple logistic regression model clearly outperforms a naive “recommend to everyone” strategy. This establishes a strong baseline for the capstone project and shows that customer demographics, segment information, and total number of products are important drivers of product adoption.

---

## 7. Repository / File Structure

Example structure (you can adjust to match your repo):

```text
.
├── README.md                          
├── data/
│   └── train_ver2.csv                 # Santander dataset (not committed because very large)
└── notebooks/
    └── rec_sys.ipynb
