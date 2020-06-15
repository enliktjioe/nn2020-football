# nn2020-football

## Team Members
- Enlik -
- Farhan Syakir
- Roland Hei Chun Shum
- Isaac Buo

## General Information
This neural network group project using Kaggle's [European Soccer Database](https://www.kaggle.com/hugomathien/soccer) as the main data source. We focused on training our model with data of two of the most popular leagues - English Premier League (EPL) and Spanish La Liga.

## Label Encoding (Our Target Variable)
### FTR (Full Time Result)
- 0 = A (Away team wins)
- 1 = D (Draw)
- 2 = H (Home team wins)

## Files Information
### 1. Folder /assets
Contains all files generated by jupyter notebook 

### 2. Folder /datasets
Contains some small .csv files required for data processing. For big size dataset such as `database.sqlite`, `EPL.csv`, `LaLiga.csv` can be downloaded from [here](https://drive.google.com/drive/folders/1Hvl0FX2EEwRTywcbDXjJ7uOuYgem1Vsz?usp=sharing)

### 3. 00_Data_Preparation.ipynb
Do pre-processing from the main dataset, split it into two datasets `EPL_sort.csv` and `LaLiga_sort.csv`

### 4. 01_EPL_stats and 01_LaLiga_stats
Do pre-processing from multiple csv files that contain the result and statistic from every season, create two datasets `epl_stats.csv` and `la_liga_stats.csv`

### 5. 02_FeatureSelection_EPL and 02_FeatureSelection_LaLiga
Do feature engineering for 30 features in total

### 6. BaselineModel_DT_LR_XGB
Baseline model using Decision Tree, Logistic Regression, and XGBoost Classifier

### 7. BaselineModel_RandomForest
Baseline model using RandomForest

### 8. HyperParameter_Result_EPL_LaLiga
Load the .npy files from assets folder that contains all the result from hyperparameter tuning process in related notebook

### 9. Prediction_Betting_EPL_LaLiga
Predict the betting result from `B365 bet odds` using our data and model

### 10. Prediction_FTR_EPL / LaLiga
Predict label data `FTR` for both leagues using Keras neural network model

### 11. Prediction_FTR_EPL_HyperParameter_Tuning / LaLiga
Do the hyperparameter tuning for both league with different configuration including: `learning_rates`, `hidden_layers`, `dropouts`, and `batch_size`

### 12. Stupid_Model_Only_Odd
Do two stupid betting strategies by follow the favor odd and against it

### 13. betting_utils.py
Contains all function helper for betting prediction


## Documentation
### NN Checkpoint 1
https://docs.google.com/document/d/1HNShLolCUOwyPjtI3_xDX5vFHFqyw-_3Le1IyMV9NQ8/edit

### NN Checkpoint 2
https://docs.google.com/document/d/1JsPqpf6NqSJDMVzoLV-ktXeFPDXmvQMlcDH1-Q8k6F0/edit?usp=sharing

### NN Project Plan
https://docs.google.com/document/d/1L3cmXeCscnx8eZP1kXbZaLn1YGdzrmDEwujWsW3Q8Ag/edit

### Blog Draft
https://docs.google.com/document/d/1jqoUnOtxa79pPdQPpHnHCO4ZJ5r2HStaiUfd2VXRCJ0/edit?ts=5ec67eac

### Resources
https://docs.google.com/document/d/1XbKOxCfPFMOn8m_ZItnZSHYc2IyGN22q42GrlFPamWE/edit
