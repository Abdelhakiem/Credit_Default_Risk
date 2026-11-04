1. Data Cleaning:
    - Features, Target
TARGET_COL = 'TARGET'


applications.csv 

Check duplicates: SK_ID_CURR


Unique records check: SK_ID_CURR
Data Types conversion: 
_MODE : float64
FLAG_OWN_CAR , FLAG_OWN_REALTY : int64


Data validation: 
Unique values for cat types: TARGET ... Flags
Dist for numeric types

- Split Data : train, val, test
- Preprocessing 
------





2. EDA:

- Adding New Features (applications): credit and annuty to income , LOAN_LENGTH , EMPLOYEMENT_TO_AGE


- New Agg features from other tables: 

applications +  burea : Join on SK_ID_CURR (Agg: Num Cred,active-credit count = 'Active', max days overdue 
)


applications +  previous_application: Join on SK_ID_CURR (
Cnt . Approved , Approval Rate , Number of prior applications , Sum Approved applications , average requested/granted amount : AMT_APPLICATION / AMT_CREDIT
)
credit_card_balance : Join on SK_ID_CURR (Agg:Avg Credit card utilization rate: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, Average balance AMT_BALANCE)

installments_payments (Late-payment rate, payment-to-installment ratio)


-> Imputation  of Missing with 0 

- Feature Selection: 


3. Feature Engineering Pipeline


- Feature Selection : backward with Drop in score improvement.


    
- Training Pipeline DVC
    
4. Modeling: 
    - Modeling
    - Eval Metrics
    - XAI
    - Calibration
    - Track with MLFLOW
5. Inference 
    - FAST API
    - Deployement: Docker
    - Observability, Tracking ..etc
