from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model
from kfp import compiler
from typing import NamedTuple, List
import pandas as pd
# local_df = pd.read_csv("insurance.csv")
# csv_string = local_df.to_csv(index=False)

@dsl.component(
    base_image="python:3.9.0",
    packages_to_install=["pandas", "scikit-learn"]
)
def create_dataset(dataset_path: str, insurance_dataset: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(dataset_path)
    with open (insurance_dataset.path, 'w') as f:
        df.to_csv(f,index=False)
    
@dsl.component(
    base_image="python:3.9.0",
    packages_to_install=["pandas", "scikit-learn"]
)
def process_dataset(input_dataset: Input[Dataset], processed_dataset: Output[Dataset]):
    import pandas as pd
    with open(input_dataset.path, 'r') as f:
        df = pd.read_csv(f)
    print (df.head())
    # feature 1: calculate bmi
    df["bmi"] = df["weight"] / (df["height"] **2)
    # Feature 2 : Age group
    # < 25 : young , < 45 : adult , < 60 : middle_aged , > 60 senior
    def age_group(age):
        if age < 25:
            return "young"
        if age < 45:
            return "adult"
        if age < 60:
            return "mid_aged"
        return "senior"
    df["age_group"] = df["age"].apply(age_group)
    # feature 3 : Lifestyle Risk
    def lifestyle_risk(row):
        if row["bmi"] > 30 and row["smoker"]:
            return "high"
        elif row["bmi"] > 20 and row["smoker"]:
            return "medium"
        return "low"
    # return df
    df["lifestyle_risk"] = df.apply(lifestyle_risk, axis=1)
    # Feature 4: city tier
    tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
    tier_2_cities = [
        "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
        "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
        "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
        "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
        "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
        "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
    ]
    def city_tier(city):
        if city in tier_1_cities:
            return 1
        elif city in tier_2_cities:
            return 2
        else:
            return 3
    df["city_tier"] = df["city"].apply(city_tier)
    df = df.drop(columns=['age','weight','height','smoker','city'])
    print ("### Dataset after Pre-processing ###")
    print (df.head())
    with open(processed_dataset.path, 'w') as f:
        df.to_csv(f)
# df_new = load_data()

@dsl.component(
    base_image="python:3.9.0",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_model(input_dataset: Input[Dataset],model: Output[Model]):
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    with open(input_dataset.path, 'r') as f:
        df = pd.read_csv(f)

    X = df[['income_lpa', 'occupation',  'bmi',
       'age_group', 'lifestyle_risk', 'city_tier']]
    y = df[['insurance_premium_category']]

    print("######output####")
    print (y)

    text_columns = ['occupation','age_group','lifestyle_risk','city_tier']
    numeric_columns = ['income_lpa','bmi']

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",OneHotEncoder(),text_columns),
            ("num","passthrough",numeric_columns)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    pipeline.fit(X_train,y_train)
    with open(model.path, 'wb') as f:
        pickle.dump(pipeline, f)



    

@dsl.pipeline(
    name="test-pipeline",
    description="This is a test pipeline",
    # pipeline_root="s3://anirban-kubeflow-s3-bucket/kubeflow-artifacts"
)
def testPipeline(csv_data: str):
    # input_dataset = 'insurance.csv'
    loaded_dataset = create_dataset(dataset_path=csv_data).outputs['insurance_dataset']
    processed_dataset = process_dataset(input_dataset=loaded_dataset).outputs['processed_dataset']
    train_model(input_dataset=processed_dataset)

    
if __name__ == "__main__":
    compiler.Compiler().compile(testPipeline, "test-pipeline.yaml")





