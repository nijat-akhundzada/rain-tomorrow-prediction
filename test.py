import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, is_train=True):
    df = df.copy()

    # Impute Categorical (Mode)
    for col in categorical_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df.groupby("Location")[col].transform(
                    lambda s: s.mode()[0] if not s.mode().empty else None
                )
            )

    for col in categorical_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Impute Numerical (Mean)
    for col in numerical_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df.groupby("Location")[col].transform(
                    lambda s: s.mean() if not s.mean() else None
                )
            )

    for col in numerical_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Map RainToday
    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0}).fillna(0)

    return df


train_df = pd.read_csv("Train_data.csv")
test_df = pd.read_csv("Test_data.csv")

test_ids = test_df["Unnamed: 0"] if "Unnamed: 0" in test_df.columns else test_df.index
model = joblib.load("random_forest_rain_prediction_model.pkl")


test_ids = test_df["Unnamed: 0"] if "Unnamed: 0" in test_df.columns else test_df.index

if "Unnamed: 0" in train_df.columns:
    train_df.drop(["Unnamed: 0"], axis=1, inplace=True)
if "Unnamed: 0" in test_df.columns:
    test_df.drop(["Unnamed: 0"], axis=1, inplace=True)

X_train_raw = train_df.drop(columns=["RainTomorrow"])
X_test_raw = test_df.copy()

categorical_features = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X_train_raw.select_dtypes(include=["number"]).columns.tolist()


X_train_clean = preprocess_data(X_train_raw)
X_test_clean = preprocess_data(X_test_raw)

train_objs_num = len(X_train_clean)
dataset = pd.concat(objs=[X_train_clean, X_test_clean], axis=0)
dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

X_train_encoded = dataset[:train_objs_num]
X_test_encoded = dataset[train_objs_num:]


scaler = StandardScaler()
X_train_encoded[numerical_features] = scaler.fit_transform(X_train_encoded[numerical_features])
X_test_encoded[numerical_features] = scaler.transform(X_test_encoded[numerical_features])

pca = PCA(n_components=34)  # The model expects exactly 34 components
X_train_pca = pca.fit_transform(X_train_encoded)
X_test_pca = pca.transform(X_test_encoded)

X_test_final = pd.DataFrame(X_test_pca, index=X_test_encoded.index)

predictions = model.predict(X_test_final)

output = pd.DataFrame({"id": test_ids, "RainTomorrow": predictions})
output = output.sort_values(by="id").reset_index(drop=True).drop(["id"], axis=1)
output.to_csv("submission.csv", index=False)
