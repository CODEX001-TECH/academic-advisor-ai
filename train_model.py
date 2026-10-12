import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ["cgpa", "attendance_pct", "study_hours_per_day", "failed_courses", "credits_completed"]
df = pd.read_csv("student_data.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["risk_level"], test_size=0.2, random_state=303, stratify=df["risk_level"]
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=303, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
with open("academic_advisor_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved academic_advisor_model.pkl")
