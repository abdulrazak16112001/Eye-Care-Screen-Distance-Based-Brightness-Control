import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


print("📁 Loading dataset...")

df = pd.read_csv("eye_distance_log.csv")


if df.empty or "status" not in df.columns:
    print("❌ CSV empty. Run app.py first.")
    exit()


# Convert time

df["time_seconds"] = df["time"].apply(
    lambda t: int(t.split(":")[0])*3600 +
              int(t.split(":")[1])*60 +
              int(t.split(":")[2])
)


# Encode

encoder = LabelEncoder()
df["status_encoded"] = encoder.fit_transform(df["status"])


X = df[["time_seconds"]]
y = df["status_encoded"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


print("🤖 Training...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluate

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc*100:.2f}%")


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)

plt.show()


# Save

joblib.dump(model, "eye_distance_model.joblib")
joblib.dump(encoder, "label_encoder.joblib")

print("💾 Model Saved")


# Test prediction

t = time.strftime("%H:%M:%S")
h, m, s = t.split(":")

sec = int(h)*3600 + int(m)*60 + int(s)

p = model.predict([[sec]])[0]

label = encoder.inverse_transform([p])[0]

print("🔮 Now Prediction:", label)
