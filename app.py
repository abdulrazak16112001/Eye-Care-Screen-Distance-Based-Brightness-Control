import cv2
import pyttsx3
import time
import pandas as pd
import matplotlib.pyplot as plt
import screen_brightness_control as sbc
import joblib
from collections import deque


# ---------------- SETUP ---------------- #

engine = pyttsx3.init()
engine.setProperty('rate', 150)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# Load ML model
try:
    model = joblib.load("eye_distance_model.joblib")
    encoder = joblib.load("label_encoder.joblib")
    print("✅ AI Model Loaded")
except:
    model = None
    encoder = None
    print("⚠️ AI Model Not Found")


# Camera (change to 1 if 0 not works)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()


data = []
area_buffer = deque(maxlen=5)
last_brightness = 100

last_voice = 0
cooldown = 5


print("🚀 Started (Press Q to Quit)")


# ---------------- MAIN LOOP ---------------- #

while True:

    ret, frame = cap.read()

    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Safe"
    now = time.time()


    for (x, y, w, h) in faces:

        area = w * h
        area_buffer.append(area)

        cv2.rectangle(
            frame, (x, y), (x+w, y+h),
            (255, 0, 0), 2
        )


        # -------- Distance Check -------- #

        if area > 40000:

            status = "Close"

            cv2.putText(frame, "Too Close!",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            if last_brightness != 30:
                try:
                    sbc.set_brightness(30)
                    last_brightness = 30
                except:
                    pass


            if now-last_voice > cooldown:

                engine.say("Please move back")
                engine.runAndWait()
                last_voice = now


        else:

            cv2.putText(frame, "Safe",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            if last_brightness != 100:
                try:
                    sbc.set_brightness(100)
                    last_brightness = 100
                except:
                    pass


        # -------- AI Prediction -------- #

        if model and encoder:

            t = time.strftime("%H:%M:%S")
            h, m, s = t.split(":")

            sec = int(h)*3600 + int(m)*60 + int(s)

            pred = model.predict([[sec]])[0]
            label = encoder.inverse_transform([pred])[0]

            if label == "Close":

                cv2.putText(frame, "AI: Warning!",
                            (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2)


    # -------- Save Frame Data -------- #

    data.append({
        "time": time.strftime("%H:%M:%S"),
        "status": status
    })


    cv2.imshow("Eye Distance Monitor", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------- CLEANUP ---------------- #

cap.release()
cv2.destroyAllWindows()


# ---------------- SAVE CSV ---------------- #

df = pd.DataFrame(data)

if df.empty:
    print("❌ No data collected. Run longer.")
    exit()


df.to_csv("eye_distance_log.csv", index=False)

print("📁 CSV Saved")


# ---------------- ANALYSIS ---------------- #

if "status" not in df.columns:
    print("❌ Invalid CSV")
    exit()


total = len(df)

close = len(df[df["status"] == "Close"])
safe = len(df[df["status"] == "Safe"])


cp = (close/total)*100
sp = (safe/total)*100


print(f"Close: {cp:.2f}% | Safe: {sp:.2f}%")


plt.figure()
plt.bar(["Close", "Safe"], [cp, sp])
plt.title("Distance Analysis")
plt.show()


plt.figure()
plt.pie([cp, sp],
        labels=["Close", "Safe"],
        autopct="%1.1f%%")

plt.show()
