import numpy as np
import pandas as pd
import re
import string
import csv
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load datasets with error handling
try:
    fake_data = pd.read_csv("Fake.csv", encoding="utf-8", on_bad_lines="skip", quoting=csv.QUOTE_NONE)
    true_data = pd.read_csv("True.csv", encoding="utf-8", on_bad_lines="skip", quoting=csv.QUOTE_NONE)
except FileNotFoundError:
    print("Error: CSV file(s) not found. Please check the file path.")
    exit()
except pd.errors.ParserError as e:
    print(f"CSV Parsing Error: {e}")
    exit()

# Check if files are loaded properly
print("Fake News Sample:")
print(fake_data.head())
print("True News Sample:")
print(true_data.head())

# Add labels
fake_data['class'] = 0
true_data['class'] = 1

# Merge and shuffle data
data_merge = pd.concat([fake_data, true_data], axis=0).dropna().sample(frac=1).reset_index(drop=True)

# Ensure 'text' column exists
if "text" not in data_merge.columns:
    print("Error: 'text' column missing in dataset!")
    exit()

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    return text

data_merge['text'] = data_merge['text'].apply(lambda x: wordopt(str(x)))

x = data_merge['text']
y = data_merge['class']
if len(x) == 0 or len(y) == 0:
    print("Error: Dataset is empty after preprocessing.")
    exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
GB = GradientBoostingClassifier(random_state=42)
GB.fit(xv_train, y_train)
RF = RandomForestClassifier(random_state=42)
RF.fit(xv_train, y_train)

def output_label(n):
    return "Fake News" if n == 0 else "Not Fake News"

def manual_testing(news):
    if not vectorization.vocabulary_:
        return "Error: Vectorizer not fitted properly."
    new_test = pd.DataFrame({"text": [wordopt(news)]})
    new_xv_test = vectorization.transform(new_test["text"])
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    return f"\nLR Prediction: {output_label(pred_LR[0])}\nDT Prediction: {output_label(pred_DT[0])}\nGBC Prediction: {output_label(pred_GB[0])}\nRFC Prediction: {output_label(pred_RF[0])}"

def check_news():
    news = text_input.get("1.0", tk.END).strip()
    if not news:
        result_display.config(state=tk.NORMAL)
        result_display.delete("1.0", tk.END)
        result_display.insert(tk.INSERT, "Please enter some text.")
        result_display.config(state=tk.DISABLED)
        return
    result = manual_testing(news)
    result_display.config(state=tk.NORMAL)
    result_display.delete("1.0", tk.END)
    result_display.insert(tk.INSERT, result)
    result_display.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Fake News Detector")
root.geometry("800x600")
root.configure(bg="#D4ECDD")  # Soft pastel green

try:
    image = Image.open("cute_ai_bot.png")
    image = image.resize((120, 120), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(root, image=photo, bg="#D4ECDD")
    img_label.pack(pady=10)
except:
    pass

label = tk.Label(root, text="Enter News Text:", font=("Poppins", 18, "bold"), bg="#D4ECDD", fg="#4A7C59")
label.pack(pady=10)

text_input = scrolledtext.ScrolledText(root, width=70, height=5, font=("Arial", 12), bg="#FFFFFF", relief="solid", bd=2, wrap=tk.WORD)
text_input.pack(padx=20)

check_button = tk.Button(root, text="Check News", command=check_news, font=("Arial", 14, "bold"), bg="#6B9080", fg="white", relief="ridge", bd=3, padx=10, pady=5)
check_button.pack(pady=10)

result_display = scrolledtext.ScrolledText(root, width=70, height=5, font=("Arial", 12), bg="#FFFFFF", relief="solid", bd=2, wrap=tk.WORD, state=tk.DISABLED)
result_display.pack(padx=20, pady=10)

root.mainloop()
