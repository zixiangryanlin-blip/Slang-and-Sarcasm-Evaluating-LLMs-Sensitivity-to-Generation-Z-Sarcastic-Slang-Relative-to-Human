#This file of code is executed in Google Colab

from openai import OpenAI
import pandas as pd
import re
import os
import csv
from google.colab import drive

try:
   drive.mount('/content/drive')
except ValueError:
   drive.mount('/content/drive', force_remount=True)

BASE_PATH = "" # Drive access removed
INPUT_CSV = os.path.join(BASE_PATH, "Slangcastic_200.csv")
df_input = pd.read_csv(INPUT_CSV)
sentences = df_input["Sentence"].astype(str).tolist()
print(f"Loaded {len(sentences)} sentences from CSV.")

client = OpenAI(api_key="")  # API key removed

raw_output = ""
data = []

for sentence in sentences:
   prompt = f"""
   Classify the sarcasm polarity of the tweet as Sarcastic or Not Sarcastic.
   Assign a numerical value between -5 (disgust/extreme discontent) and 5 (obvious joy/extreme pleasure).
   Tweet: "{sentence}"
   Intensity Score:
   """

   response = client.chat.completions.create(
       model="gpt-4.1", # switch to GPT-5.1 by replacing "gpt-4.1" to "gpt-5.1"
       messages=[{"role": "user", "content": prompt}],
       temperature=0.0
   )
   reply = response.choices[0].message.content
   raw_output += f"{sentence}\n{reply}\n{'-'*20}\n"
   print(f"{sentence}\n{reply}\n{'-'*20}\n")
   polarity_match = re.search(r"(Sarcastic|Not Sarcastic)", reply)
   score_match = re.search(r"(-?\d+)", reply)
   polarity = polarity_match.group(0) if polarity_match else "Unknown"
   score = int(score_match.group(0)) if score_match else None
   data.append([sentence, polarity, score])

df = pd.DataFrame(
   data,
   columns=["Sentence", "Sarcasm Polarity", "Intensity Score"]
)

print("\nFinal Table:\n")
print(df.to_string(index=False))


def save_raw_output_csv_lines(raw_output_str, filenameSTR):
   filepath = os.path.join(BASE_PATH, filenameSTR + "_raw_output.csv")
   lines = raw_output_str.strip().split("\n")
   with open(filepath, mode="w", newline="", encoding="utf-8-sig") as file:
       writer = csv.writer(file)
       writer.writerow(["Line"])
       for line in lines:
           writer.writerow([line])
   print(f"Raw output saved to: {filepath}")


def save_dataframe_csv(df, filenameSTR):
   filepath = os.path.join(BASE_PATH, filenameSTR + "_table.csv")
   df.to_csv(filepath, index=False, encoding="utf-8-sig")
   print(f"DataFrame saved to: {filepath}")

save_raw_output_csv_lines(raw_output, "Slangcastic_GPT-4.1")
save_dataframe_csv(df, "Slangcastic_GPT-4.1")
