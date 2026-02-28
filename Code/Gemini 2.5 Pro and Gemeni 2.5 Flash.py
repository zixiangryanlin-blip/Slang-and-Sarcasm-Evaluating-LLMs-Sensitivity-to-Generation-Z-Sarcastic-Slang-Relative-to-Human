#This file of code is executed in Google Colab

pip install -q -U google-genai

import google.generativeai as genai

from google.colab import userdata
userdata.get('GOOGLE_API_KEY')
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

from google import genai
client = genai.Client(api_key= ) # API key removed

import pandas as pd
import re
import os
import csv
from google.colab import drive
from google.genai import types

try:
   drive.mount('/content/drive')
except ValueError:
   drive.mount('/content/drive', force_remount=True)


BASE_PATH = "/content/drive/MyDrive/AP/"
INPUT_CSV = os.path.join(BASE_PATH, "Slangcastic_200.csv")
df_input = pd.read_csv(INPUT_CSV)
sentences = df_input["Sentence"].astype(str).tolist()
print(f"Loaded {len(sentences)} sentences from CSV.")

raw_output = ""
data = []


for sentence in sentences:
   prompt = f"""
   Classify the sarcasm polarity of the tweet as Sarcastic or Not Sarcastic.
   Assign a numerical value between -5 (disgust/extreme discontent) and 5 (obvious joy/extreme pleasure).
   Tweet: "{sentence}"
   Intensity Score:
   """

   response = client.models.generate_content(
      model="gemini-2.5-pro", # switch to gemini-2.5-flash by replacing "gemini-2.5-pro" to "gemini-2.5-flash"
      contents=prompt,
      config=types.GenerateContentConfig(
         temperature = 0.0
      )
     
   )

   reply = response.text

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


save_raw_output_csv_lines(raw_output, "Slangcastic_Gemini_2.5_pro.csv")
save_dataframe_csv(df, "Slangcastic_Gemini_2.5_pro")
