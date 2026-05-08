# Slang-and-Sarcasm-Evaluating-LLMs-Sensitivity-to-Generation-Z-Sarcastic-Slang-Relative-to-Human
This repository serves as a complete archive of the data, code, and methodology used to investigate how well Large Language Models (LLMs) detect and interpret Generation Z sarcastic slang compared to human baselines in sarcasm detection and sentiment analysis tasks.

What's Included in This Repository:
1. **Code folder:** Python scripts used to interface with the LLMs;
2. **Slangcastic Dataset:** 200 sarcastic and non-sarcastic tweets;
3. **Slangcastic Dataset and Annotations:** 200 sarcastic and non-sarcastic tweets with LLMs' side-by-side outputs and human annotations;
4. **Human Annotation Questionnaire folder:**  Online Google Form used to gather the human baseline data. This includes the evaluation rubric and the samplers used to ensure consistent grading among participants, and notes that the questionnaire is separated into two PDFs. Please download the PDF if the preview is not available.

**Abstract:**
As younger generations increasingly use slang and sarcasm in digital communication, it is crucial for Large Language Models (LLMs) to accurately interpret these nuanced linguistic phenomena. This study investigates the extent to which six contemporary LLMs (GPT-4.1, GPT-5.1, GPT-OSS-20b, GPT-OSS-120b, Gemini-2.5-Flash, and Gemini-2.5-Pro) align with eight Generation Z human annotators in detecting sarcasm and evaluating sentiment within tweets containing sarcastic slang. To facilitate this, the research introduces SLANGCASTIC, a fine-grained English dataset comprising 200 tweets embedded with Generation Z sarcastic and non-sarcastic slang. Results show that while newer models like GPT-5.1 demonstrate strong alignment, all evaluated LLMs systematically over-annotate sarcasm and perceive more negative sentiment than humans. This discrepancy arises because LLMs tend to perform excessive reasoning and over-analyze semantic details. In contrast, human annotators rely on holistic, pragmatic reasoning to infer the speaker's true intent, highlighting the need for better cultural and contextual understanding in AI.
