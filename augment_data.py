import pandas as pd
import random

diseases_symptoms = {
    "COVID-19": [
        "fever", "dry cough", "tiredness", "loss of taste or smell", "difficulty breathing",
        "chest pain", "sore throat", "headache", "muscle or joint pain", "chills"
    ],
    "Malaria": [
        "high fever", "shaking chills", "profuse sweating", "headache", "nausea",
        "vomiting", "abdominal pain", "diarrhea", "muscle pain", "deep fatigue"
    ],
    "Tuberculosis": [
        "persistent cough lasting weeks", "chest pain", "coughing up blood", "fatigue",
        "night sweats", "chills", "fever", "loss of appetite", "weight loss"
    ],
    "Diabetes Type 2": [
        "increased thirst", "frequent urination", "increased hunger", "unintended weight loss",
        "fatigue", "blurred vision", "slow-healing sores", "frequent infections", "numbness in hands or feet"
    ],
    "Migraine": [
        "severe throbbing pain", "pulsing pain on one side of head", "nausea", "vomiting",
        "extreme sensitivity to light", "sensitivity to sound", "visual auras", "lightheadedness"
    ],
    "Appendicitis": [
        "sudden pain on the right side of lower abdomen", "sudden pain around navel shifting to right abdomen",
        "pain that worsens with coughing", "nausea", "vomiting", "loss of appetite", "fever", "abdominal bloating"
    ],
    "Gastroenteritis": [
        "watery diarrhea", "abdominal cramps", "nausea", "vomiting", "muscle aches", "headache", "low-grade fever"
    ],
    "Cholera": [
        "severe watery diarrhea", "vomiting", "leg cramps", "rapid fluid loss", "dehydration", "extreme thirst", "fatigue"
    ],
    "Rabies": [
        "fever", "headache", "excessive salivation", "muscle spasms", "mental confusion", "paralysis"
    ],
    "Measles": [
        "high fever", "cough", "runny nose", "inflamed eyes", "sore throat", "red blotchy skin rash", "Koplik spots"
    ],
    "Mumps": [
        "swollen salivary glands", "pain while chewing or swallowing", "fever", "headache", "muscle aches", "weakness"
    ],
    "Tetanus": [
        "jaw cramping", "muscle spasms", "painful muscle stiffness", "trouble swallowing", "seizures", "headache", "fever"
    ],
    "Hypothyroidism": [
        "fatigue", "increased sensitivity to cold", "constipation", "dry skin", "weight gain", "puffy face", "muscle weakness", "thinning hair"
    ],
    "Hyperthyroidism": [
        "unintentional weight loss", "rapid heartbeat", "irregular heartbeat", "increased appetite", "nervousness", "anxiety", "sweating", "difficulty sleeping"
    ],
    "Iron Deficiency Anemia": [
        "extreme fatigue", "weakness", "pale skin", "chest pain", "fast heartbeat", "shortness of breath", "cold hands and feet", "brittle nails"
    ],
    "Endometriosis": [
        "pelvic pain", "painful periods", "pain with intercourse", "pain with bowel movements", "excessive bleeding", "infertility", "fatigue"
    ],
    "PCOS": [
        "irregular periods", "excess androgen", "polycystic ovaries", "excess facial hair", "severe acne", "male-pattern baldness"
    ],
    "General Anxiety Disorder": [
        "feeling restless", "feeling wound-up", "being easily fatigued", "difficulty concentrating", "irritability", "muscle tension", "sleep problems"
    ],
    "Major Depressive Disorder": [
        "feeling sad or empty", "loss of interest in activities", "changes in appetite", "trouble sleeping", "loss of energy", "feeling worthless", "difficulty thinking"
    ],
    "Asthma": [
        "shortness of breath", "chest tightness", "wheezing when exhaling", "trouble sleeping caused by shortness of breath", "coughing or wheezing attacks"
    ]
}

templates = [
    "I have been experiencing a lot of {s1} recently, and also a bit of {s2}.",
    "I'm really worried because I have {s1}, {s2}, and {s3}.",
    "My main symptoms are {s1} and {s2}. I've had them for a few days.",
    "Lately, I feel {s1}. Also noticing some {s2} and {s3}.",
    "I'm suffering from {s1} and severe {s2}. It is very uncomfortable.",
    "There is a lot of {s1} going on with my body, combined with {s2}.",
    "I've been feeling {s1} constantly. And I also have {s2}.",
    "For the past week, I've had {s1}, {s2}, and a strong sense of {s3}.",
    "My doctor asked me about my symptoms. I told them about the {s1} and {s2}.",
    "Every day I wake up with {s1} and {s2}. Sometime I also experience {s3}."
]

augmented_data = []
all_diseases = list(diseases_symptoms.keys())

for disease in all_diseases:
    symptoms_list = diseases_symptoms[disease]
    for _ in range(40):
        # Pick a random template
        template = random.choice(templates)
        # Pick 3 random distinct symptoms
        s1, s2, s3 = random.sample(symptoms_list, 3)
        sentence = template.format(s1=s1, s2=s2, s3=s3)
        # Append data
        augmented_data.append({"label": disease, "text": sentence})

df_new = pd.DataFrame(augmented_data)

try:
    df_existing = pd.read_csv("Symptom2Disease.csv")
    if 'Unnamed: 0' in df_existing.columns:
        start_idx = df_existing['Unnamed: 0'].max() + 1
    else:
        # Based on index being first col named ''
        start_idx = len(df_existing)
except FileNotFoundError:
    df_existing = pd.DataFrame()
    start_idx = 0

# Set new index
df_new.index = range(start_idx, start_idx + len(df_new))

# Reset index to match the formatting of the existing CSV
if not df_existing.empty:
    first_col = df_existing.columns[0] # Usually 'Unnamed: 0'
    df_new[first_col] = df_new.index

df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Important: recreate the initial row index if to persist same structure
df_combined.to_csv("Symptom2Disease.csv", index=False)

print(f"Generated {len(df_new)} new rows.")
print(f"Total rows in dataset now: {len(df_combined)}")
print(f"Unique diseases now: {len(df_combined['label'].unique())}")
