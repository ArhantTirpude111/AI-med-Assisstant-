"""Curated medical lexicons used by rule-based extraction, NER fallback, and
guardrails.

These are intentionally *small, high-precision* gazetteers — enough to make the
rule-based path useful when SciSpaCy models are not installed. In production
these would be backed by ontologies (SNOMED-CT, LOINC, RxNorm, ICD-10) loaded
from a terminology service; the structure here mirrors that design so the swap
is mechanical.
"""

from __future__ import annotations

# --- Section headers commonly found in clinical reports --------------------
SECTION_HEADERS: dict[str, list[str]] = {
    "diagnosis": ["diagnosis", "diagnoses", "impression", "assessment", "dx", "findings"],
    "medication": ["medication", "medications", "current medications", "rx", "prescription", "drugs"],
    "symptom": ["symptoms", "complaints", "chief complaint", "presenting complaint", "history of present illness"],
    "recommendation": ["recommendation", "recommendations", "plan", "follow-up", "follow up", "advice", "instructions"],
    "physician_note": ["notes", "physician notes", "comments", "remarks", "clinical notes"],
    "lab": ["laboratory", "lab results", "lab findings", "investigations", "results"],
}

# --- Common laboratory analytes (name -> canonical) ------------------------
COMMON_LAB_TESTS: set[str] = {
    "hemoglobin", "hgb", "hct", "hematocrit", "wbc", "rbc", "platelet", "platelets",
    "mcv", "mch", "mchc", "rdw", "neutrophils", "lymphocytes", "monocytes", "eosinophils",
    "glucose", "fasting glucose", "hba1c", "a1c", "creatinine", "bun", "egfr", "urea",
    "sodium", "potassium", "chloride", "bicarbonate", "calcium", "magnesium", "phosphorus",
    "alt", "sgpt", "ast", "sgot", "alp", "alkaline phosphatase", "bilirubin", "albumin",
    "total protein", "ggt", "ldh", "ck", "troponin", "bnp",
    "cholesterol", "ldl", "hdl", "triglycerides", "tsh", "t3", "t4", "free t4",
    "vitamin d", "vitamin b12", "ferritin", "iron", "tibc", "crp", "esr", "psa", "inr",
}

# --- Medication signals ----------------------------------------------------
MEDICATION_SUFFIXES: tuple[str, ...] = (
    "cillin", "mycin", "prazole", "olol", "sartan", "pril", "statin", "dipine",
    "formin", "azepam", "azole", "tinib", "mab", "parin", "oxetine", "pam",
    "vir", "cycline", "floxacin", "azine", "caine", "done", "ide",
)
COMMON_MEDICATIONS: set[str] = {
    "aspirin", "acetaminophen", "paracetamol", "ibuprofen", "metformin", "atorvastatin",
    "simvastatin", "lisinopril", "amlodipine", "losartan", "metoprolol", "omeprazole",
    "pantoprazole", "levothyroxine", "insulin", "warfarin", "clopidogrel", "amoxicillin",
    "azithromycin", "ciprofloxacin", "prednisone", "albuterol", "gabapentin", "sertraline",
    "furosemide", "hydrochlorothiazide", "ramipril", "rosuvastatin", "enalapril",
    "glimepiride", "sitagliptin", "empagliflozin", "heparin", "enoxaparin",
}

# --- Symptoms --------------------------------------------------------------
SYMPTOMS: set[str] = {
    "fever", "cough", "fatigue", "headache", "nausea", "vomiting", "diarrhea",
    "constipation", "dizziness", "shortness of breath", "dyspnea", "chest pain",
    "abdominal pain", "back pain", "joint pain", "swelling", "edema", "weight loss",
    "weight gain", "night sweats", "palpitations", "rash", "itching", "jaundice",
    "blurred vision", "numbness", "tingling", "weakness", "anxiety", "insomnia",
    "loss of appetite", "malaise", "chills", "sore throat", "muscle pain",
}

# --- Diseases / conditions -------------------------------------------------
DISEASES: set[str] = {
    "diabetes", "diabetes mellitus", "type 2 diabetes", "hypertension", "hyperlipidemia",
    "hepatic steatosis", "fatty liver", "cirrhosis", "hepatitis", "anemia", "iron deficiency",
    "hypothyroidism", "hyperthyroidism", "asthma", "copd", "pneumonia", "bronchitis",
    "myocardial infarction", "heart failure", "atrial fibrillation", "coronary artery disease",
    "chronic kidney disease", "acute kidney injury", "urinary tract infection",
    "osteoarthritis", "rheumatoid arthritis", "osteoporosis", "gerd", "peptic ulcer",
    "depression", "anxiety disorder", "migraine", "stroke", "obesity", "metabolic syndrome",
    "vitamin d deficiency", "hypokalemia", "hyperkalemia", "hyponatremia", "leukocytosis",
    "thrombocytopenia", "neutropenia", "dyslipidemia",
}

# --- Anatomy ---------------------------------------------------------------
ANATOMY: set[str] = {
    "liver", "kidney", "heart", "lung", "lungs", "brain", "pancreas", "spleen",
    "gallbladder", "thyroid", "stomach", "intestine", "colon", "bladder", "prostate",
    "bone", "spine", "artery", "vein", "aorta", "lymph node",
}

# --- Emergency / red-flag phrases (guardrails) -----------------------------
EMERGENCY_TERMS: set[str] = {
    "chest pain", "crushing chest pain", "difficulty breathing", "shortness of breath at rest",
    "severe bleeding", "uncontrolled bleeding", "loss of consciousness", "unconscious",
    "stroke", "facial drooping", "slurred speech", "sudden weakness", "sudden numbness",
    "suicidal", "suicide", "self-harm", "overdose", "anaphylaxis", "severe allergic reaction",
    "sepsis", "septic shock", "seizure", "stiff neck with fever", "coughing up blood",
    "blood in vomit", "severe abdominal pain", "high fever", "meningitis",
}
