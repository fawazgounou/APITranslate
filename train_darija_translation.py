from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# Charger le dataset Darija
print("📥 Chargement du dataset de traduction Anglais → Darija...")
dataset = load_dataset("atlasia/darija-translation")

# Vérifier la structure du dataset
print("🔍 Exemple de données :", dataset["train"][0])

# Vérifier que la colonne "fr" est vide
if dataset["train"][0]["fr"] is None:
    print("❌ Pas de données Français → Darija ! Entraînement uniquement pour Anglais → Darija.")
    lang_source = "en"
else:
    lang_source = "fr"

# Charger le modèle de base MBart
model_name = "facebook/mbart-large-50"
tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang=f"{lang_source}_XX", tgt_lang="ar_AR")
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Préparation des données avec filtrage des entrées invalides
def preprocess_function(examples):
    source_texts = examples.get(lang_source, "")
    target_texts = examples.get("darija", "")

    # Filtrer les entrées vides ou None
    valid_indices = [i for i in range(len(source_texts)) if source_texts[i] and target_texts[i]]
    source_texts = [source_texts[i] for i in valid_indices]
    target_texts = [target_texts[i] for i in valid_indices]

    # Tokenisation
    inputs = tokenizer(source_texts, truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(target_texts, truncation=True, padding="max_length", max_length=128)

    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokeniser les données
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Configuration des hyperparamètres
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./darija_model_{lang_source}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    push_to_hub=False,
)

# Entraîneur
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Lancer l'entraînement
print(f"🚀 Entraînement du modèle pour {lang_source} → Darija...")
trainer.train()

# Sauvegarder le modèle entraîné
print(f"✅ Modèle entraîné pour {lang_source} → Darija ! Sauvegarde...")
model.save_pretrained(f"./darija_model_{lang_source}")
tokenizer.save_pretrained(f"./darija_model_{lang_source}")
