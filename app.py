from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration CORS pour les requêtes React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# 📥 Chargement des modèles de traduction
print("📥 Chargement des modèles de traduction...")

# Correction des modèles Helsinki-NLP en utilisant ceux qui existent réellement
translation_models = {
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "nl-en": "Helsinki-NLP/opus-mt-nl-en",
    "tr-en": "Helsinki-NLP/opus-mt-tr-en",
    "pl-en": "Helsinki-NLP/opus-mt-pl-en",
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "ko-en": "Helsinki-NLP/opus-mt-ko-en",
    "sv-en": "Helsinki-NLP/opus-mt-sv-en",
    "mul-en": "Helsinki-NLP/opus-mt-mul-en",  # Remplacement pour les langues non spécifiques

    "en-darija": "BAKKALIAYOUB/DarijaTranslation-V1",
}

# Charger les modèles et tokenizers
translation_pipelines = {}
for key, model_name in translation_models.items():
    print(f"🔹 Chargement du modèle {key} : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_pipelines[key] = pipeline("translation", model=model_name, tokenizer=tokenizer)

print("✅ Tous les modèles de traduction sont prêts !")

@app.get("/translate/")
def translate(text: str, source_lang: str):
    """
    API de traduction multi-langue.
    - `source_lang` : Langue source à traduire en Darija.
    """
    try:
        if not text.strip():
            return {"error": "Le texte d'entrée est vide."}

        print(f"🔹 Texte reçu ({source_lang} → Darija) : {text}")

        if source_lang == "darija":
            return {"error": "La traduction Darija → Darija n'est pas nécessaire."}

        key = f"{source_lang}-en"
        if key not in translation_pipelines:
            return {"error": f"Traduction non supportée : {source_lang} → en"}

        intermediate_text = translation_pipelines[key](text)[0]["translation_text"]
        print(f"🛠️ Traduction intermédiaire ({source_lang} → En) : {intermediate_text}")

        translated_text = translation_pipelines["en-darija"](intermediate_text)[0]["translation_text"]
        print(f"✅ Traduction finale (En → Darija) : {translated_text}")

        return {"original": text, "intermediate": intermediate_text, "translated": translated_text}

    except Exception as e:
        print(f"⚠️ Erreur pendant la traduction : {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
