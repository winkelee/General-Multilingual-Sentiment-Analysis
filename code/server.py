import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import joblib
import re
import langcodes
from contextlib import asynccontextmanager
import warnings
from nltk.tokenize import WordPunctTokenizer, sent_tokenize
import unicodedata

#model_name = "facebook/m2m100_418M"
model_name = "facebook/nllb-200-distilled-600M"
import nltk
nltk.download('punkt_tab')

MODEL_LANGUAGE_PATH = "D:/programming stuff/NLP_NN/code/naive_bayes_bow_language_classification.pkl"
COUNT_VECTORIZER_PATH = "D:/programming stuff/NLP_NN/code/count_vectorizer.pkl"
MODEL_SENTIMENT_PATH = "D:/programming stuff/NLP_NN/code/sentiment_classifier_gru.pth"
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
TOXIC_CLASS_NAMES = ["Non-Toxic", "Toxic"]
TOKENIZER_DATA_PATH = "D:/programming stuff/NLP_NN/code/tokenizer_data.pkl"
TOXIC_TOKENIZER_DATA_PATH = "D:/programming stuff/NLP_NN/code/toxic_tokenizer_data.pkl"
TOXIC_COUNT_VECTORIZER_PATH = "D:/programming stuff/NLP_NN/code/toxic_count_vectorizer.pkl"
TOXIC_MODEL_PATH = "D:/programming stuff/NLP_NN/code/sgd_toxicity_classification.pkl"

model = None
model_language = None
model_sentiment = None
tokenizer = None
count_vectorizer = None
tokens = None
token_to_id = None
word_tokenizer = WordPunctTokenizer()
translation_needed = False

iso_to_nllb = {
    'ar': 'arb_Arab',
    'be': 'bel_Cyrl',
    'ber': 'zgh_Tfng',
    'bg': 'bul_Cyrl',
    'ckb': 'ckb_Arab',
    'cs': 'ces_Latn',
    'da': 'dan_Latn',
    'nl': 'nld_Latn',
    'en': 'eng_Latn',
    'eo': 'epo_Latn',
    'fi': 'fin_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'he': 'heb_Hebr',
    'hi': 'hin_Deva',
    'hu': 'hun_Latn',
    'is': 'isl_Latn',
    'id': 'ind_Latn',
    'ia': 'ina_Latn',
    'fa': 'pes_Arab',
    'it': 'ita_Latn',
    'ja': 'jpn_Jpan',
    'kab': 'kab_Latn',
    'tlh': 'tlh_Latn',
    'la': 'lat_Latn',
    'lfn': 'lfn_Latn',
    'lt': 'lit_Latn',
    'jbo': 'jbo_Latn',
    'nds': 'nds_Latn',
    'mk': 'mkd_Cyrl',
    'zh': 'zho_Hans',
    'mr': 'mar_Deva',
    'el': 'ell_Grek',
    'nb': 'nob_Latn',
    'pl': 'pol_Latn',
    'pt': 'por_Latn',
    'ro': 'ron_Latn',
    'ru': 'rus_Cyrl',
    'sk': 'slk_Latn',
    'es': 'spa_Latn',
    'sv': 'swe_Latn',
    'tl': 'tgl_Latn',
    'tt': 'tat_Cyrl',
    'tok': 'tok_Latn',
    'tr': 'tur_Latn',
    'uk': 'ukr_Cyrl',
    'vi': 'vie_Latn'
}

warnings.filterwarnings('ignore', category=DeprecationWarning)

def load_tokenizer_data(path):
    with open(path, "rb") as f:
        tokens, token_to_id = pickle.load(f)
    return tokens, token_to_id

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class SentimentClassifierGRU(nn.Module):
    def __init__(self, vocabulary_dimention, num_classes, hidden_dim, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_dimention, 300)

        self.embedding_dropout = nn.Dropout(0.5)
        self.rnn = nn.GRU(300, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.rnn_dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = x.permute(1, 0, 2)
        _, h_n = self.rnn(x)
        if self.rnn.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]

        final_hidden = self.rnn_dropout(final_hidden)
        out = self.fc(final_hidden)
        return out

class InputText(BaseModel):
    source_text: str

def load_models():
    global model, model_language, model_sentiment, tokenizer, token_to_id, tokens, count_vectorizer, token_to_id_toxic, tokens_toxic, toxic_count_vectorizer, model_toxic
    try:
        print(f"Loading tokenizer data: {TOKENIZER_DATA_PATH}")
        tokens, token_to_id = load_tokenizer_data(TOKENIZER_DATA_PATH)
        print(f"Loading toxicity tokenizer data: {TOXIC_TOKENIZER_DATA_PATH}")
        tokens_toxic, token_to_id_toxic = load_tokenizer_data(TOXIC_TOKENIZER_DATA_PATH)
        print(f"Loading sentiment classification model from: {MODEL_SENTIMENT_PATH}")
        print(f"Loaded device: {device}")
        model_sentiment = SentimentClassifierGRU(vocabulary_dimention=len(tokens), num_classes=3, hidden_dim=256, num_layers=4)
        model_sentiment.load_state_dict(torch.load(MODEL_SENTIMENT_PATH, map_location=torch.device(device)))
        model_sentiment.eval()
        print(f"Loading language classification model from: {MODEL_LANGUAGE_PATH}")
        model_language = joblib.load(MODEL_LANGUAGE_PATH)
        print(f"Loading translation model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"Loading count vectorizer: {COUNT_VECTORIZER_PATH}")
        count_vectorizer = joblib.load(COUNT_VECTORIZER_PATH)
        print(f"Loading toxic count vectorizer: {TOXIC_COUNT_VECTORIZER_PATH}")
        toxic_count_vectorizer = joblib.load(TOXIC_COUNT_VECTORIZER_PATH)
        print(f"Loading toxicity classifier model: {TOXIC_MODEL_PATH}")
        model_toxic = joblib.load(TOXIC_MODEL_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load models.")
        print(e)
        model = None
        model_language = None
        model_sentiment = None
        model_toxic = None

def translate(text, source_lang_code, target_lang_code, tokenizer, model):
    if source_lang_code == 'sr':
        type = detect_serbian_script(text)
        if type == 'cyrillic':
            langcode_nllb = 'srp_Cyrl'
        else:
            langcode_nllb = 'srp_Latn'
    else:
        langcode_nllb = iso_to_nllb[source_lang_code]
    tokenizer.src_lang = langcode_nllb
    encoded = tokenizer(text, return_tensors='pt')
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code), max_new_tokens=800, do_sample=False, num_beams=3)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def detect_serbian_script(text):
    cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')
    cyrillic_count = len(cyrillic_pattern.findall(text))
    total_letter_count = len(re.findall(r'\w', text))

    if cyrillic_count / max(total_letter_count, 1) > 0.5:
        return 'cyrillic'
    else:
        return 'latin'

def clean_input(input):
    input = str(input)
    input = input.lower()
    input = re.sub(r'<.*?>', '', input)
    input = re.sub(r'https?://\S+|www\.\S+', '', input)
    input = ''.join(c for c in input if unicodedata.category(c).startswith('L') or c.isspace())
    input = re.sub(r'\s+', ' ', input).strip()
    return input

def encode_text(text, token_to_id, max_len=100):
    words = text.split()
    token_ids = [token_to_id.get(word, token_to_id.get("<UNK>", 0)) for word in words]
    
    if len(token_ids) < max_len:
        token_ids += [token_to_id.get("<PAD>", 0)] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
        
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    return prediction.item(), confidence.item()

def full_prediction_pipeline(input):
        #Returns predicted language,
        translation_needed = False
        input = str(input)
        print("Got inference input: ", input)
        sentences = sent_tokenize(input)
        translated_sentences = []
        for sentence in sentences:
            print("Input sentence: ", sentence)
            sentence = ' '.join(word_tokenizer.tokenize(sentence))
            clean_sentence = clean_input(sentence)
            list_clean_sentence = [clean_sentence]
            prediction_sentence = count_vectorizer.transform(list_clean_sentence)
            prediction_sentence = model_language.predict(prediction_sentence)
            print("Inference sentence language prediction: ", prediction_sentence)
            language_code_sentence = langcodes.find(prediction_sentence[0]).language
            if language_code_sentence != 'en':
                translation_needed = True
                translated_input_sentence = translate(clean_sentence, language_code_sentence, 'eng_Latn', tokenizer, model)
                print("Inference translated sentence: ", translated_input_sentence)
            else:
                translated_input_sentence = clean_sentence
                print("Translation not needed, skipping sentence.")
            translated_sentences.append(translated_input_sentence)
        input_text = ' '.join(word_tokenizer.tokenize(input))
        print("Parsed with WordPunctTokenizer: ", input_text)
        clean_input_text = clean_input(input_text)
        print('Cleaned: ', clean_input_text)
        list_clean_input = [clean_input_text]
        prediction_input = count_vectorizer.transform(list_clean_input)
        prediction_input = model_language.predict(prediction_input)
        print("Inference language prediction: ", prediction_input)
        language_code = langcodes.find(prediction_input[0]).language
        print("Inference predicted ISO language code: ", language_code)
        #if language_code != 'en':
        #    translation_needed = True
        #    translated_input = translate(clean_input_text, language_code, 'eng_Latn', tokenizer, model)
        #    print("Inference translated sample input: ", translated_input)
        #else:
        #    translated_input = clean_input_text
        #    print("Translation not needed, skipping step.")
        translated_input = ' '.join(translated_sentences)
        clean_translated_input = clean_input(translated_input)
        encoded_input = encode_text(clean_translated_input, token_to_id)
        prediction_idx, confidence = predict(model_sentiment, encoded_input)
        prediction_class = CLASS_NAMES[prediction_idx]
        print("Inference naive sentiment prediction: ", prediction_class)
        print("With confidence: ", confidence)
        list_translated_input = [clean_translated_input]
        toxicity_prediction = toxic_count_vectorizer.transform(list_translated_input)
        toxicity_prediction, toxicity_conficence = predict_toxicity(model_toxic, toxicity_prediction)
        toxicity_class = TOXIC_CLASS_NAMES[toxicity_prediction]
        print("Inference toxicity sentiment prediction: ", toxicity_class)
        print("With confidence: ", toxicity_conficence)
        if toxicity_class == 'Toxic' or prediction_class == 'Negative':
            general_sentiment = 'Negative'
        elif toxicity_class == 'Non-Toxic' and prediction_class == "Neutral":
            general_sentiment = 'Neutral'
        elif toxicity_class == 'Non-Toxic' and prediction_class == "Positive":
            general_sentiment = 'Positive'
        return prediction_input[0], language_code, translation_needed, translated_input, prediction_idx, prediction_class, confidence, toxicity_class, toxicity_conficence, general_sentiment

def predict_toxicity(model_toxic, input):
    probas = model_toxic.predict_proba(input)
    confidence = probas.max()
    prediction = probas.argmax()
    return prediction, confidence

def models_warm_up():
    try:
        sample_input = 'Ceci est un morceau de texte pour réchauffer les modèles.' #This is a sample text to warm up the models.
        clean_sample_input = clean_input(sample_input)
        list_clean_sample_input = [clean_sample_input]
        prediction_sample_input = count_vectorizer.transform(list_clean_sample_input)
        prediction_sample_input = model_language.predict(prediction_sample_input)
        print("Sample language prediction: ", prediction_sample_input)
        sample_language_code = langcodes.find(prediction_sample_input[0]).language
        print("Predicted language code: ", iso_to_nllb[sample_language_code])
        translated_input = translate(clean_sample_input, sample_language_code, 'eng_Latn', tokenizer, model)
        print("Translated sample input: ", translated_input)
        clean_translated_sample_input = clean_input(translated_input)
        encoded_sample_input = encode_text(clean_translated_sample_input, token_to_id)
        prediction_idx, confidence = predict(model_sentiment, encoded_sample_input)
        prediction_class = CLASS_NAMES[prediction_idx]
        list_translated_sample_input = [clean_translated_sample_input]
        prediction_toxic_sample_input = toxic_count_vectorizer.transform(list_translated_sample_input)
        prediction_toxic_sample_input, toxicity_confidence = predict_toxicity(model_toxic, prediction_toxic_sample_input)
        print("Toxicity prediction: ", TOXIC_CLASS_NAMES[prediction_toxic_sample_input])
        print("With confidence: ", toxicity_confidence)
        print("Naive sentiment prediction: ", prediction_class)
        print("With confidence: ", confidence)
        if TOXIC_CLASS_NAMES[prediction_toxic_sample_input] == 'Toxic' or prediction_class == 'Negative':
            general_sentiment_sample = 'Negative'
        elif TOXIC_CLASS_NAMES[prediction_toxic_sample_input] == 'Non-Toxic' and prediction_class == "Neutral":
            general_sentiment_sample = 'Neutral'
        elif TOXIC_CLASS_NAMES[prediction_toxic_sample_input] == 'Non-Toxic' and prediction_class == "Positive":
            general_sentiment_sample = 'Positive'
        print("General sentiment: ", general_sentiment_sample)
        print("Models are ready to work!")
        print('----------')
    except Exception as e:
        print(f"ERROR: Could not warm up models.")
        print(e)

app = FastAPI(title="Multilingual sentiment classifier", version="1.0.0")

allowed_origins = ["http://localhost:5500", "http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event('startup')
async def startup_event():
    load_models()
    models_warm_up()

@app.post("/predict/")
async def predict_sentiment(input: InputText):
    print(f"DEBUG: model={model}, model_language={model_language}, model_sentiment={model_sentiment}")
    if model is None or model_language is None or model_sentiment is None or model_toxic is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet or failed to load. Check server logs for more information.")

    input_text = input.source_text

    try:
        predicted_language, predicted_language_code, prediction_translation_needed, predicted_translated_input, prediction_idx, prediction_class, confidence, toxicity_class, toxicity_confidence, general_sentiment = full_prediction_pipeline(input_text)
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed. Error: {e}")

    return JSONResponse(content={
        "predicted_language": predicted_language,
        "predicted_language_code": predicted_language_code,
        "prediction_translation_needed": prediction_translation_needed,
        "predicted_translated_input": predicted_translated_input,
        "prediction_idx": prediction_idx,
        "prediction_class": prediction_class,
        "confidence": confidence,
        "toxicity_class": toxicity_class,
        "toxicity_confidence": toxicity_confidence,
        "general_sentiment": general_sentiment
    })

if __name__ == "__main__":
    #load_models()
    #models_warm_up()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)