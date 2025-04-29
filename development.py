import numpy as np
import nltk
import sklearn
import pandas as pd
import torch

#torch.cuda.empty_cache()
#torch.cuda.reset_peak_memory_stats()
df1 = pd.read_csv(filepath_or_buffer='./archive.zip', compression='zip', encoding='utf-8')
df1.head()
import matplotlib.pyplot as plt

language_count = df1['Language'].value_counts()
plt.figure(figsize=(6, 4))
language_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
df2 = pd.read_csv(filepath_or_buffer='sentences.csv', encoding='utf-8')
df2.head()
import json
languages = None
with open('lan_to_language.json') as language_data:
    languages = json.load(language_data)
    print('Type: ', type(languages))
print("Example: cmn")
print(languages['cmn'])

df2['language'] = df2['lan_code'].map(languages)
languages_unique = list()
def get_unique_languages(text):
    text = str(text)
    if text not in languages_unique:
        languages_unique.append(text)
    return languages_unique
df2['language'].map(get_unique_languages)
print("UNIQUE LANGUAGES: ", len(languages_unique))
df2.head()
language_count = df2['language'].value_counts()
languages_to_keep = language_count[language_count >= 10000].index
df2_filtered = df2[df2['language'].isin(languages_to_keep)].copy()

languages_unique = list()
df2_filtered['language'].map(get_unique_languages)
print("UNIQUE LANGUAGES: ", len(languages_unique))
print("LANGUAGES LEFT: ")
print(sorted(languages_unique))
df2_filtered.head()
language_count = df2_filtered['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.tail(25).sort_values().plot(kind='barh', color='mediumseagreen')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
language_count = df2_filtered['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.head(25).sort_values().plot(kind='barh', color='mediumseagreen')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
min_sentences = 5
max_sentences = 6

np.random.seed(42)

paragraphs = []

for lan_code, group in df2_filtered.groupby("lan_code"):

    group = group.sample(frac=1).reset_index(drop=True)

    sentences = group["sentence"].tolist()
    i = 0
    paragraph_id = 0
    while i < len(sentences) - min_sentences:
        n = np.random.randint(min_sentences, max_sentences + 1)
        if i + n > len(sentences):
            break
        paragraph = " ".join(sentences[i:i+n])
        paragraphs.append({
            "paragraph_id": f"{lan_code}_{paragraph_id}",
            "lan_code": lan_code,
            "language": group["language"].iloc[0],
            "paragraph_text": paragraph
        })
        i += n
        paragraph_id += 1

paragraph_df = pd.DataFrame(paragraphs)


paragraph_df.head()
paragraph_df.tail()
language_count = paragraph_df['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.head(25).sort_values().plot(kind='barh', color='tomato')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
language_count = paragraph_df['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.tail(25).sort_values().plot(kind='barh', color='tomato')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
df_result = paragraph_df.groupby('language').apply(
    lambda x: x.sample(min(len(x), 25000), random_state=123)
).reset_index(drop=True)
df_result.tail()
language_count = df_result['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.head(25).sort_values().plot(kind='barh', color='orange')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
language_count = df_result['language'].value_counts()
plt.figure(figsize=(15, 6))
language_count.tail(25).sort_values().plot(kind='barh', color='orange')
plt.title('Language distribution')
plt.xlabel('Language')
plt.ylabel('Number of occurences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
def to_lowercase(text):
    text = str(text)
    return text.lower()
df_result['language'] = df_result['language'].map(to_lowercase)
df_result['paragraph_text'] = df_result['paragraph_text'].map(to_lowercase)

df_result.head()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

print(df_result.isnull().sum())

x = np.array(df_result['paragraph_text'])
y = np.array(df_result['language'])

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("Entries for training: ", x_train.shape)
print("Entries for testing: ", x_test.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
manual_testing_strings = ["Danas je bio veoma lep dan. Sunce je sijalo tokom celog jutra, a uveče smo otišli u šetnju kraj reke. Deca su se igrala u parku dok su roditelji razgovarali. U prodavnici sam kupio sve što nam je trebalo za večeru. Sutra planiramo da idemo kod bake i deke na ručak.",
                          "Сегодня был очень хороший день. Солнце светило ярко, и мы решили прогуляться в парке. Ветер был лёгкий, и воздух казался особенно свежим. Мы зашли в кафе и выпили по чашке кофе. Завтра я собираюсь навестить своих родителей",
                          "Dnes bolo krásne počasie, a tak sme sa rozhodli ísť na výlet. Navštívili sme malé mestečko neďaleko nášho domova. Deti sa tešili na zmrzlinu a jazdu na bicykli. Cestou späť sme sa zastavili v kníhkupectve. Večer sme strávili pri spoločnej večeri.",
                          "Danes je bil čudovit dan. Odpravili smo se na sprehod po gozdu in opazovali ptice. Vzeli smo s seboj malico in uživali v naravi. Otroci so tekali po travniku in se smejali. Na poti domov smo kupili sveže sadje na tržnici.",
                          "Dnes bylo opravdu hezky, a tak jsme šli ven na procházku. This is an English sentence. Navštívili jsme místní park, kde kvetly první jarní květiny. Děti si hrály na hřišti, zatímco jsme si povídali na lavičce. Po cestě domů jsme se zastavili v pekárně. Večer jsme si pustili film a odpočívali."]
manual_testing_labels = ['Serbian', 'Russian', 'Slovak', 'Slovenian', 'Czech']
manual_testing_predictions = list()

for text in manual_testing_strings:
    output = model.predict(count_vectorizer.transform([text]).toarray())
    manual_testing_predictions.append(output)

print(manual_testing_predictions)
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.get_autocast_gpu_dtype())
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class LanguageClassifierCNN(nn.Module):
    def __init__(self, vocabulary_dim, num_languages):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_dim, 100)
        self.conv1 = nn.Conv1d(100, 256, 7)
        self.pool1 = nn.AdaptiveAvgPool1d(output_size=128)
        self.conv2 = nn.Conv1d(256, 128, 5)
        self.pool2 = nn.AdaptiveAvgPool1d(output_size=64)
        self.conv3 = nn.Conv1d(128, 64, 3)
        self.pool3 = nn.AdaptiveAvgPool1d(output_size=32)

        self.fc1 = nn.Linear(in_features=(32 * 64), out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_languages)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LanguageClassifierCNN(len(count_vectorizer.vocabulary_), 48).to(device)
tokens = list()
tokens.append("UNK")
tokens.append("PAD")
for item in list(count_vectorizer.vocabulary_):
    tokens.append(item)
token_to_id = {token: id for id, token in enumerate(tokens)}
print("VOCABULARY LENGTH: ", len(list(token_to_id)))
print()
print("SAMPLE ID OF PAD: ", token_to_id['PAD'])
UNK_IX, PAD_IX = map(token_to_id.get, ["UNK", "PAD"])

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
        
    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix

def to_tensors(batch, device):
    batch_tensors = dict()
    for key, arr in batch.items():
        if key == 'paragraph_text' or key == 'language':
            batch_tensors[key] = torch.tensor(arr, device=device, dtype=torch.int64)
        else:
            batch_tensors[key] = torch.tensor(arr, device=device)
    return batch_tensors

def apply_word_dropout(matrix, keep_prop, replace_with=UNK_IX, pad_ix=PAD_IX,):
    dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop]).astype(np.float32)
    dropout_mask &= matrix != pad_ix
    return np.choose(dropout_mask, [matrix, np.full_like(matrix, replace_with)])

def make_batch(data, max_len=None, word_dropout=0, device=device):
    batch = {}
    batch['paragraph_text'] = as_matrix(data['paragraph_text'].values, max_len)

    if word_dropout != 0:
        batch['paragraph_text'] = apply_word_dropout(batch['paragraph_text'], 1. - word_dropout)
    
    if 'language_id' in data.columns:
        batch['language'] = data['language_id'].values
    return to_tensors(batch, device)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_result['language_id'] = le.fit_transform(df_result['language'])

data_train, data_test = train_test_split(df_result, test_size=0.2)
make_batch(data_train[:3], max_len=10)

batch = make_batch(data_train[:100], device=device)
criterion = nn.CrossEntropyLoss()
print(list(batch))

prediction = model(batch['paragraph_text'])
print(prediction.shape)
print(batch['language'])
loss = criterion(prediction, batch['language'])

def iterate_minibatches(data, batch_size=256, shuffle=True, cycle=False, device=device, **kwargs):
    while True:
        indices = np.arange(len(data))
        if shuffle:
            indices = np.random.permutation(indices).astype(np.float32)
            

        for start in range(0, len(indices), batch_size):
            batch = make_batch(data.iloc[indices[start : start + batch_size]], device=device, **kwargs)
            yield batch
        
        if not cycle: break

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
batch_size = 128

total_accuracy = list()
total_loss = list()
scaler = GradScaler()

for epoch in range(epochs):
    total_correct = 0
    total_examples = 0
    total_loss_epoch = 0
    print(f"epoch: ", {epoch})
    model.train()
    for i, batch in tqdm(enumerate(iterate_minibatches(data_train, batch_size, device=device)), total=len(data_train) // batch_size):
        optimizer.zero_grad()
        targets = batch['language']
        #with autocast(device_type=device):
        try:
            logits = model(batch['paragraph_text'])
            if targets.max() >= 48 or targets.min() < 0:
                print(f"Invalid target detected! Min: {targets.min()}, Max: {targets.max()}")
                print("All targets:", targets.cpu().tolist())
                raise ValueError("Bad target index in batch!")
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
        except RuntimeError as e:
            print(f"Failed at step {i} in epoch {epoch}")
            print(f"Input shape: {batch['paragraph_text'].shape}")
            print(f"Target shape: {targets.shape}")
            print(f"Loss: {loss}")
            print(f"Is loss NaN?: {torch.isnan(loss)}")
            print(f"Error: {e}")
            raise e

        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        total_loss_epoch += loss.item()
        total_correct = total_correct + correct
        total_examples += targets.size(0)
    
    epoch_loss = total_loss_epoch / total_examples
    epoch_accuracy = total_correct / total_examples
    total_loss.append(total_loss_epoch)
    total_accuracy.append(epoch_accuracy)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")