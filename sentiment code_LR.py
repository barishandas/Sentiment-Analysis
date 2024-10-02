!wget https://raw.githubusercontent.com/debajyotimaz/nlp_assignment/main/train_split.csv
!wget https://raw.githubusercontent.com/debajyotimaz/nlp_assignment/main/test_split.csv

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')  # Optional: for more language support

train_df = pd.read_csv('train_split.csv')
test_df = pd.read_csv('test_split.csv')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
      
emotion_lexicon = {
    'joy': ['buoyed','gorgeous','grand','breeze','delicious','home','excited', 'happy', 'giggle', 'awe','joy','celebrate','party','sun','shine','laugh', 'smile', 'satisfied', 'pleased','ecstatic','enjoyment','warm','melted','giggling','impressed','love','dancing','helping','roses','balloons','marriage','magic'],
    'fear': ['trauma','spooky','nightmare','intruder','ghost','odd','blackout','shaking','invisible','trouble', 'caught', 'trapped', 'overwhelmed','freak', 'seep', 'vanish', 'slaughter', 'danger', 'worry','scared','frightened','terrified','panic','nervous','dark','trapped','unnerving','unconscious','shook','petrified','turbulence','intense','storm','creepy','scream','psycho','prayer','unaware','blood','terror','gruesome','pressure','threat','gun'],
    'anger': ['drug','embarrassment','wtf','annoyed','mess','pissed','angry', 'furious', 'rage', 'mad', 'fucking', 'broke','hate','bitch','irritated','yell','snarled','disagreed','worst','spite','bullshit'],
    'sadness': ['illness','guilt','confusion','pain','down','grief','trouble', 'sadness', 'tears', 'weary', 'sorrow', 'sad', 'broken', 'fell', 'mind', 'buried','cry','hurt','ill','suffering','disappointed','pain','heaviness','ache','lonely','heartbreaking','awful'],
    'surprise': ['surreal','loose','suddenly','what','wonder','surprised', 'shocked', 'astonished', 'amazed', 'realized', 'gasp','unexpected', 'wow','mysterious','strange','freaky','weird','hallucinate']
}

def preprocess_text(text):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what's": "what is",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }

    text = text.lower()

    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)

    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
    return ' '.join(lemmatized_tokens)

def lemmatize_emotion_lexicon(lexicon, lemmatizer):
    lemmatized_lexicon = {}
    for emotion, words in lexicon.items():
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_lexicon[emotion] = lemmatized_words
    return lemmatized_lexicon

emotion_lexicon_lemmatized = lemmatize_emotion_lexicon(emotion_lexicon, lemmatizer)

def extract_emotions(text):
    tokens = word_tokenize(preprocess_text(text))
    bigrams = [' '.join(bigram) for bigram in ngrams(tokens, 2)]
    trigrams = [' '.join(trigram) for trigram in ngrams(tokens, 3)]

    emotion_scores = {emotion: 0 for emotion in emotion_lexicon_lemmatized}

    for emotion, keywords in emotion_lexicon_lemmatized.items():
        for word in tokens + bigrams + trigrams:
            if word in keywords:
                emotion_scores[emotion] += 1

    return emotion_scores

def create_feature_matrix(texts):
    features = [extract_emotions(text) for text in texts]
    return pd.DataFrame(features)

X_train = create_feature_matrix(train_df['text'])
y_train = train_df[['Joy', 'Fear', 'Anger', 'Sadness', 'Surprise']]

X_test = create_feature_matrix(test_df['text'])
y_test = test_df[['Joy', 'Fear', 'Anger', 'Sadness', 'Surprise']]

model = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag',max_iter=1000), n_jobs=1))
])

param_grid = {
    'clf__estimator__C': [0.001,0.01, 0.1, 1, 10],
    'clf__estimator__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(model, param_grid, cv=20, scoring='f1_macro', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['joy', 'fear', 'anger', 'sadness', 'surprise'],zero_division=1))
