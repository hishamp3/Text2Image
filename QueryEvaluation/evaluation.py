from qdrant_client import QdrantClient
from retrieval.creds import cluster_url, api_key
import base64
import io

import matplotlib.pyplot as plt
from PIL import Image

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from transformers import AutoTokenizer, AutoModel

model_name = 'bert-base-cased'
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
wnl = WordNetLemmatizer()

Positives = ["cat", "bike", "baby", "boxing", "football", "gun violence", "a group of cars", "violence"]
Negatives = ["guitar", "black cat", "a poster about gun violence"]

# test query
user_query = "a poster about gun violence"

if __name__ == "__main__":
    qdrant_client = QdrantClient(
        cluster_url,
        api_key=api_key,
    )

    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    user_query = user_query.lower()

    lemma_sentence = "".join(
        [wnl.lemmatize(word, pos="v")
         for word in
         [token for token in
          tokenizer.tokenize(user_query)
          if not token.lower() in stop_words]])

    user_embeddings = bert_tokenizer(lemma_sentence, padding='max_length', max_length=10, truncation=True,
                                     return_tensors="pt")

    user_vector = bert_model(user_embeddings['input_ids'].squeeze(1),
                             attention_mask=user_embeddings['attention_mask'].squeeze(1),
                             token_type_ids=user_embeddings['token_type_ids'].squeeze(1))[1].tolist()[0]

    search_result = qdrant_client.search(
        collection_name="text2image", query_vector=user_vector, limit=1
    )

    for result in search_result:
        encoded_string = result.payload['url']
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(encoded_string, "utf-8"))))
        imgplot = plt.imshow(img)
        plt.show()
