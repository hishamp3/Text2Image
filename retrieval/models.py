from django.db import models

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from transformers import AutoTokenizer, AutoModel

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from .creds import api_key, cluster_url

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
wnl = WordNetLemmatizer()

model_name = 'bert-base-cased'


class VectorDB:
    def __init__(self):
        self.client = QdrantClient(cluster_url, api_key=api_key,)
        self.info = None
        self.search_result = None

    def create_collection(self, name, size):
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.EUCLID),
        )

    def update_collection(self, name, vid, vector, payload):
        self.info = self.client.upsert(
            collection_name=name,
            wait=True,
            points=[
                PointStruct(id=vid, vector=vector, payload=payload),
            ],
        )

    def update_bulk(self, name, vids, vectors, payloads):
        self.info = self.client.upsert(
            collection_name=name,
            wait=True,
            points=models.Batch(
                ids=vids,
                vectors=vectors.tolist(),
                payloads=payloads
            )
        )

    def delete_collection(self, name):
        self.client.delete_collection(name)

    def search_vector(self, name, vector, limit):
        self.search_result = self.client.search(
            collection_name=name, query_vector=vector, limit=limit
        )

    def search_query(self, name, vector, query, limit):
        self.search_result = self.client.search(
            collection_name=name,
            query_vector=vector,
            query_filter=Filter(
                must=[FieldCondition(key="city", match=MatchValue(value=query))]
            ),
            limit=limit
        )

    def close_connection(self):
        self.client.close()


class LLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.lemma = []
        self.embeddings = []
        self.outputs = []
        self.user_vector = []

    def lemmatization(self, tokens):
        for token in tokens:
            self.lemma.append("".join([wnl.lemmatize(word, pos="v")
                                       for word in
                                       [token for token in
                                        tokenizer.tokenize(token)
                                        if not token.lower() in stop_words]]))
        return self.lemma

    def query_output(self, user_query):
        lemma_sentence = "".join(
            [wnl.lemmatize(word, pos="v")
             for word in
             [token for token in
              tokenizer.tokenize(user_query)
              if not token.lower() in stop_words]])

        user_embeddings = self.tokenizer(lemma_sentence, padding='max_length', max_length=10, truncation=True,
                                         return_tensors="pt")

        self.user_vector = self.model(user_embeddings['input_ids'].squeeze(1),
                                      attention_mask=user_embeddings['attention_mask'].squeeze(1),
                                      token_type_ids=user_embeddings['token_type_ids'].squeeze(1))[1].tolist()[0]

        return self.user_vector

    def get_embeddings(self, tokens):
        self.embeddings = [self.tokenizer(text, padding='max_length', max_length=10, truncation=True,
                                          return_tensors="pt") for text in tokens]
        return self.embeddings

    def model_outputs(self, embeddings):
        self.outputs = [self.model(embedding['input_ids'].squeeze(1),
                                   attention_mask=embedding['attention_mask'].squeeze(1),
                                   token_type_ids=embedding['token_type_ids'].squeeze(1))[1].tolist()[0]
                        for embedding in embeddings]
        return self.outputs
