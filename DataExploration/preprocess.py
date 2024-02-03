import pandas as pd
from retrieval.models import VectorDB
import time
from retrieval.models import LLM


# Generate vector csv
def preprocess_corpus() -> pd.DataFrame:
    # Eliminate duplicates and strings of length 3
    df = pd.read_csv("./raw.csv", usecols=['texts', 'images'])
    df = df.drop_duplicates(subset='texts', keep="first").reset_index(drop=True)
    df['length'] = df['texts'].str.len()

    print(df.describe())
    df = df[df['length'] >= 3].reset_index(drop=True)

    # Stop word, Lemmatization, Bert-embedding
    llm = LLM()
    df['lemma'] = llm.lemmatization(df['texts'].tolist())
    embeddings = llm.get_embeddings(df['lemma'].tolist())
    df['embeddings'] = llm.model_outputs(embeddings)

    df.to_csv("./vector.csv")
    return df


# create collection
def create_vector_db():
    df = pd.read_csv("./vector.csv", converters={'images': pd.eval, 'embeddings': pd.eval})

    ids = [i for i in range(1, len(df) + 1)]
    payloads = [{"url": imgstr} for imgstr in df['images'].tolist()]
    vectors = df['embeddings'].tolist()

    counter = 8
    min_len = 1

    # Adding entries to cloud collection in batch of 8
    while counter <= len(df):
        q1 = VectorDB()
        max_len = counter

        for i in range(min_len, max_len):
            q1.update_collection("text2image", ids[i], vectors[i], payloads[i])

        print(q1.info)
        print(counter)
        counter += 8
        min_len = max_len
        q1.close_connection()

        time.sleep(10)
