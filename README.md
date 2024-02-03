## Text2Image Search

![architecture](https://github.com/hishamp3/Text2Image/assets/101292796/6c795053-6739-42df-9b87-3710e32fc08d)

### Dataset ADS-16 : https://www.kaggle.com/datasets/groffo/ads16-dataset/data

### Text Preprocessing:
![preprocess](https://github.com/hishamp3/Text2Image/assets/101292796/c8ef7343-a32e-40d1-995e-6b212b2c9551)

### Image Processing
All images that have been encoded as a Base64 string.

### Qdrant cloud
a collection named "text2image" is created using Bert pooled output as vector and Base64 image string as payload.
cat = (id,bert-output vector,payload={url:Base64 string})

### Challenges encountered during implementation
1. Deciding on the max-length for truncation for tokenization
2. Write-out timeout issue when inserting data into qdrant cloud collection.

### Potential improvements
1. Checking for a better dataset with the structure of tag,description and image for better search.
   ("cat", "cats are playing in the garden", Image) the current datasets consists of pairs of tags and images.
2. For a longer sequence, BLEU or ROUGE score are better evaluation metrics in comparision to euclidean distance. The current systems struggles with the large sequence input.

## How to run
1. replace your Qdrant API key and cluster url in file retrieval -> creds.py.
2. Run the Django application using "python manage.py runserver"
                  or
1. Run "docker compose up --build" 
