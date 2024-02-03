import os
import shutil
import pandas as pd
import base64

parent_dir = "./Corpus"
custom_dir = "./Data"
Pos = "-IM-POS"
Neg = "-IM-NEG"
files = os.listdir(parent_dir)

images = []
texts = []

pos_cols = ['fave1', 'fave2', 'fave3', 'fave4', 'fave5']
neg_cols = ['unfave1', 'unfave2', 'unfave3', 'unfave4', 'unfave5']
sentiments = ["-IM-POS", "-IM-NEG"]

os.mkdir(custom_dir)
counter = 1

# Creating raw csv with text tags and Base64 image strings
for i in range(0, len(files)):
    for sentiment in sentiments:
        try:
            filepath = os.path.join(parent_dir, files[i], files[i] + sentiment + ".csv")
            img_path = os.path.join(parent_dir, files[i], files[i] + sentiment)
            df = pd.read_csv(filepath, sep=';')

            if sentiment == "-IM-POS":
                cols = pos_cols
            else:
                cols = neg_cols

            for col in cols:
                img = df[col][0].split("/")[1]
                source_img = os.path.join(img_path, img)
                new_img = os.path.join(custom_dir, str(counter) + ".png")
                shutil.copy(source_img, new_img)
                texts.append(df[col][1])
                with open(new_img, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    image_file.close()
                images.append(encoded_string)
                counter += 1
        except Exception as e:
            print(e)

data = {"texts": texts, "images": images}
df = pd.DataFrame(data)
csv_path = os.path.join(custom_dir, "raw.csv")
df.to_csv(csv_path)
