from django.shortcuts import render
from . import models


def getImage(image_query):
    q1 = models.VectorDB()
    llm = models.LLM()

    query_vector = llm.query_output(image_query.lower())
    q1.search_vector("text2image", query_vector, 1)
    for result in q1.search_result:
        encoded_string = result.payload['url']
        return encoded_string


def retrieval(request):
    if request.method == 'POST':
        image_query = request.POST['image_query']
        image = getImage(image_query)
        return render(request, 'index.html',
                      {'image_query': image_query, 'image': image})
    return render(request, 'index.html')
