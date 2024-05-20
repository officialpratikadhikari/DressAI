import os
import json
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import FashionItem, UploadedImage
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Construct absolute paths to data files
embeddings_path = os.path.join(settings.BASE_DIR, 'recommender/data/embeddings.pkl')
filenames_path = os.path.join(settings.BASE_DIR, 'recommender/data/filenames.pkl')

# Load the feature list and filenames
feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))
filenames = pickle.load(open(filenames_path, 'rb'))

# Define the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def home(request):
    items = FashionItem.objects.all()
    return render(request, 'home.html', {'items': items})

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(name)

        # Feature extraction
        features = feature_extraction(fs.path(name), model)
        indices = recommend(features, feature_list)

        # Construct the correct URLs for the recommended images and load their JSON data
        recommended_items = []
        for index in indices[0]:
            image_path = filenames[index]
            image_url = settings.DATA_URL + image_path

            # Extract the filename without the directory part
            filename = os.path.basename(image_path).replace('.jpg', '.json')
            json_file_path = os.path.join(settings.DATA_ROOT, 'styles', filename)

            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                recommended_items.append({
                    'image': image_url,
                    'image_name': os.path.basename(image_path),  # Extract only the filename
                    'data': json_data['data']
                })
            else:
                print(f"JSON file does not exist: {json_file_path}")
                recommended_items.append({
                    'image': image_url,
                    'image_name': os.path.basename(image_path),  # Extract only the filename
                    'data': {}
                })

        return render(request, 'recommendations.html', {
            'uploaded_file_url': uploaded_file_url,
            'recommendations': recommended_items
        })
    return render(request, 'upload_image.html')

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# New functions for product details and cart
def get_product_details(image_name):
    json_path = os.path.join(settings.DATA_ROOT, 'styles', f"{image_name}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            product_details = json.load(json_file)
            return product_details
    return None

def product_details(request, image_name):
    product = get_product_details(image_name.replace('.jpg', ''))
    if product:
        return render(request, 'product_details.html', {'product': product})
    return redirect('recommendations')

def product_details(request, image_name):
    json_file_path = os.path.join(settings.DATA_ROOT, 'styles', image_name.replace('.jpg', '.json'))
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            product = json.load(f)
        product['image'] = f"{settings.DATA_URL}/images/{image_name}"
        product['image_name'] = image_name
        return render(request, 'product_details.html', {'product': product})
    return redirect('recommendations')

