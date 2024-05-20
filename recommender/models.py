from django.db import models

class FashionItem(models.Model):
    image_path = models.CharField(max_length=255)
    price = models.FloatField()

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

