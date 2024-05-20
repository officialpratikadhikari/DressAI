from django.contrib import admin
from django.urls import path, include
from recommender import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', views.home, name='home'),
                  path('upload_image/', views.upload_image, name='upload_image'),
                  path('about/', views.about, name='about'),
                  path('contact/', views.contact, name='contact'),
                  path('product/<str:image_name>/', views.product_details, name='product_details'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
