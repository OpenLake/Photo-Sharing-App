from django.contrib import admin
from .models import Photo, Person, PersonGallery

# Register your models here.
admin.site.register(Photo)
admin.site.register(Person)
admin.site.register(PersonGallery)
