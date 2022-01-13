from django.db import models
from django.contrib.auth.models import User
# Create your models here.
# class Room(models.Model):
#     code = models.CharField(max_length=10)
#     pwd = models.CharField(max_length=122)

#     def __str__(self):
#         return self.code


class Photo(models.Model):    
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    image = models.ImageField(null=False, blank=False)
    def __str__(self):
        return self.user