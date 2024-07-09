from django.db import models
# from django.contrib.auth.models import AbstractUser

# # Create your models here.
# # Creating company models

# class Company(models.Model):
#   company_id = models.AutoField(primary_key=True)
#   name = models.CharField(max_length=50)
#   location = models.CharField(max_length=50)
#   about = models.TextField()
#   type = models.CharField(max_length=50, choices=(('IT', 'IT'), ('Non IT', 'Non IT'), ('Mobile phone', 'Mobile phone')))
#   added_date = models.DateField(auto_now=True)
#   active = models.BooleanField(default=True)

#   def __str__(self):
#     return self.name + ', ' + self.location

# # Creating employee models
# class Employee(models.Model):
#   name=models.CharField(max_length=100)
#   email=models.CharField(max_length=50)
#   address=models.CharField(max_length=200)
#   phone=models.CharField(max_length=10)
#   about=models.TextField()
#   position=models.CharField(max_length=50,choices=(
#       ('Manager','manager'),
#       ('Software Developer','sd'),
#       ('Project Leader','pl')
#   ))

#   company = models.ForeignKey(Company, on_delete=models.CASCADE)

# class User(AbstractUser):
#   name = models.CharField(max_length=255)
#   email = models.CharField(max_length=255, unique=True)
#   password = models.CharField(max_length=255)
#   username = None

#   USERNAME_FIELD = 'email'
#   REQUIRED_FIELDS = []