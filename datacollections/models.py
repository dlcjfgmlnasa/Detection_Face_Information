from django.db import models
from products import models as products_models
from products.models import Product as product
from products_users.models import Products_User as pu

# Create your models here.
class Datacollection(models.Model):
    """How Data looks like"""

    GENDER_MALE = "male"
    GENDER_FEMALE = "female"
    GENDER_OTHER = "other"

    GENDER_CHOICES = (
        (GENDER_MALE, "Male"),
        (GENDER_FEMALE, "Female"),
        (GENDER_OTHER, "Other"),
    )

    image = models.ImageField(upload_to="data_photos", blank=True)
    data_gender = models.CharField(choices=GENDER_CHOICES, max_length=10, blank=True)
    data_age = models.PositiveIntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    products = models.ForeignKey(
        "products.Product",
        related_name="datacollections",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

