from django.db import models
from products_users import models as user_models

# Create your models here.
class Product(models.Model):
    """Rasberry pi model"""

    name = models.CharField(max_length=50)
    serial_number = models.CharField(max_length=50)
    user = models.ForeignKey(
        user_models.Products_User,
        related_name="products",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
