from django.contrib import admin
from . import models

# Register your models here.
@admin.register(models.Product)
class ProductsAdmin(admin.ModelAdmin):
    list_display = ("name", "serial_number", "user", "created")

    list_filter = ("user", "created")

    def __str__(self):
        return self.user
