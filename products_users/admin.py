from django.contrib import admin
from . import models

# Register your models here.


@admin.register(models.Products_User)
class ProductsUserAdmin(admin.ModelAdmin):
    list_display = ("user_name",)

