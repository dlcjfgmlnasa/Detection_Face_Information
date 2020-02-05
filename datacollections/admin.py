from django.contrib import admin
from . import models
from products.models import Product as product
from products_users.models import Products_User as pu


# Register your models here.
@admin.register(models.Datacollection)
class DatacollectionAdmin(admin.ModelAdmin):
    """Datacolelction Admin Definition"""

    def whoisuser(self, obj):
        all_product = product.objects.get(name=obj.products)
        username = pu.objects.get(user_name=all_product.user)
        return username

    whoisuser.short_description = "user"

    list_display = (
        "products",
        "whoisuser",
        "data_gender",
        "data_age",
        "created",
    )
    fieldsets = (("Basic Info", {"fields": ("data_gender", "data_age", "products")},),)

    list_filter = (
        "products",
        "products__user",
        "data_gender",
        "data_age",
        "created",
    )

    change_list_template = "change_list_graph.html"

