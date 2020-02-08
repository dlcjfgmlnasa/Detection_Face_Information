from django.contrib import admin
from . import models
from products.models import Product as product
from products_users.models import Products_User as pu
from django.utils.html import mark_safe

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
        "get_name",
        "image",
        # "get_image",
        "products",
        "whoisuser",
        "data_gender",
        "data_age",
        "created",
    )
    fieldsets = (
        ("Basic Info", {"fields": ("data_gender", "image", "data_age", "products")},),
    )

    list_filter = (
        "products",
        "products__user",
        "data_gender",
        "data_age",
        "created",
    )

    change_list_template = "change_list_graph.html"

    def get_name(self, obj):
        return f"data from {obj.products} at {obj.created}"

    get_name.short_description = "data INfo"

    def get_image(self, obj):
        if obj.image.url != None:
            return mark_safe(f'<img width="50px" src="{obj.image.url} "/>')
        return "No Image"
