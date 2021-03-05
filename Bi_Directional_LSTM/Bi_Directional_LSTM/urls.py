"""Bi_Directional_LSTM URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Bi_Directional_LSTM.views import index,index2,index3

urlpatterns = [
    path('admin/', admin.site.urls),
    path('project/', index,name="template3"),
    path('abstract/',index2,name="template1"),
    path('teammembers/',index3,name="template2"),
]
