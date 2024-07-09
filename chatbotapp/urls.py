from django.contrib import admin
from django.urls import path, include
from chatbotapp.views import RagViewSet
from rest_framework import routers
# from .views import LogoutView

router=routers.DefaultRouter()
# router.register(r'companies', CompanyViewSet)
# router.register(r'employees', EmployeeViewSet)
# router.register(r'rag', RagViewSet, basename='rag')

urlpatterns = [
    # path('', views.appview),
    path('', include(router.urls)),
    # path('register', RegisterView.as_view()),
    # path('login', LoginView),
    # path('user', UserView),
    # path('logout', LogoutView),
    path('rag/', RagViewSet),
]
