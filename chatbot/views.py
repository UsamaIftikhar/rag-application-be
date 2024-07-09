from django.shortcuts import HttpResponse, render

def homepage(request):
  data = {'name': 'usama iftikhar'}
  return render(request, 'homepage.html', data)

def aboutuspage(request):
  return HttpResponse('This is about us page')
