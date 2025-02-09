from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


@csrf_exempt
def view_test(request):
    return JsonResponse({"msg": "testst"})
    