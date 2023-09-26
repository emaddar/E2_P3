from django.shortcuts import render
from django.http import HttpResponse
from .utils import make_prediction


def index(request):
    return render(request, 'index.html')


# def predict(request):
#     if request.method == 'POST':
#         X_predict = {}
#         for var in [
#                 'TotalSF', 'Overall Qual', 'Neighborhood', 'Bsmt Qual', 'Exter Qual',
#                 'Kitchen Qual', 'Garage Cars', 'TotalBathrooms', 'Age', 'Garage Finish']:
#             if var in ["Neighborhood", "Garage Finish", "Bsmt Qual",  "Exter Qual", "Kitchen Qual"]:
#                 X_predict[var] = request.POST.get(var)
#             else:
#                 X_predict[var] = int(request.POST.get(var))
#         pred = make_prediction(X_predict)

#         if pred != 0:
#             return render(request, 'index.html', {'data': int(pred)})
#         else:
#             return HttpResponse("The Input is not Correct")
#     else:
#         return HttpResponse("Method Not Allowed")


############################## my functions 

def predict(request):
    if request.method == 'POST':
        X_predict = {}
        for var in [
                'TotalSF', 'Overall Qual', 'Neighborhood', 'Bsmt Qual', 'Exter Qual',
                'Kitchen Qual', 'Garage Cars', 'TotalBathrooms', 'Age', 'Garage Finish']:
            if var in ["Neighborhood", "Garage Finish", "Bsmt Qual",  "Exter Qual", "Kitchen Qual"]:
                X_predict[var] = request.POST.get(var)
            else:
                value = int(request.POST.get(var))
                if value >= 0:
                    X_predict[var] = value
                else:
                    return HttpResponse(f"Invalid value for {var}")
        
        total_sf = int(request.POST.get('TotalSF'))
        if total_sf > 0:
            X_predict['TotalSF'] = total_sf
        else:
            return HttpResponse("TotalSF must be greater than 0")

        pred = make_prediction(X_predict)

        if pred != 0:
            return render(request, 'index.html', {'data': int(pred)})
        else:
            return HttpResponse("The Input is not Correct")
    else:
        return HttpResponse("Method Not Allowed")

