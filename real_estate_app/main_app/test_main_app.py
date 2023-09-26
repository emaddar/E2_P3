import pytest
from django.urls import reverse
from .utils import make_prediction


@pytest.mark.django_db
def test_good_prediction():
    good_input = {
        'TotalSF' : 80,
        'Overall Qual': 7,
        'Neighborhood': 'Gilbert',
        'Bsmt Qual': 'Gd', 
        'Exter Qual': 'Po', 
        'Kitchen Qual' : 'Gd',
        'Garage Cars' : 2,
        'TotalBathrooms' : 2,
        'Age' : 23,
        'Garage Finish' : 'Fin'
    }

    prediction = make_prediction(good_input)
    assert prediction == 136543


@pytest.mark.django_db
def test_wrong_prediction():
    wrong_input = {
        'TotalSF' : 80,
        'Overall Qual': 7,
        'Neighborhood': 'Gilbert',
        'Bsmt Qual': 'Gd', 
        'Exter Qual': 'Po', 
        'Kitchen Qual' : 'Gd',
        'Garage Cars' : 2,
        'TotalBathrooms' : 'OK',
        'Age' : 23,
        'Garage Finish' : 'Fin'
    }

    prediction = make_prediction(wrong_input)
    assert prediction == 0


@pytest.mark.django_db
def test_index_view(client):
    url = reverse('index')
    response = client.get(url)
    assert response.status_code == 200


@pytest.mark.django_db
def test_predict_view_post_method_good_input(client):
    url = reverse('predict')
    data = {
        'TotalSF' : 80,
        'Overall Qual': 7,
        'Neighborhood': 'Gilbert',
        'Bsmt Qual': 'Gd', 
        'Exter Qual': 'Po', 
        'Kitchen Qual' : 'Gd',
        'Garage Cars' : 2,
        'TotalBathrooms' : 2,
        'Age' : 23,
        'Garage Finish' : 'Fin'
    }
    response = client.post(url, data)
    assert response.status_code == 200
    assert int(response.context['data']) == 136543


# @pytest.mark.django_db
# def test_predict_view_post_method_wrong_input(client):
#     url = reverse('predict')
#     data = {
#         'TotalSF' : 80,
#         'Overall Qual': 7,
#         'Neighborhood': 'Gilbert',
#         'Bsmt Qual': 'Gd', 
#         'Exter Qual': 'Po', 
#         'Kitchen Qual' : 'Gd',
#         'Garage Cars' : 2,
#         'TotalBathrooms' : 2,
#         'Age' : 23,
#         'Garage Finish' : 'Fin'
#     }
#     response = client.post(url, data)
#     assert response.status_code == 200
#     assert response.content == b"The Input is not Correct"


@pytest.mark.django_db
def test_predict_view_get_method(client):
    url = reverse('predict')
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == b"Method Not Allowed"


##################################### my tests





@pytest.mark.django_db
def test_predict_view_post_method_non_negative_input(client):
    url = reverse('predict')  # Replace 'predict' with the actual URL name
    data = {
        'TotalSF': 150,  # A reasonable value for total square footage
        'Overall Qual': 8,  # A valid value for overall quality
        'Neighborhood': 'StoneBr',  # A valid neighborhood name
        'Bsmt Qual': 'Gd',
        'Exter Qual': 'Gd',
        'Kitchen Qual': 'Gd',
        'Garage Cars': -1,  # A non-negative value, but still invalid
        'TotalBathrooms': 2,  # A valid value for total bathrooms
        'Age': 10,  # A reasonable value for age
        'Garage Finish': 'Fin'
    }
    response = client.post(url, data)
    
    assert response.status_code == 200
    assert b'Invalid value for Garage Cars' in response.content  # Check for an error message




@pytest.mark.django_db
def test_predict_view_post_method_non_zero_total_sf(client):
    url = reverse('predict')  # Replace 'predict' with the actual URL name
    data = {
        'TotalSF': 0,  # A non-zero value for total square footage
        'Overall Qual': 8,  # A valid value for overall quality
        'Neighborhood': 'StoneBr',  # A valid neighborhood name
        'Bsmt Qual': 'Gd',
        'Exter Qual': 'Gd',
        'Kitchen Qual': 'Gd',
        'Garage Cars': 2,
        'TotalBathrooms': 2,  # A valid value for total bathrooms
        'Age': 10,  # A reasonable value for age
        'Garage Finish': 'Fin'
    }
    response = client.post(url, data)
    
    assert response.status_code == 200
    assert b'TotalSF must be greater than 0' in response.content  # Check for an error message
