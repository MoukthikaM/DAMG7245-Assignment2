from fastapi.testclient import TestClient
from predictfromimage import app
import predictfromimage as app1

client = TestClient(app)
print(client)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_predict():
     response = client.post(
        "/flashes",
        json={"vis": "/Users/moukthikamanapati/Desktop/BigData7245/WAF_ML_Tutorial_Part1/vis.png",
    "vil": "/Users/moukthikamanapati/Desktop/BigData7245/WAF_ML_Tutorial_Part1/vil.png",
    "ir069": "/Users/moukthikamanapati/Desktop/BigData7245/WAF_ML_Tutorial_Part1/ir069.png",
    "ir107": "/Users/moukthikamanapati/Desktop/BigData7245/WAF_ML_Tutorial_Part1/ir107.png" 
    }  )
     assert response.status_code == 200
     assert response.json() == {
       
        "flashes": 189.26632288706858

    }




def test_features():
    xtest = [[0.00000000e+00,5.24317755e-04,3.37497127e-03, 5.06716817e-03,
  6.67591488e-03, 5.25881514e-02, 1.37311457e-01, 2.06406842e-01,
  8.70725285e-01, 6.85337480e-02, 6.90518686e-02 ,7.01049914e-02,
  7.18612909e-02, 7.24356441e-02, 7.29280359e-02, 7.32776926e-02,
  7.37663293e-02, 7.40805239e-02, 3.09757816e-03 ,5.45132372e-03,
  7.74448031e-03, 1.03943459e-02, 1.65144056e-02 ,3.27861505e-02,
  6.83047268e-02, 1.02772625e-01, 1.87669596e-01 ,0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ,3.22382967e-03,
  8.34234982e-03, 4.74654987e-02 ,2.58551750e-01 ,8.05709460e-01]]


    assert app1.predictflash(xtest) == 189.2663229386946


def test_imageprocessing():
     vis="/Users/moukthikamanapati/Desktop/BigData7245/WAF_ML_Tutorial_Part1/vis.png" 
     assert app1.imagepreprocessing(vis).shape == (1,9)
