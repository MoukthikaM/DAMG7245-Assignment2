from fastapi.testclient import TestClient
from dataasaservice import app
client = TestClient(app)


def test_desc_column():
    response = client.get("/desc/q100")
    assert response.status_code == 200
    assert response.json() == {
    "count": 618926.0,
    "mean": 4294.050482610199,
    "std": 4425.823644839205,
    "min": 8.0,
    "25%": 31.0,
    "50%": 3201.0,
    "75%": 8479.0,
    "max": 12521.0
}

def test_eventcount():
    response = client.get("/eventcount/Flood")
    assert response.status_code == 200
    assert response.json() == { "Event count": 6370
}


def test_getfiles():
    response = client.get("/getfiles/835047")
    assert response.status_code == 200
    assert response.json() == {
    "19299": "vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",
    "22412": "ir107/2019/SEVIR_IR107_STORMEVENTS_2019_0101_0630.h5",
    "29236": "ir069/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5",
    "41182": "vis/2019/SEVIR_VIS_STORMEVENTS_2019_0601_0630.h5",
    "72402": "lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0601_0701.h5"
}

def test_getlocation():
    response = client.get("/location/Tornado/32.67/-94.33")
    assert response.status_code == 200
    assert response.json() == [
    {
        "event_id": 728503.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.7487482467,
        "llcrnrlon": -96.7538711491,
        "urcrnrlat": 35.0799950019,
        "urcrnrlon": -92.4806816924
    },
    {
        "event_id": 743559.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.0659651646,
        "llcrnrlon": -94.3518010573,
        "urcrnrlat": 34.3051359868,
        "urcrnrlon": -90.0216353134
    },
    {
        "event_id": 728503.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.7487482467,
        "llcrnrlon": -96.7538711491,
        "urcrnrlat": 35.0799950019,
        "urcrnrlon": -92.4806816924
    },
    {
        "event_id": 785428.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.2735247586,
        "llcrnrlon": -94.9863289403,
        "urcrnrlat": 33.5406846332,
        "urcrnrlon": -90.7232735953
    },
    {
        "event_id": 728503.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.7487482467,
        "llcrnrlon": -96.7538711491,
        "urcrnrlat": 35.0799950019,
        "urcrnrlon": -92.4806816924
    },
    {
        "event_id": 743559.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.0659651646,
        "llcrnrlon": -94.3518010573,
        "urcrnrlat": 34.3051359868,
        "urcrnrlon": -90.0216353134
    },
    {
        "event_id": 785428.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.2735247586,
        "llcrnrlon": -94.9863289403,
        "urcrnrlat": 33.5406846332,
        "urcrnrlon": -90.7232735953
    },
    {
        "event_id": 823561.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.1607106217,
        "llcrnrlon": -97.7062560696,
        "urcrnrlat": 34.5301718876,
        "urcrnrlon": -93.5021429261
    },
    {
        "event_id": 819402.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.5810956478,
        "llcrnrlon": -95.8106374765,
        "urcrnrlat": 33.8793957658,
        "urcrnrlon": -91.5628063083
    },
    {
        "event_id": 815581.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.6894933703,
        "llcrnrlon": -95.0353530133,
        "urcrnrlat": 33.95708097,
        "urcrnrlon": -90.751998274
    },
    {
        "event_id": 819402.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.5810956478,
        "llcrnrlon": -95.8106374765,
        "urcrnrlat": 33.8793957658,
        "urcrnrlon": -91.5628063083
    },
    {
        "event_id": 815581.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.6894933703,
        "llcrnrlon": -95.0353530133,
        "urcrnrlat": 33.95708097,
        "urcrnrlon": -90.751998274
    },
    {
        "event_id": 819402.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.5810956478,
        "llcrnrlon": -95.8106374765,
        "urcrnrlat": 33.8793957658,
        "urcrnrlon": -91.5628063083
    },
    {
        "event_id": 815581.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.6894933703,
        "llcrnrlon": -95.0353530133,
        "urcrnrlat": 33.95708097,
        "urcrnrlon": -90.751998274
    },
    {
        "event_id": 819402.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.5810956478,
        "llcrnrlon": -95.8106374765,
        "urcrnrlat": 33.8793957658,
        "urcrnrlon": -91.5628063083
    },
    {
        "event_id": 815581.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.6894933703,
        "llcrnrlon": -95.0353530133,
        "urcrnrlat": 33.95708097,
        "urcrnrlon": -90.751998274
    },
    {
        "event_id": 743559.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.0659651646,
        "llcrnrlon": -94.3518010573,
        "urcrnrlat": 34.3051359868,
        "urcrnrlon": -90.0216353134
    },
    {
        "event_id": 785428.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.2735247586,
        "llcrnrlon": -94.9863289403,
        "urcrnrlat": 33.5406846332,
        "urcrnrlon": -90.7232735953
    },
    {
        "event_id": 728503.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.7487482467,
        "llcrnrlon": -96.7538711491,
        "urcrnrlat": 35.0799950019,
        "urcrnrlon": -92.4806816924
    },
    {
        "event_id": 743559.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.0659651646,
        "llcrnrlon": -94.3518010573,
        "urcrnrlat": 34.3051359868,
        "urcrnrlon": -90.0216353134
    },
    {
        "event_id": 743559.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.0659651646,
        "llcrnrlon": -94.3518010573,
        "urcrnrlat": 34.3051359868,
        "urcrnrlon": -90.0216353134
    },
    {
        "event_id": 785428.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.2735247586,
        "llcrnrlon": -94.9863289403,
        "urcrnrlat": 33.5406846332,
        "urcrnrlon": -90.7232735953
    },
    {
        "event_id": 815581.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.6894933703,
        "llcrnrlon": -95.0353530133,
        "urcrnrlat": 33.95708097,
        "urcrnrlon": -90.751998274
    },
    {
        "event_id": 819402.0,
        "event_type": "Tornado",
        "llcrnrlat": 30.5810956478,
        "llcrnrlon": -95.8106374765,
        "urcrnrlat": 33.8793957658,
        "urcrnrlon": -91.5628063083
    },
    {
        "event_id": 823561.0,
        "event_type": "Tornado",
        "llcrnrlat": 31.1607106217,
        "llcrnrlon": -97.7062560696,
        "urcrnrlat": 34.5301718876,
        "urcrnrlon": -93.5021429261
    }
]

def test_count_of_storms():
    response = client.get("count/2018-03-01/2018-04-01")
    assert response.status_code == 200
    assert response.json() == {
    "Flash Flood": 1565,
    "Flood": 880,
    "Funnel Cloud": 426,
    "Hail": 4403,
    "Heavy Rain": 439,
    "Lightning": 383,
    "Thunderstorm Wind": 7965,
    "Tornado": 846
}
