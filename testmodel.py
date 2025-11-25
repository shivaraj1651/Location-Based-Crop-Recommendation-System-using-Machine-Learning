import requests
from flask import Flask,request,render_template
import numpy as np
import sklearn as sk
import pickle
import requests, json

app=Flask(__name__)

@app.route('/')
def index():

    try:

        lang = request.args.get('lang')
        lat = request.args.get('lat')
        p1 = {"lat": lat, "lon": lang}

        rest_url = "https://rest.isric.org"
        prop_query_url = f"{rest_url}/soilgrids/v2.0/properties/query"

        props = {"property": "silt", "depth": "0-5cm", "value": "mean"}
        res1 = requests.get(prop_query_url, params={**p1, **props})

        res = res1.json()['properties']["layers"][0]["depths"][0]["values"]
        silt = res["mean"] / 10

        props = {"property": "sand", "depth": "0-5cm", "value": "mean"}
        res1 = requests.get(prop_query_url, params={**p1, **props})
        res = res1.json()['properties']["layers"][0]["depths"][0]["values"]
        sand = res["mean"] / 10

        props = {"property": "phh2o", "depth": "0-5cm", "value": "mean"}
        res1 = requests.get(prop_query_url, params={**p1, **props})
        res = res1.json()['properties']["layers"][0]["depths"][0]["values"]
        ph = res["mean"] / 10

        props = {"property": "clay", "depth": "0-5cm", "value": "mean"}
        res1 = requests.get(prop_query_url, params={**p1, **props})
        res = res1.json()['properties']["layers"][0]["depths"][0]["values"]
        clay = res["mean"] / 10

        props = {"property": "nitrogen", "depth": "0-5cm", "value": "mean"}
        res1 = requests.get(prop_query_url, params={**p1, **props})
        res = res1.json()['properties']["layers"][0]["depths"][0]["values"]
        nitrogen = res["mean"] / 10

        print(ph)
        complete_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lang}&appid=407c59531dcab5eeaacf5f454311d4e6"
        response = requests.get(complete_url)
        x = response.json()
        print(x)

        cityname=x['name']
        print(cityname)

        l = x['main']
        #print(type(l))
        #print(l)
        print("temp = ",l["temp"] - (273.15))
        print("humidity = ",l["humidity"])

        temp=l["temp"] - (273.15)
        hum=l["humidity"]



        model=open("knnmodel.pkl",'rb')
        objectfile=pickle.load(model)
        testdata=np.array([[temp,hum,ph]])

        print("test data = ",testdata)
        prediction=objectfile.predict(testdata)
        print(prediction)

        cropimage=f"{prediction[0]}.jpg"

        print(cropimage)

        data={"place":cityname,"crop":prediction[0],"temp":temp,"hum":hum,"ph":ph,"lat":lat,"lang":lang,"silt":silt,"sand":sand,"clay":clay,"nitrogen":nitrogen}
        return render_template('result.html',result=data,ci=cropimage)
    except Exception as e:
        print(e)
        return render_template('noresult.html')

if __name__=='__main__':
      app.run(debug=True,host="0.0.0.0")