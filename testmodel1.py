import requests
from flask import Flask,request,render_template
import numpy as np
import sklearn as sk
import pickle
import requests, json
import joblib

app=Flask(__name__)

@app.route('/getlocation/',methods=['POST','GET'])
def index():

    lang=request.args.get("long")
    lat= request.args.get("lat")

    return render_template('getlanglat.html',lt=lat,lg=lang)



@app.route('/crop',methods=['GET','POST'])
def croprec():

    if request.method=="POST":


        testdata=[float(x) for x in request.form.values()]

        testdata=[testdata]

        model=joblib.load("myKnnmodel.pkl")

        result=model.predict(testdata)

        cropname=result[0]
        return render_template("form.html",res=cropname)

    else:

        return render_template('form.html')

@app.route('/',methods=['GET','POST'])
def getdata():

    if request.method=='POST':

        try:

            lang = request.form['lang']
            lat = request.form['lat']
            p1 = {"lat": lat, "lon": lang}
            # lat=15.9268
            # lang=76.6413
            p1={"lat":lat,"lon":lang }
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
            nitrogen=nitrogen*5

            print(ph)
            complete_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lang}&appid=407c59531dcab5eeaacf5f454311d4e6"
            response = requests.get(complete_url)
            x = response.json()
            # print(x)

            cityname=x['name']
            # print(cityname)

            l = x['main']
            #print(type(l))
            #print(l)
            print("temp = ",l["temp"] - (273.15))
            print("humidity = ",l["humidity"])

            temp=l["temp"] - (273.15)
            hum=l["humidity"]

            testdata=np.array([[temp,hum,ph]])
            print(f"Test data = {testdata}")
            model=joblib.load("newknnmodel.pkl")
            
            # testdata=np.array([[nitrogen,temp,hum,ph]])

            
            
            prediction=model.predict(testdata)
            print("knn result=",prediction[0])
            cropimage=f"{prediction[0]}.jpg"

            knnres=prediction[0]

            # model = open("newdtmodel.pkl", 'rb')
            # objectfile = pickle.load(model)

            model=joblib.load("newdtmodel.pkl")
            
            #testdata = np.array([[nitrogen,temp, hum, ph]])
            prediction = model.predict(testdata)
            print("DT result=", prediction[0])
            dtres=prediction[0]

            # model = open("newsvmmodel.pkl", 'rb')
            # objectfile = pickle.load(model)

            # testdata = np.array([[nitrogen,temp, hum, ph]])

            model=joblib.load("randommodel.pkl")
            # prediction = objectfile.predict(testdata)
            print("Random forest result=", prediction[0])

            svmres = prediction[0]

        #     dict = {"Madikeri": "rubber,jute", "Haveri": "maize,rice", "Harihar": "Rice,sugarcane",
        #             "Shiggaon": "Groundnut,maize", "Ranibennur": "Groundnut,Cotton", "Hirekerur": "Jower,maize",
        #             "Savanūr": "cotton,rice", "Haveri": "rice,cotton", "Gulbarga": "jower,sunflower",
        #             "Kolar": "groundnet,fingermillet", "Kushtagi": "Jower,wheat", "Mandya": "ragi,sugercane",
        #             "Heggadadevankote": "Betal leaf,cotton", "Sindhnur": "rice,maize", "Kankanhalli": "coconut,mango",
        #             "Hosanagara": "Arecanut,rice", "Chiknayakanhalli": "coconut,arecanut",
        #             "Hosangadi": "coconut,rubber", "Karwar": "cashew,pineapple",
        #             "BasavanaBagevadi": "Sugercane,groundnet", "Yadgir": "jower,sunflower",
        #             "Mangalore": "Arecanut,Coconut", "Dharwad": "jower,wheat", "Chitradurga": "ponagranut,onion",
        #             "Bidar": "black gram,sugarcane", "Belgaum": "sugacane,tobacco", "Yelahanka": "silk,rice",
        #             "Chikmagalur": "coffee,rice", "Bagalote": "maize,wheat", "Chamrajnagar": "sunflower,banana",
        #             "Gadag": "onion,wheat", "Hassan": "coffee,black pepper", "Closepet": "groundnut,maize",
        #             "Madikeri": "cardmum,coffee"}

        #     string = cityname

        #     new_string = ''.join(char for char in string if char.isalnum())
            dtresultnew=dtres

            data={"place":cityname,"crop1":knnres,"crop2":dtresultnew,"crop3":svmres,"temp":temp,"hum":hum,"ph":ph,"lat":lat,"lang":lang,"silt":silt,"sand":sand,"clay":clay,"nitrogen":nitrogen}
            return render_template('result.html',result=data,ci=cropimage)
        except Exception as e:
            print(f"Problem is : {e}")
            return render_template('noresult.html')
    else:
        return render_template('getlanglat.html')







if __name__=='__main__':
      app.run(debug=True,host="0.0.0.0")