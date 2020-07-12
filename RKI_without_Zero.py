#%%
from sklearn.metrics import confusion_matrix
import kmeansnet
import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd

def deleteZeroNumpy(data, wochenzaehler):
    for i in corona:
        week = i
        for o in range(0, len(week)):
            value = week[wochenzaehler]
            if value == 0.0:
                #print(o)
                data = np.delete(data, (o), axis=0)
                break    
    return data        

# nulls = []
# def deleteZeroNumpy(data, wochenzaehler):
#     indexX = 0
#     for i in data:
#         week = i
#         # for o in range(0, len(week)):
#         #     print(o)
#         value = week[wochenzaehler]
#         #print(value)
#         if value == 0.0:
#             #print(value)
#             nulls.append(indexX)
#         indexX = indexX + 1
    
#     result = np.delete(data, nulls, 0)
#     return result        



def deleteZeroPandas(data, wochenzaehler, titel):
    array=[]
    #print(wochenzaehler)
    nummer = titel[wochenzaehler]
    week = data[str(nummer)]
    ## Check if value is 0 
    for o in range(0, len(data)):
        value = week.iloc[o]
        if value == 0:
            array.append(data.index[o])
        
    data = data.drop(array)    
    return data

y = 0
wochenzaehler = 2
# For loop. to loop over all 9 weeks
for i in range(9):
    ## Load our Corona-Data
    corona = np.loadtxt("C:/Users/stream/Desktop/Caro_Uni/MachLearn/data.csv",delimiter=',')
    index = 2+i
    print(len(corona))
    corona = deleteZeroNumpy(corona, wochenzaehler)
    print(len(corona))
  
    corona[:,:index] = corona[:,:index]-corona[:,:index].mean(axis=0)

    # Randomly order the data
    order = list(range(np.shape(corona)[0]))
    corona = corona[order,:]

    ## divide the data in  train and test data
    train = corona[::2,0:index]
    valid = corona[1::4,0:index]
    test = corona[3::4,0:index]
    datensatz = corona[0::1,0:index]

    # calculate the kmeans from train and predict on the test data
    net = kmeansnet.kmeans(3,train)
    net.kmeanstrain(train)
    cluster = net.kmeansfwd(datensatz)
    
    #print("Our calculated k-means cluster:")
    #print(cluster)

    data = corona[0::1,index]
    #print("The //real// data: ")
    #print(data)

    # lodas the original data to add new rows
    names= ["LATITUDE","LONGITUDE","W1","W2","W3","W4","W5","W6","W7","W8","W9"]
    dt_header = pd.read_csv("C:/Users/stream/Desktop/Caro_Uni/MachLearn/data_mitHeader.csv", delimiter=';', usecols=names)
    dt = deleteZeroPandas(dt_header, wochenzaehler, names)
    df = dt[0::1]
    wochenzaehler = wochenzaehler+1

    ## add the calculated cluster as a row to the original data, 
    # so that we can retreve  the coordinates
    #print(len(cluster))
    #df["cluster"] = cluster

    ## make an colorarray for later coloring in the map
    cols = []
    j = 0
    for j in cluster:
        # print(j)
        if j == 0.0:
            colsi = "lightgreen"          
        if j == 1.0:
            colsi = "yellow"        
        if j == 2.0:
            colsi = "red"
            
        cols.append(colsi)


        # Scatterplot of the //original// and predicted Corona data
    X = corona[0::1, :index]
    fig,axes = plt.subplots(1, 2, figsize=(16,8))
    ## Plot 1
    axes[0].scatter(X[:, 0], X[:, 1], c=data, cmap='gist_rainbow',edgecolor='k', s=150)
    axes[0].set_xlabel('LATITUDE', fontsize=18)
    axes[0].set_ylabel('LONGITUDE', fontsize=18)
    axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[0].set_title('Actual', fontsize=18)

    ## Plot 2
    axes[1].scatter(X[:, 0], X[:, 1], c=cluster, cmap='gist_rainbow',edgecolor='k', s=150)
    axes[1].set_xlabel('LATITUDE', fontsize=18)
    axes[1].set_ylabel('LONGITUDE', fontsize=18)
    axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[1].set_title('Predicted', fontsize=18)

    plt.savefig('C:/Users/stream/Desktop/Caro_Uni/MachLearn/PlotWeekWithoutZero'+str(y))
    ## create Map with German coordinates
    ## Make Leaflet Map
    ## Retrieve german Counties from GitHub als GEOJson
    url = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/master/4_kreise'
    state_geo = f'{url}/3_mittel.geo.json'
    m = folium.Map(location=[52.2, 9], zoom_start=6)
    folium.Choropleth(
        geo_data=state_geo,
        name='choropleth',
        #data=df["cluster"],
        ).add_to(m)

    ## Preprocess the Coordinates, so we can use them for our map
    latlon=[]
    clusterarray=[]
    for l in range(0, len(df)):
        # Here we choose the whole Lat and lon column
        lat = df["LATITUDE"]
        lon = df["LONGITUDE"]

        # Get every single number/row from the column
        lati = lat.iloc[l]
        loni = lon.iloc[l]
        ## ... and append them as [lat,lon] coords in an array
        latlon.append([lati,loni])
    
    ## loop over all coordinates and add the respective color for the cluster value
    k = 0
    for coord in latlon:
        folium.Circle(radius=100, location=[coord[0], coord[1]],color=cols[k],fill=True,
            ).add_to(m)
        k = k+1
    
    for y in range(8):
        y = i+1

    m.add_child(folium.LatLngPopup())
    folium.LayerControl().add_to(m)
    
    #print(y)

    m
    m.save('C:/Users/stream/Desktop/Caro_Uni/MachLearn/indexWithoutZero'+str(y)+'.html')


# %%

