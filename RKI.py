#%%
from sklearn.metrics import confusion_matrix
import kmeansnet
import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd


## Load our Corona-Data
iris = np.loadtxt("C:/Users/stream/Desktop/Caro_Uni/MachLearn/data.csv",delimiter=',')


y = 0
# For loop. to loop over all 9 weeks
for i in range(9):
    index = 2+i
    iris[:,:index] = iris[:,:index]-iris[:,:index].mean(axis=0)

    # Randomly order the data
    order = list(range(np.shape(iris)[0]))
    iris = iris[order,:]

    ## divide the data in  train and test data
    train = iris[::2,0:index]
    valid = iris[1::4,0:index]
    test = iris[3::4,0:index]
    ## since it is not necessary for us to devide the data in test und train
    # w whant to use the whole dataset for the application of the k-mean on the coron dataset
    datensatz = iris[0::1,0:index]

    # calculate the kmeans from train and predict on the test data
    net = kmeansnet.kmeans(3,train)
    net.kmeanstrain(train)
    cluster = net.kmeansfwd(datensatz)
    
    #print("Our calculated k-means cluster:")
    #print(cluster)

    data = iris[0::1,index]
    #print("The //real// data: ")
    #print(data)

    # lodas the original data to add new rows
    dt = pd.read_csv("C:/Users/stream/Desktop/Caro_Uni/MachLearn/data_mitHeader.csv", delimiter=';')
    df = dt[0::1]

    ## add the calculated cluster as a row to the original data, 
    # so that we can retreve  the coordinates
    df["cluster"] = cluster

    ## make an colorarray for later coloring in the map
    cols = []
    j = 0
    for j in df["cluster"]:
        # print(j)
        if j == 0.0:
            colsi = "lightgreen"          
        if j == 1.0:
            colsi = "yellow"        
        if j == 2.0:
            colsi = "red"
            
        cols.append(colsi)
    #df["color"] = cols

    # Scatterplot of the //original// and predicted Corona data
    X = iris[0::1, :index]
    fig,axes = plt.subplots(1, 2, figsize=(16,8))
    ## Plot 1
    axes[0].scatter(X[:, 0], X[:, 1], c=data, cmap='gist_rainbow',edgecolor='k', s=150)
    axes[0].set_xlabel('Sepal length', fontsize=18)
    axes[0].set_ylabel('Sepal width', fontsize=18)
    axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[0].set_title('Actual', fontsize=18)

    ## Plot 2
    axes[1].scatter(X[:, 0], X[:, 1], c=cluster, cmap='gist_rainbow',edgecolor='k', s=150)
    axes[1].set_xlabel('Sepal length', fontsize=18)
    axes[1].set_ylabel('Sepal width', fontsize=18)
    axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[1].set_title('Predicted', fontsize=18)
    
    #plt.savefig('C:/Users/stream/Desktop/Caro_Uni/MachLearn/PlotWeek'+str(y))

    ## create Map with German coordinates
    ## Make Leaflet Map
    ## Retrieve german Counties from GitHub als GEOJson
    url = 'https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/master/4_kreise'
    state_geo = f'{url}/3_mittel.geo.json'
    m = folium.Map(location=[52.2, 9], zoom_start=6)
    folium.Choropleth(
        geo_data=state_geo,
        # not really necassary here but useful for future work 
        name='choropleth',
        data=df["cluster"],
        ).add_to(m)

    ## Preprocess the Coordinates, so we can use them for our map
    latlon=[]
    clusterarray=[]
    #print(df)
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
    m.save('C:/Users/stream/Desktop/Caro_Uni/MachLearn/index'+str(y)+'.html')


# %%
