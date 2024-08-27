def clusteringFunc(x_data,y_data,k):    
    import numpy as np
    from sklearn.cluster import KMeans

 
    print('Start Clustering.............!')
    normals=[]
    sicks=[]
    for i in range(len(y_data)):
        if y_data[i]==0:
            normals.append(x_data[i])
        else:
            sicks.append(x_data[i])

    normals=np.array(normals)
    sicks=np.array(sicks)


    model=KMeans(n_clusters=k)   
    y_n=model.fit_predict(normals)  
    y_s=model.fit_predict(sicks)
    

    y_s2=[]
    for item in y_s:
        y_s2.append(item+k)


    y_n=list(y_n)
    y_n.extend(y_s2)
    y=np.array(y_n)


    normals=list(normals)
    sicks=list(sicks)
    normals.extend(sicks)
    x=np.array(normals)

    return x,y


