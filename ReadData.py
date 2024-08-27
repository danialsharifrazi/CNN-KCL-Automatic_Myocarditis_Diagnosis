def Read_Data():    
    import numpy as np
    import cv2
    from skimage.io import imread
    import glob
    import os


    normals=[]
    main_path='./Converted Dataset/Normal/'
    main_folders=next(os.walk(main_path))[1]
    for i in main_folders:
        path=main_path+i+'/'
        folders=next(os.walk(path))[1]
        for x in folders:
            new_path=path+x+'/'
            data=glob.glob(new_path+'*.jpg')
            if len(data)<1:
                indent_folders=next(os.walk(new_path))[1]
                for y in indent_folders:
                    new_path=new_path+y+'/'
                    data=glob.glob(new_path+'*.jpg')
            normals.extend(data)


    #read sicks files
    sicks=[]
    main_path='./Converted Dataset/Sick/'
    main_folders=next(os.walk(main_path))[1]
    for i in main_folders:
        path=main_path+i+'/'
        folders=next(os.walk(path))[1]
        for x in folders:
            new_path=path+x+'/'
            data=glob.glob(new_path+'*.jpg')
            if len(data)<1:
                indent_folders=next(os.walk(new_path))[1]
                for y in indent_folders:
                    new_path=new_path+y+'/'
                    data=glob.glob(new_path+'*.jpg')
            sicks.extend(data)
    
    #load normal files
    labels_n=[]
    train_data_n=[]
    for id in normals:    
        img=imread(id)
        img=cv2.resize(img,(100,100))
        img=img.flatten()
        train_data_n.append(img)
        labels_n.append(0)


    #load sick files
    labels_s=[]
    train_data_s=[]
    for id in sicks:    
        img=imread(id)
        img=cv2.resize(img,(100,100))
        img=img.flatten()
        train_data_s.append(img)
        labels_s.append(1)

    train_data_n.extend(train_data_s)
    labels_n.extend(labels_s)

    x_data=np.array(train_data_n)
    y_data=np.array(labels_n)


    k=1    
    from CNN_kfold import DeepCNN
    DeepCNN(x_data,y_data,k)


Read_Data()
