def DeepCNN(x_data,y_data,k):

    import datetime
    from sklearn.metrics import ( auc, classification_report,
                                confusion_matrix, roc_curve)
    from sklearn.model_selection import KFold, train_test_split
    from keras.layers import Conv1D,Dense, Dropout, Flatten
    from keras.losses import binary_crossentropy
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.utils import np_utils
    from keras.callbacks import CSVLogger
    import PlotHistory_kFold
    from Clustering import clusteringFunc


    print('Start Deep Learning............!')


    lst_loss=[]
    lst_acc=[]
    lst_reports=[]
    lst_AUC=[]
    lst_matrix=[]
    lst_times=[]
    lst_history=[]
    fold_number=1
    n_epch=30

    kfold=KFold(n_splits=10,shuffle=True,random_state=None)
    for train,test in kfold.split(x_data,y_data):

        x_train=x_data[train]
        x_test=x_data[test]
        y_train=y_data[train]
        y_test=y_data[test]

        x_train=x_train.reshape((x_train.shape[0],100,100))
        x_test=x_test.reshape((x_test.shape[0],100,100))

        x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=None)

        print(f'train: {x_train.shape}  {y_train.shape}')
        print(f'test: {x_test.shape}  {y_test.shape}')
        print(f'valid: {x_test.shape}  {y_valid.shape}')


        calback=CSVLogger(f'./logger_fold{fold_number}.log')

        y_train=np_utils.to_categorical(y_train)
        y_test=np_utils.to_categorical(y_test)
        y_valid=np_utils.to_categorical(y_valid)



        #Architecture CNN
        model=Sequential()
        model.add(Conv1D(32,3,padding='same',activation='relu',strides=2,input_shape=(100,100)))
        model.add(Conv1D(64,3,padding='same',activation='relu',strides=2))
        model.add(Conv1D(128,3,padding='same',activation='relu',strides=2))
        model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
        model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
        model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(k*2,activation='sigmoid'))

        model.compile(optimizer=Adam(),loss=binary_crossentropy,metrics=['accuracy'])
            
        start=datetime.datetime.now()
        net_history=model.fit(x_train, y_train, batch_size=512, epochs=n_epch,validation_data=[x_valid,y_valid],callbacks=[calback])
        end=datetime.datetime.now()
        training_time=end-start

        model.save(f'./CNN_fold{fold_number}.h5')

        test_loss, test_acc=model.evaluate(x_test,y_test)

        predicts=model.predict(x_test)
        predicts=predicts.argmax(axis=1)
        actuals=y_test.argmax(axis=1)

        fpr,tpr,_=roc_curve(actuals,predicts)
        a=auc(fpr,tpr)
        r=classification_report(actuals,predicts)
        c=confusion_matrix(actuals,predicts)



        lst_history.append(net_history)
        lst_times.append(training_time)
        lst_acc.append(test_acc)
        lst_loss.append(test_loss)
        lst_AUC.append(a)
        lst_reports.append(r)
        lst_matrix.append(c)
        fold_number+=1


    path=f'./CNN_Kmeans_Results.txt' 
    f1=open(path,'a')
    f1.write('\nAccuracies: '+str(lst_acc)+'\nLosses: '+str(lst_loss))
    f1.write('\n\nMetrics for all Folds: \n\n')
    for i in range(len(lst_reports)):
        f1.write(str(lst_reports[i]))
        f1.write('\n\nTraining Time: '+str(lst_times[i])+'\nAUC: '+str(lst_AUC[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n__________________________________________________________\n')
    f1.close()



