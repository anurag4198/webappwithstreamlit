import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
import matplotlib as plt
import seaborn as sns


#sidebar
st.sidebar.header('Anomaly Detection')
model=st.sidebar.selectbox('Function',('Home','Load & overview data','Deploy model', 'View model') )

if model=='Load & overview data':
    st.header('Load CSV data')
    data=st.file_uploader('Upload your dataset',type=['csv'])
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df.head())
        #plot
        all_columns_names = df.columns.to_list()
        selected_columns_names = st.multiselect('select columns to plot',all_columns_names)
        cust_data = df[selected_columns_names]
        st.area_chart(cust_data)
        st.bar_chart(cust_data)
        st.line_chart(cust_data)
        cust_dataNN = df[selected_columns_names].plot(kind='hist')
        st.write(cust_dataNN)
        st.pyplot()
        cust_dataNN = df[selected_columns_names].plot(kind='box')
        st.write(cust_dataNN)
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

elif model == 'Deploy model':
    modelList=st.sidebar.selectbox('Model List',('Select a DNN model.....','AutoEncoders','CNN','LSTM') )

    if modelList == 'AutoEncoders':
        st.write('Deploying AutoEncoder model...')
        # Add code to deploy AutoEncoder model here

        dataset=st.file_uploader('Upload your dataset',type=['csv'])
        if dataset is not None:
            df=pd.read_csv(dataset)
            st.dataframe(df.head())
            st.subheader('Press the button to Train DNN Model')

            if "button_clicked" not in st.session_state:    
                st.session_state.button_clicked = False
            def callback():
                st.session_state.button_clicked= True

            if st.button ('Train Model',on_click=callback) | st.session_state.button_clicked:

                non_fraud = df[df.Class == 0]
                fraud = df[df.Class == 1]

                # Standard Scaling 'Amount' feature by dropping time
                data = df.drop(['Time (second)'], axis = 1)
                data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

                # Training the data
                X_train, X_test = train_test_split(data, test_size= 0.2, random_state = 42) # Might need to change random state later for better result
                X_train = X_train[X_train.Class == 0]
                X_train = X_train.drop(['Class'], axis = 1)

                y_test = X_test['Class']
                X_test = X_test.drop(['Class'], axis = 1)

                X_train = X_train.values
                X_test = X_test.values

                # Declaring the dimensions 
                input_dim = X_train.shape[1]
                encoding_dim = 14

                # defining the layers

                #input layer
                input_layer = Input(shape = (input_dim, ))

                # encoder
                encoder = Dense(encoding_dim, activation = 'tanh')(input_layer)
                encoder = Dense(int(encoding_dim / 2), activation = 'relu')(encoder)

                #decoder
                decoder = Dense(int(encoding_dim / 2), activation = 'tanh')(encoder)
                decoder = Dense(input_dim, activation = 'relu')(decoder)

                autoencoder = Model(inputs = input_layer, outputs = decoder)

                # declaring the epochs, batch size and running the 
                nb_epoch = 10
                batch_size = 32

                # compiling the autoencoder
                autoencoder.compile(optimizer = 'Adam',
                                    loss = 'mean_squared_error',
                                    metrics = ['Accuracy'])

                # checkpointer -> to save the results temporarily


                # training the data upto epochs
                history = autoencoder.fit(X_train, X_train,
                                        epochs = nb_epoch,
                                        batch_size = batch_size,
                                        shuffle = 'True',
                                        validation_data = (X_test, X_test),
                                        verbose = 1).history
                


                predictions = autoencoder.predict(X_test)

                st.success('Model Trained Successfully')

                
                if st.button('Visualization'): 

                    # redefining Mean squared error
                    mse = np.mean(np.power(X_test - predictions, 2), axis = 1) #reference this one
                    error_df = pd.DataFrame({'reconstruction_error': mse,
                                            'true_class': y_test})

                    fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
                    roc_auc = auc(fpr, tpr)

                    st.write('ROC Score: ', roc_auc)

                    threshold = 4.5

                    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
                    conf_matrix = confusion_matrix(error_df.true_class, y_pred)

                    # Visualizing the confusion matrix
                    st.write('Visualising the data')
                    sns.heatmap(conf_matrix, annot = True, fmt = 'd')
                    st.pyplot()
                    
                    # Function to plot confusion matrix

                    st.write('Evaluating the scores..')
                    st.write('Accuracy', accuracy_score(error_df.true_class, y_pred)) #need to check it later
                    st.write('F1-score', f1_score(error_df.true_class, y_pred, average = 'micro'))
                    st.write('Precision', precision_score(error_df.true_class, y_pred, average = 'macro'))
                    st.write('Recall', recall_score(error_df.true_class, y_pred, average = 'macro'))
                    
                    st.line_chart(history)
                    st.bar_chart(history)
                    st.area_chart(history)
                



    elif modelList == 'CNN':
        st.write('Deploying CNN model...')
        # Add code to deploy CNN model here
        data=st.file_uploader('Upload your dataset',type=['csv'])
        if data is not None:
            dataset=pd.read_csv(data)
            st.dataframe(dataset.head())
        
            # Preprocessing
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values

            # Normalize data
            sc = StandardScaler()
            X = sc.fit_transform(X)

            # Reshape data for CNN model
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create CNN model
            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=(X.shape[1], 1)))
            model.add(Conv1D(64, 2, activation='relu'))
            # model.add(MaxPooling1D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the model
            history = model.fit(X_train, y_train, epochs=10, batch_size=32)

            # Evaluate the model
            score = model.evaluate(X_test, y_test)
            st.write(f'Test Loss: {score[0]}')
            st.write(f'Test Accuracy: {score[1]}')


            st.write('Visualization')

            st.line_chart(history.history)
            st.bar_chart(history.history)
            st.area_chart(history.history)
            

    elif modelList == 'LSTM':
        st.write('Deploying LSTM model...')
        # Add code to deploy LSTM model here

    else:
        st.subheader('Please select a DNN Model!!')