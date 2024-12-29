import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def sales_prediction_page():
    st.title("Parashikimi i shitjeve duke perdorur Machine Learning Algorithms :chart_with_upwards_trend:")

    uploaded_file = st.file_uploader("Ngarkoni nje file CSV  me te dhenat historike te shitjeve: ", accept_multiple_files=False)
    ###
    # 'https://medium.com/@deyzra/forecast-sales-using-machine-learning-case-store-item-demand-and-forecasting-challenge-train-csv-49911cd1ed34'
    ###
    if uploaded_file is not None:
        store_sales = pd.read_csv(uploaded_file)
        check = st.checkbox("Shfaq dataset")
        if check:
            st.dataframe(store_sales, use_container_width=True, hide_index=True)
        # col1, col2 = st.columns(2)
        selected_model = st.sidebar.selectbox("Zgjidhni algoritmin", options=["Linear Regeression", "Random Forest"])
        # store_sales = pd.read_csv('Sales/train.csv')
        store_sales = store_sales.drop(['store','item'], axis=1)
        store_sales['date'] = pd.to_datetime(store_sales['date'])
        store_sales['date'] = store_sales['date'].dt.to_period('M')
        monthly_sales = store_sales.groupby('date').sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        last_year_sales = monthly_sales[-12:]['sales']

        # plt.figure(figsize=(15,5))
        # plt.plot(monthly_sales['date'], monthly_sales['sales'])
        # plt.xlabel('Date')
        # plt.xlabel('Sales')
        # plt.title("Monthly Customer Sales")
        # st.pyplot(plt.gcf())


        monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
        monthly_sales = monthly_sales.dropna()

        # plt.figure(figsize=(15,5))
        # plt.plot(monthly_sales['date'], monthly_sales['sales_diff'])
        # plt.xlabel('Date')
        # plt.xlabel('Sales')
        # plt.title("Monthly Customer Sales Diff")
        # st.pyplot(plt.gcf())

        supverised_data = monthly_sales.drop(['date','sales'], axis=1)
        for i in range(1,13):
            col_name = 'month_' + str(i)
            supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
        supverised_data = supverised_data.dropna().reset_index(drop=True)
        # st.dataframe(supverised_data.head(10))

        train_data = supverised_data[:-12]
        test_data = supverised_data[-12:]
        # st.write('Train Data Shape:', train_data.shape)
        # st.write('Test Data Shape:', test_data.shape)

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        X_train, y_train = train_data[:,1:], train_data[:,0:1]
        X_test, y_test = test_data[:,1:], test_data[:,0:1]
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        # st.write('X_train Shape:', X_train.shape)
        # st.write('y_train Shape:', y_train.shape)
        # st.write('X_test Shape:', X_test.shape)
        # st.write('y_test Shape:', y_test.shape)

        sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
        predict_df = pd.DataFrame(sales_dates)
        # predict_df

        act_sales = monthly_sales['sales'][-13:].to_list()
        if selected_model == "Linear Regeression":
            linreg_model = LinearRegression()
            linreg_model.fit(X_train, y_train)
            linreg_pred = linreg_model.predict(X_test)

            linreg_pred = linreg_pred.reshape(-1,1)
            linreg_pred_test_set = np.concatenate([linreg_pred,X_test], axis=1)
            linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

            result_list = []
            for index in range(0, len(linreg_pred_test_set)):
                result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
            linreg_pred_series = pd.Series(result_list,name='Vlerat e parashikuara')
            predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)
            predict_df['date'] = predict_df['date'].dt.to_period('D')

            st.subheader('Vlerat e parashikuara te shitjeve vjetore')
            # predict_df['Vlerat Reale'] = monthly_sales['sales'][-12:]
            # a = predict_df.append(monthly_sales[-12:]['sales'], ignore_index=True)
            predict_df.insert(1, 'Vlerat origjinale', last_year_sales.values)
            predict_df

            linreg_rmse = np.sqrt(mean_squared_error(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:]))
            linreg_mae = mean_absolute_error(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:])
            linreg_r2 = r2_score(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:])
            perc_lr = linreg_r2 * 100
            # st.write('Linear Regression RMSE: ', linreg_rmse)
            # st.write('Linear Regression MAE: ', linreg_mae)
            # st.write('Linear Regression R2 Score: ', linreg_r2)
            # st.write(f'Saktesia e Algoritmit Linear Regression: {perc_lr:.2f}%')

            plt.figure(figsize=(15,7))
            plt.plot(monthly_sales['date'], monthly_sales['sales'])
            plt.plot(predict_df['date'], predict_df['Vlerat e parashikuara'])
            plt.title("Parashikimi i shitjeve me Linear Regression")
            plt.xlabel("Periudha")
            plt.ylabel("Numri i shitjeve")
            plt.legend(["Shitjet Origjinale", "Shitjet e Parashikuara"])
            plt.grid()
            st.pyplot(plt.gcf())
            st.success(f'Saktesia e Algoritmit Linear Regression: {perc_lr:.2f}%')
        else:
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=20)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            rf_pred = rf_pred.reshape(-1,1)
            rf_pred_test_set = np.concatenate([rf_pred,X_test], axis=1)
            rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

            result_list = []
            for index in range(0, len(rf_pred_test_set)):
                result_list.append(rf_pred_test_set[index][0] + act_sales[index])
            rf_pred_series = pd.Series(result_list, name='Vlerat e parashikuara')
            predict_df = predict_df.merge(rf_pred_series, left_index=True, right_index=True)
            predict_df['date'] = predict_df['date'].dt.to_period('D')
            st.subheader('Vlerat e parashikuara te shitjeve vjetore')
            predict_df.insert(1, 'Vlerat origjinale', last_year_sales.values)
            predict_df
            # predict_df

            rf_rmse = np.sqrt(mean_squared_error(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:]))
            rf_mae = mean_absolute_error(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:])
            rf_r2 = r2_score(predict_df['Vlerat e parashikuara'], monthly_sales['sales'][-12:])
            perc_rf = rf_r2*100
            # st.write('Random Forest RMSE: ', rf_rmse)
            # st.write('Random Forest MAE: ', rf_mae)

            plt.figure(figsize=(15,7))
            plt.plot(monthly_sales['date'], monthly_sales['sales'])
            plt.plot(predict_df['date'], predict_df['Vlerat e parashikuara'])
            plt.title("Parashikimi i shitjeve me Random Forest")
            plt.xlabel("Periudha")
            plt.ylabel("Numri i shitjeve")
            plt.legend(["Shitjet Origjinale", "Shitjet e Parashikuara"])
            # plt.grid()
            plt.grid()
            st.pyplot(plt.gcf())

            st.success(f'Saktesia e Algoritmit Random Forest: {perc_rf:.2f}%')