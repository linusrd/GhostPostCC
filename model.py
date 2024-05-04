import csv
import math
import sys
from datetime import datetime as dt
from datetime import timezone as tz

import numpy as np
import pandas as pd
import pytz as pytz
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scipy.sparse import data
from sklearn.utils import validation
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import pickle
import utility

class ANNModel():
    def __init__(self):
        self.NUM_FEATURES_DICT = {"coal": 6, "nat_gas": 6, "nuclear": 6, "oil": 6, "hydro": 6, "solar": 6,
                            "wind": 6, "others": 6}

        self.NUM_VAL_DAYS = 30
        self.NUM_TEST_DAYS = 184
        self.TRAINING_WINDOW_HOURS = 24
        self.PREDICTION_WINDOW_HOURS = 24
        self.MODEL_SLIDING_WINDOW_LEN = 24

        COAL = 1
        NAT_GAS = 2
        NUCLEAR = 3
        OIL = 4
        HYDRO = 5
        SOLAR = 6
        WIND = 7
        OTHERS = 8

        self.FUEL = {COAL: "coal", NAT_GAS: "nat_gas", NUCLEAR: "nuclear", OIL: "oil", HYDRO: "hydro", SOLAR: "solar", WIND: "wind", OTHERS: "others"}
        self.SOURCE_TO_SOURCE_COL_MAP = {y: x for x, y in self.FUEL.items()}

        self.CARBON_INTENSITY_COLUMN = 1
        self.carbonRateDirect = {"avg_coal_production_forecast": 1003.7, "avg_biomass_production_forecast": 0, 
                        "avg_nat_gas_production_forecast": 409.43, "avg_geothermal_production_forecast": 0, 
                        "avg_hydro_production_forecast": 0, "avg_nuclear_production_forecast": 0, 
                        "avg_oil_production_forecast": 406, "avg_solar_production_forecast": 0, 
                        "avg_unknown_production_forecast": 575, "avg_others_production_forecast": 575, 
                        "avg_wind_production_forecast": 0} # g/kWh

    def initDataset(self, inFileName, sourceCol):
        dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, parse_dates=['UTC Time at End of Hour'], index_col=['UTC Time at End of Hour'])

        print(dataset.head())
        print(dataset.columns)
        dateTime = dataset.index.values
        
        print("\nAdding features related to date & time...")
        modifiedDataset = utility.addDateTimeFeatures(dataset, dateTime, sourceCol)
        dataset = modifiedDataset
        print("Features related to date & time added")
        
        for i in range(sourceCol, len(dataset.columns.values)):
            col = dataset.columns.values[i]
            dataset[col] = dataset[col].astype(np.float64)
            # print(col, dataset[col].dtype)

        return dataset, dateTime

    # convert training data into inputs and outputs (labels)
    def manipulateTrainingDataShape(self, data, trainWindowHours, labelWindowHours): 
        print("Data shape: ", data.shape)
        X, y = list(), list()
        # step over the entire history one time step at a time
        for i in range(len(data)-(trainWindowHours+labelWindowHours)+1):
            # define the end of the input sequence
            trainWindow = i + trainWindowHours
            labelWindow = trainWindow + labelWindowHours
            xInput = data[i:trainWindow, :]
            # xInput = xInput.reshape((len(xInput), 1))
            X.append(xInput)
            y.append(data[trainWindow:labelWindow, 0])
            # print(data[trainWindow:labelWindow, 0])
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    def manipulateTestDataShape(self, data, slidingWindowLen, predictionWindowHours, isDates=False): 
        X = list()
        # step over the entire history one time step at a time
        for i in range(0, len(data)-(predictionWindowHours)+1, slidingWindowLen):
            # define the end of the input sequence
            predictionWindow = i + predictionWindowHours
            X.append(data[i:predictionWindow])
        if (isDates is False):
            X = np.array(X, dtype=np.float64)
        else:
            X = np.array(X)
        return X

    def trainANN(self, trainX, trainY, valX, valY, hyperParams, modelDir):
        n_timesteps, n_features, nOutputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
        epochs = 1 #hyperParams['epoch']
        batchSize = hyperParams['batchsize']
        activationFunc = hyperParams['actv']
        lossFunc = hyperParams['loss']
        optimizer = hyperParams['optim']
        hiddenDims = hyperParams['hidden']
        learningRates = hyperParams['lr']
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(hiddenDims[0], input_shape=(n_timesteps, n_features), activation=activationFunc)) # 20 for coal, nat_gas, nuclear
        model.add(Dense(hiddenDims[1], activation='relu')) # 50 for coal, nat_gas, nuclear
        model.add(Dense(nOutputs))

        opt = tf.keras.optimizers.Adam(learning_rate = learningRates)
        model.compile(loss=lossFunc, optimizer=optimizer[0],
                        metrics=['mean_absolute_error'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(modelDir, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # fit network
        hist = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize[0], verbose=2,
                            validation_data=(valX, valY), callbacks=[es, mc])
        model = load_model(modelDir)
        utility.showModelSummary(hist, model)
        return model, n_features

    def getDayAheadForecasts(self, trainX, trainY, model, history, testData, 
                                trainWindowHours, numFeatures, depVarColumn):
        # walk-forward validation over each day
        print("Testing...")
        predictions = list()
        for i in range(0, len(testData)//24):
            dayAheadPredictions = list()
            tempHistory = history.copy()
            currentDayHours = i* self.MODEL_SLIDING_WINDOW_LEN
            for j in range(0, self.PREDICTION_WINDOW_HOURS, 24):
                yhat_sequence, newTrainingData = self.getForecasts(model, tempHistory, trainWindowHours, numFeatures)
                dayAheadPredictions.extend(yhat_sequence)
                # add current prediction to history for predicting the next day
                # following 3 lines are redundant currently. Will become useful if
                # prediction period goes beyond 24 hours.
                latestHistory = testData[currentDayHours+j:currentDayHours+j+24, :].tolist()
                for k in range(24):
                    latestHistory[k][depVarColumn] = yhat_sequence[k]
                tempHistory.extend(latestHistory)

            # get real observation and add to history for predicting the next day
            history.extend(testData[currentDayHours:currentDayHours+self.MODEL_SLIDING_WINDOW_LEN, :].tolist())
            predictions.append(dayAheadPredictions)

        # evaluate predictions days for each day
        predictedData = np.array(predictions, dtype=np.float64)
        return predictedData


    def getForecasts(self, model, history, trainWindowHours, numFeatures):
        # flatten data
        data = np.array(history, dtype=np.float64)
        # retrieve last observations for input data
        input_x = data[-trainWindowHours:]
        # reshape into [1, n_input, num_features]
        input_x = input_x.reshape((1, len(input_x), numFeatures))
        # print("ip_x shape: ", input_x.shape)
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat, input_x

    def getANNHyperParams(self):
        hyperParams = {}
        hyperParams['epoch'] = 100 
        hyperParams['batchsize'] = [10] 
        hyperParams['actv'] = "relu"
        hyperParams['loss'] = "mse"
        hyperParams['optim'] = ["adam"] #, "rmsprop"]
        hyperParams['lr'] = 1e-2 #, 1e-3
        hyperParams['hidden'] = [20, 50] #, [50, 50]]#, [20, 50]] #, [50, 50]]
        return hyperParams

    def forecast_all_fuel_sources(self, fuel_sources):
        for source in fuel_sources:
            IN_FILE_NAME = "data/MW_electricity_cleaned.csv"

            OUT_FILE_NAME_PREFIX = 'data/src_prod_forecast'
            OUT_FILE_NAME = OUT_FILE_NAME_PREFIX + "_" + source + ".csv"
            OUT_MODEL_NAME = 'model/' + source + "_ann.keras"

            SOURCE_COL = self.SOURCE_TO_SOURCE_COL_MAP[source]
            NUM_FEATURES = self.NUM_FEATURES_DICT[self.FUEL[SOURCE_COL]]

            dataset, dateTime = self.initDataset(IN_FILE_NAME, SOURCE_COL)

            trainData, valData, testData, fullTrainData = utility.splitDataset(dataset.values, self.NUM_TEST_DAYS, self.NUM_VAL_DAYS)
            trainDates = dateTime[: -(self.NUM_TEST_DAYS*24)]
            fullTrainDates = np.copy(trainDates)
            trainDates, validationDates = trainDates[: -(self.NUM_VAL_DAYS*24)], trainDates[-(self.NUM_VAL_DAYS*24):]
            testDates = dateTime[-(self.NUM_TEST_DAYS*24):]
            trainData = trainData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]
            valData = valData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]
            testData = testData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]

            print("TrainData shape: ", trainData.shape) # (days x hour) x features
            print("ValData shape: ", valData.shape) # (days x hour) x features
            print("TestData shape: ", testData.shape) # (days x hour) x features
            print("***** Dataset split done *****")

            for i in range(trainData.shape[0]):
                for j in range(trainData.shape[1]):
                    if(np.isnan(trainData[i, j])):
                        trainData[i, j] = trainData[i-1, j]

            for i in range(valData.shape[0]):
                for j in range(valData.shape[1]):
                    if(np.isnan(valData[i, j])):
                        valData[i, j] = valData[i-1, j]

            for i in range(testData.shape[0]):
                for j in range(testData.shape[1]):
                    if(np.isnan(testData[i, j])):
                        testData[i, j] = testData[i-1, j]

            featureList = dataset.columns.values[SOURCE_COL:SOURCE_COL+NUM_FEATURES]
            print("Features: ", featureList)

            print("Scaling data...")
            trainData, valData, testData, ftMin, ftMax = utility.scaleDataset(trainData, valData, testData)
            print("***** Data scaling done *****")
            print(trainData.shape, valData.shape, testData.shape)

            print("\nManipulating training data...")
            X, y = self.manipulateTrainingDataShape(trainData, self.TRAINING_WINDOW_HOURS, self.TRAINING_WINDOW_HOURS)
            valX, valY = self.manipulateTrainingDataShape(valData, self.TRAINING_WINDOW_HOURS, self.TRAINING_WINDOW_HOURS)
            print("***** Training data manipulation done *****")
            print("X.shape, y.shape: ", X.shape, y.shape)

            hyperParams = self.getANNHyperParams()

            print("\nStarting training)...")
            bestModel, numFeatures = self.trainANN(X, y, valX, valY, hyperParams, OUT_MODEL_NAME)
            print("***** Training done *****")
            history = valData[-self.TRAINING_WINDOW_HOURS:, :].tolist()
            predictedData = self.getDayAheadForecasts(X, y, bestModel, history, testData, 
                            self.TRAINING_WINDOW_HOURS, numFeatures, 0)            
            actualData = self.manipulateTestDataShape(testData[:, 0], 
                    self.MODEL_SLIDING_WINDOW_LEN, self.PREDICTION_WINDOW_HOURS, False)
            formattedTestDates = self.manipulateTestDataShape(testDates, 
                    self.MODEL_SLIDING_WINDOW_LEN, self.PREDICTION_WINDOW_HOURS, True)
            formattedTestDates = np.reshape(formattedTestDates, 
                    formattedTestDates.shape[0]*formattedTestDates.shape[1])
            actualData = actualData.astype(np.float64)
            print("ActualData shape: ", actualData.shape)
            actual = np.reshape(actualData, actualData.shape[0]*actualData.shape[1])
            print("actual.shape: ", actual.shape)
            unscaledTestData = utility.inverseDataScaling(actual, ftMax[0], 
                            ftMin[0])
            predictedData = predictedData.astype(np.float64)
            print("PredictedData shape: ", predictedData.shape)
            predicted = np.reshape(predictedData, predictedData.shape[0]*predictedData.shape[1])
            print("predicted.shape: ", predicted.shape)
            unScaledPredictedData = utility.inverseDataScaling(predicted, 
                    ftMax[0], ftMin[0])
            rmseScore, mapeScore = utility.getScores(actualData, predictedData, 
                                    unscaledTestData, unScaledPredictedData)
            print("***** Forecast done *****")
            print("Overall RMSE score: ", rmseScore)
            print(rmseScore)

            data = [bestModel, ftMin, ftMax]
            with open('model/' + source + "_ann.pkl", 'wb') as f:
                for d in data:
                    pickle.dump(d, f)

            data = []
            for i in range(len(unScaledPredictedData)):
                    row = []
                    row.append(str(formattedTestDates[i]))
                    row.append(str(unscaledTestData[i]))
                    row.append(str(unScaledPredictedData[i]))
                    data.append(row)
            utility.writeOutFuelForecastFile(OUT_FILE_NAME, data, source)

    def scale_dataset_trained(self, valData, testData, ftMin, ftMax):
        # Scaling columns to range (0, 1)
        row, col = valData.shape[0], valData.shape[1]
        for i in range(col):
            if((ftMax[i] - ftMin[i]) == 0):
                continue
            valData[:, i] = (valData[:, i] - ftMin[i]) / (ftMax[i] - ftMin[i])
            testData[:, i] = (testData[:, i] - ftMin[i]) / (ftMax[i] - ftMin[i])

        return valData, testData

    def get_day_ahead_forecasts(self, model_filepath, history, train_window_hours, num_features, dep_var_column):
        
        # Load the trained ANN model
        model = load_model(model_filepath)

        # walk-forward validation over each day
        print("Generating day-ahead forecasts...")
        predictions = []
        for i in range(len(history) // 24):
            day_ahead_predictions = []
            temp_history = history.copy()
            current_day_hours = i * self.MODEL_SLIDING_WINDOW_LEN
            for j in range(0, self.PREDICTION_WINDOW_HOURS, 24):
                # Get forecasts for the next 24 hours
                yhat_sequence, new_training_data = self.get_forecasts(model, temp_history, train_window_hours, num_features)
                day_ahead_predictions.extend(yhat_sequence)
                # Update history for predicting the next day
                latest_history = history[current_day_hours + j : current_day_hours + j + 24].copy()
                for k in range(24):
                    latest_history[k][dep_var_column] = yhat_sequence[k]
                temp_history = np.concatenate([temp_history, latest_history], axis=0)

            # Update history for predicting the next day
            history = np.concatenate([history, history[current_day_hours : current_day_hours + self.MODEL_SLIDING_WINDOW_LEN]], axis=0)
            predictions.append(day_ahead_predictions)

        # Convert predictions to numpy array
        predicted_data = np.array(predictions, dtype=np.float64)
        return predicted_data


    def get_forecasts(self, model, history, train_window_hours, num_features):
        # Flatten data
        data = np.array(history, dtype=np.float64)
        # Retrieve last observations for input data
        input_x = data[-train_window_hours:]
        # Reshape into [1, n_input, num_features]
        input_x = input_x.reshape((1, len(input_x), num_features))
        # Make predictions
        yhat = model.predict(input_x, verbose=0)
        # Extract the vector forecast
        yhat = yhat[0]
        return yhat, input_x
    
    def calculateCarbonIntensity(self, dataset, carbonRate):
        carbonIntensity = 0
        carbonCol = []
        miniDataset = dataset.iloc[:, self.CARBON_INTENSITY_COLUMN:]
        print("**", miniDataset.columns.values)
        rowSum = miniDataset.sum(axis=1).to_list()
        for i in range(len(miniDataset)):
            if(rowSum[i] == 0):
                # basic algorithm to fill missing values if all sources are missing
                # just using the previous hour's value
                # same as electricityMap
                for j in range(1, len(dataset.columns.values)):
                    if(dataset.iloc[i, j] == 0):
                        dataset.iloc[i, j] = dataset.iloc[i-1, j]
                    miniDataset.iloc[i] = dataset.iloc[i, self.CARBON_INTENSITY_COLUMN:]
                    # print(miniDataset.iloc[i])
                rowSum[i] = rowSum[i-1]
            carbonIntensity = 0
            for j in range(len(miniDataset.columns.values)):
                source = miniDataset.columns.values[j]
                sourceContribFrac = miniDataset.iloc[i, j]/rowSum[i]
                # print(sourceContribFrac, carbonRate[source])
                carbonIntensity += (sourceContribFrac * carbonRate[source])
            if (carbonIntensity == 0):
                print(miniDataset.iloc[i])
            carbonCol.append(round(carbonIntensity, 2)) # rounding to 2 values after decimal place
        dataset.insert(loc=self.CARBON_INTENSITY_COLUMN, column="carbon_intensity", value=carbonCol)
        return dataset

    def inference(self, fuel_sources, inference_timestamp):
        combined_data = pd.DataFrame(columns=["timestamp"])
        for source in fuel_sources:
            IN_FILE_NAME = "data/MW_electricity_cleaned.csv"
            IN_MODEL_NAME = 'model/' + source + "_ann.keras"

            NUM_FEATURES_DICT = {"coal": 6, "nat_gas": 6, "nuclear": 6, "oil": 6, "hydro": 6, "solar": 6,
                                "wind": 6, "others": 6}

            NUM_VAL_DAYS = 30
            NUM_TEST_DAYS = 184
            TRAINING_WINDOW_HOURS = 24
            PREDICTION_WINDOW_HOURS = 24
            MODEL_SLIDING_WINDOW_LEN = 24

            COAL = 1
            NAT_GAS = 2
            NUCLEAR = 3
            OIL = 4
            HYDRO = 5
            SOLAR = 6
            WIND = 7
            OTHERS = 8

            FUEL = {COAL: "coal", NAT_GAS: "nat_gas", NUCLEAR: "nuclear", OIL: "oil", HYDRO: "hydro", SOLAR: "solar", WIND: "wind", OTHERS: "others"}
            SOURCE_TO_SOURCE_COL_MAP = {y: x for x, y in FUEL.items()}

            SOURCE_COL = SOURCE_TO_SOURCE_COL_MAP[source]
            NUM_FEATURES = NUM_FEATURES_DICT[FUEL[SOURCE_COL]]

            print("initializing dataset...")
            
            dataset, dateTime = self.initDataset(IN_FILE_NAME, SOURCE_COL)
            nearest_lower_timestamp = max(filter(lambda x: x <= np.datetime64(inference_timestamp), dateTime))

            # Get data up to last_date and last 24 hours of data
            last_past_date = pd.to_datetime(nearest_lower_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            past = dataset.loc[dataset.index <= last_past_date].tail(24)

            # Get data minimum last_date, max last_date + 24 hours of data
            last_future_date = (pd.to_datetime(nearest_lower_timestamp) + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            future = dataset.loc[dataset.index <= last_future_date].tail(24)

            trainDates = dateTime[: -(NUM_TEST_DAYS*24)]
            fullTrainDates = np.copy(trainDates)
            trainDates, validationDates = trainDates[: -(NUM_VAL_DAYS*24)], trainDates[-(NUM_VAL_DAYS*24):]
            testDates = future.index

            past = past.iloc[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES].values
            future = future.iloc[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES].values

            print("past shape: ", past.shape) # (days x hour) x features
            print("future shape: ", future.shape) # (days x hour) x features

            for i in range(past.shape[0]):
                for j in range(past.shape[1]):
                    if(np.isnan(past[i, j])):
                        past[i, j] = past[i-1, j]

            featureList = dataset.columns.values[SOURCE_COL:SOURCE_COL+NUM_FEATURES]
            print("Features: ", featureList)

            data = []
            with open(f'../GhostPostCC/model/{source}_ann.pkl', 'rb') as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
            
            model = data[0]
            ftMin = data[1]
            ftMax = data[2]

            print("Scaling data...")
            past, future = self.scale_dataset_trained(past, future, ftMin, ftMax)
            print("***** Data scaling done *****")
            print(past.shape, future.shape)

            history = past.tolist()
            predictedData = self.get_day_ahead_forecasts(IN_MODEL_NAME, history, TRAINING_WINDOW_HOURS, NUM_FEATURES, 0)
            formattedTestDates = self.manipulateTestDataShape(testDates, MODEL_SLIDING_WINDOW_LEN, PREDICTION_WINDOW_HOURS, True)
            formattedTestDates = np.reshape(formattedTestDates, formattedTestDates.shape[0]*formattedTestDates.shape[1])

            predictedData = predictedData.astype(np.float64)
            print("PredictedData shape: ", predictedData.shape)
            predicted = np.reshape(predictedData, predictedData.shape[0]*predictedData.shape[1])
            print("predicted.shape: ", predicted.shape)
            unScaledPredictedData = utility.inverseDataScaling(predicted, ftMax[0], ftMin[0])

            df = pd.DataFrame(unScaledPredictedData, columns=[f"avg_{source}_production_forecast"])
            combined_data = pd.concat([combined_data, df], axis=1)
        combined_data["timestamp"] = pd.to_datetime(formattedTestDates).strftime("%Y-%m-%d %H:%M:%S")

        combined_data = self.calculateCarbonIntensity(combined_data, self.carbonRateDirect)

        return combined_data.to_json()