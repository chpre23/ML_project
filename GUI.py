# -*- coding: utf-8 -*-
# """
#Created on Tue May  7 11:03:45 2024

#@author: CHPRE23
#"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import zipfile
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter.font as tkFont
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from tkinter import messagebox
import pickle
import os
import tensorflow as tf
import holidays
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Input
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from matplotlib.ticker import FuncFormatter, MaxNLocator




# Adaptation of the requests module for HQ website
class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(TLSAdapter, self).proxy_manager_for(*args, **kwargs)
    
    
def generate_dates_list_for_urls(start_date, end_date):
    # Choice depending on the input
    if selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load":
        # Convert the start and end dates to pandas Timestamp objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Adjust the start date to the first day of the month if necessary
        if start_date.day != 1:
            start_date = start_date.replace(day=1)
    
        # Generate a date range of the first days of each month between the start and end dates
        first_days = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Convert the Timestamp objects to strings using the desired date format (e.g., 'YYYY-MM-DD')
        first_days_as_strings = first_days.strftime('%Y%m%d')
    
        # Convert the result to a list and return it
        return list(first_days_as_strings)
    elif selected_data_type.get() == "Moyenne (MW)":
        # Convert the start and end dates to pandas Timestamp objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # Extract the years
        start_year = start_date.year
        end_year = end_date.year
    
        # Generate a list of years from start_year to end_year inclusive
        years_list = list(range(start_year, end_year + 1))
        return years_list



def filter_csv_files(date_list, start_date, end_date):
    # Convert start_date and end_date to the format YYYYMMDD (int)
    start_date_formatted = int(start_date.replace("-", ""))
    end_date_formatted = int(end_date.replace("-", ""))
    
    # Filter the list to only include dates within the start and end date range
    filtered_list = [date for date in date_list if start_date_formatted <= int(date[0:8]) <= end_date_formatted]
    
    return filtered_list

         

def create_url_list_to_download(start_date, end_date):
    urls_list = []
    user_start_date = start_date 
    user_end_date = end_date 
    
    dates_list = generate_dates_list_for_urls(user_start_date, user_end_date)
    
    if selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load":
        for date in dates_list:
            if selected_data_type.get() == "LBMP ($/MWHr)":
                url = "http://mis.nyiso.com/public/csv/damlbmp/" + date + "damlbmp_zone_csv.zip"
                urls_list.append(url)
            elif selected_data_type.get() == "Energy Bid Load":
                url = "http://mis.nyiso.com/public/csv/zonalBidLoad/" + date + "zonalBidLoad_csv.zip"
                urls_list.append(url)
    elif selected_data_type.get() == "Moyenne (MW)":
        for date in dates_list:
            url = "https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/xlsx/historique-demande/" + str(date) + "-demande-electricite-quebec.xlsx"
            urls_list.append(url)    
    return urls_list


            
def combine_zip_files_content(zip_urls):
    # Create a BytesIO object to hold the combined zip archive
    combined_bytes_io = io.BytesIO()
    
    # Create a new ZIP file within the BytesIO object (file-like object)
    with zipfile.ZipFile(combined_bytes_io, 'w') as combined_zip:
        # Iterate over the URLs
        for url in zip_urls:
            # Send an HTTP GET request to download the zip file
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Create a BytesIO object to hold the content of the response
                zip_content = io.BytesIO(response.content)
                
                # Open the zip file using the zipfile module
                with zipfile.ZipFile(zip_content, 'r') as zip_file:
                    # Iterate over the files in the zip file
                    for file_name in zip_file.namelist():
                        # Open each file in the zip file
                        with zip_file.open(file_name) as file:
                            # Read the file content
                            file_content = file.read()
                                                        
                            # Write the file content into the new combined ZIP file, maintaining its file name
                            combined_zip.writestr(file_name, file_content)
            else:
                print(f"Failed to download {url}: {response.status_code}")
    
    # At this point, combined_bytes_io contains a new zip archive with the combined content from all the source zip files
    # Reset the buffer position to the beginning
    combined_bytes_io.seek(0)
    return combined_bytes_io



def format_y_values(y, _):
    """Custom function to format y-values."""
    if y >= 1_000:
        return f'{y / 1_000:.1f}K'  # Format thousands
    else:
        return f'{y:.0f}'  # Keep it as is for lower values
      


def create_figure(dataframe):
    df = dataframe
    column_selection = selected_data_type.get()
    user_location = selected_location.get()
        
    # Create a figure object
    fig, ax = plt.subplots(figsize=(8, 5))
        
    # Ensure the timestamp column is in datetime format
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    
    # Extract the date and hours parts from the timestamp
    df['date'] = df[df.columns[0]].dt.date
    df['hour'] = df[df.columns[0]].dt.hour
    
    # Get unique days
    unique_days = df['date'].unique()

    # Plot data for each day
    for day in unique_days:
        # Filter data for the specific day
        daily_data = df[df['date'] == day]
        
        # Plot the data
        ax.plot(daily_data['hour'], daily_data[column_selection])
    
    # Add labels and title
    ax.set_title("Load data", fontsize=20)
    ax.set_xlim([0, 23])
    #ax.set_ylim([0, df[column_selection].max()])
    
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_values))  # Set custom formatter
    
    ax.set_xlabel('Time / 1 hour', fontsize=20)
    if column_selection == "Energy Bid Load":
        ax.set_ylabel(f'{column_selection} [MW]', fontsize=20, labelpad=0)
    else:
        ax.set_ylabel("Energy Load [MW]", fontsize=20, labelpad=0)
        
    ax.tick_params(axis='both', which='major', labelsize=16)  # Ticks fontsize
    #ax.legend("Load data", fontsize=16)  # Legend fontsize
    
 
    return fig



def generate_GUI_figure(all_zips_data, user_location, start_date, end_date):
    # Initialize an empty dictionary to store the DataFrames
    df_dict = {}
    
    # Create a matplotlib Figure
    fig = Figure(figsize=(6, 4), dpi=100)
    
    # Create an Axes instance
    ax = fig.add_subplot(111)
    
    # Open the ZIP file
    with zipfile.ZipFile(all_zips_data, 'r') as zip_ref:
        # Extract all files in the ZIP file to memory
        zip_files = zip_ref.namelist()  # Get a list of filenames in the ZIP file
        filtered_zip_files = filter_csv_files(zip_files, start_date, end_date)
        
        # Loop through each file in the ZIP archive
        for file_name in filtered_zip_files:
            # Extract the file as a BytesIO object
            with zip_ref.open(file_name) as file:
                # Alternatively, you can use pandas to read the CSV file
                df = pd.read_csv(file)
                location_column_index = df.columns.get_loc("Name")
                mask = df.iloc[:, location_column_index] == user_location

                # Filter the DataFrame using the mask
                filtered_df = df[mask]
                
                # Add the DataFrame to the dictionary with a title as the key
                title = file_name
                df_dict[title] = filtered_df
                
                x_data = filtered_df.iloc[:, 0].str[10:16]
                column_selection = selected_data_type.get()
                y_data = filtered_df[column_selection]

                ax.plot(x_data, y_data)
                
                # Add labels and title
                ax.set_xlim([0, 23])
                ax.set_xticks([0, 4, 8, 12, 16, 20])
                ax.set_xlabel('Time')
                ax.set_ylabel(column_selection)
    
    return fig



def plot_button_function(): 
    #user_location = selected_location.get() #'CAPITL'
    user_start_date = LineEdit_01.get() #'2023-05-02'
    user_end_date = LineEdit_02.get() #'2023-07-03
    
    # Validate user inputs
    #print('hello')
    if not validate_input_dates(user_start_date, user_end_date):
        return
    
    if selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load":
        df = create_dataframe_from_zip_files_NYISO(user_start_date, user_end_date)
        
    elif selected_data_type.get() == "Moyenne (MW)":
        df = create_dataframe_from_xlsx_HQ(user_start_date, user_end_date)
    
    GUI_figure = create_figure(df)
    
    # Create a canvas for the figure
    canvas = FigureCanvasTkAgg(GUI_figure, master=root)
    canvas.draw()
    
    # Place the canvas widget in the grid layout
    # Adjust row and column to display each plot as desired
    canvas.get_tk_widget().place(relx=0.3, rely=0.725, anchor=tk.CENTER, width=647.5, height=447.5)



def create_dataframe_from_xlsx_HQ(user_start_date, user_end_date):
    # Create the list containing the urls to download
    urls_to_download_list = create_url_list_to_download(user_start_date, user_end_date)
    # Create an empty list to hold all DataFrames
    all_dataframes = []

    session = requests.Session()
    session.mount('https://', TLSAdapter())
    
    for url in urls_to_download_list:
        # Send an HTTP GET request to download the zip file
        response = session.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Use BytesIO to handle the binary content of the file
            file_content = io.BytesIO(response.content)
            
            # Read the Excel file into a DataFrame
            df = pd.read_excel(file_content)#, engine='openpyxl')
            # Convert the "Date" column to datetime format
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

            # Filter the DataFrame
            start_date = pd.to_datetime(user_start_date).date()
            end_date = pd.to_datetime(user_end_date).date()
            filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
            # Append the DataFrame to the list
            all_dataframes.append(filtered_df)
            
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df



def create_dataframe_from_zip_files_NYISO(user_start_date, user_end_date):
    user_location = selected_location.get() # GLineEdit_933.get() #'CAPITL'
    # Create a list containing all urls
    urls_to_download_list = create_url_list_to_download(user_start_date, user_end_date)
    
    # Create an empty list to hold all DataFrames
    all_dataframes = []
    
    # Iterate over the URLs in urls_to_download_list
    for url in urls_to_download_list:
        # Send an HTTP GET request to download the zip file
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Create a BytesIO object to hold the content of the response
            zip_content = io.BytesIO(response.content)
            
            # Open the zip file using the zipfile module
            with zipfile.ZipFile(zip_content, 'r') as zip_file:
                # Filter the CSV files with user input start and end dates
                csv_file_list = filter_csv_files(zip_file.namelist(), user_start_date, user_end_date)
                # Iterate over the filtered CSV files
                for file_name in csv_file_list :# zip_file.namelist():
                    if file_name.endswith('.csv'):  # Ensure it is a CSV file
                        # Open the CSV file
                        with zip_file.open(file_name) as file:
                            # Read the CSV file into a pandas DataFrame
                            df = pd.read_csv(file)
                            # Append the DataFrame to the list
                            all_dataframes.append(df)
                            
        else:
            print(f"Failed to download {url}: {response.status_code}")
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Filter dataframe with user input location
    location_column_index = combined_df.columns.get_loc("Name")
    mask = combined_df.iloc[:, location_column_index] == user_location
    unique_location_df = combined_df[mask]
    
    return unique_location_df



def download_button_function():  

    user_start_date = LineEdit_01.get() #'2023-05-02'
    user_end_date = LineEdit_02.get() #'2023-07-03
    
    # Validate user inputs
    if not validate_input_dates(user_start_date, user_end_date):
        return
    
    # Open a file save dialog to choose where to save the file
    save_path = tk.filedialog.asksaveasfilename(
    title="Save file as",
    defaultextension=".csv",  # Default file extension (change as needed)
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    # Save and handle errors
    if save_path:
        if selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load":
            try:
                data_df = create_dataframe_from_zip_files_NYISO(user_start_date, user_end_date)
                data_df.to_csv(save_path, index=False)
                print(f"file saved to: {save_path}")
            except Exception as e:
                print(f"Error while saving DataFrame: {e}")
        elif selected_data_type.get() == "Moyenne (MW)":
            try:
                data_df = create_dataframe_from_xlsx_HQ(user_start_date, user_end_date)
                data_df.to_csv(save_path, index=False)
                print(f"file saved to: {save_path}")
            except Exception as e:
                print(f"Error while saving DataFrame: {e}")



def get_locations():
    # Get today's date
    date = datetime.today().strftime('%Y%m%d')
    
    # The URL that contains the directory listing
    if selected_data_type.get() == "LBMP ($/MWHr)":
        base_url = "http://mis.nyiso.com/public/csv/damlbmp/" + date + "damlbmp_zone.csv"
    elif selected_data_type.get() == "Energy Bid Load":
        base_url = "http://mis.nyiso.com/public/csv/zonalBidLoad/" + date + "zonalBidLoad.csv"
    
    # Send an HTTP GET request to download the CSV file
    response = requests.get(base_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Read the CSV file using pandas
        df = pd.read_csv(io.StringIO(response.text))
    else:
        print(f"Failed to download {base_url}: {response.status_code}")
        
    # Use the .text for results
    location_options = df["Name"].unique() 
    dropdown_menu_locations = tk.OptionMenu(root, selected_location, *location_options)#, command=option_selected)
    dropdown_menu_locations["bg"] ="#e9e9ed"
    dropdown_menu_locations["font"] = ft
    dropdown_menu_locations["fg"] = "#000000"
    dropdown_menu_locations["justify"] = "center"
    #dropdown_menu_locations.place(relx=0.3, rely=0.21, anchor=tk.CENTER, width=150, height=30)
    
    return location_options
    


def validate_input_dates(start_date, end_date):
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        messagebox.showerror("Error", "Invalid format for start date or end date")
        return False
    if start_date > end_date:
        messagebox.showerror("Error", "Invalid dates : start date must be before end date")
        return False
    elif datetime.strptime(end_date, "%Y-%m-%d") > datetime.today():
        messagebox.showerror("Error", "Invalid dates : end date must be before today")
        return False
    elif (selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load") and start_date < "2001-06-01":
        messagebox.showerror("Error", "Invalid dates : start date must be > 2001-06-01 for NYISO data")   
        return False
    elif (selected_data_type.get() == "Moyenne (MW)") and datetime.strptime(start_date, '%Y-%m-%d').year >= datetime.today().year:
        messagebox.showerror("Error", "Invalid dates : current year data is not available for HQ data")
        return False
    elif (selected_data_type.get() == "Moyenne (MW)") and start_date < "2019-01-01":
        messagebox.showerror("Error", "Invalid dates : start date must be > 2019-01-01 for HQ data")
        return False
    elif (selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load") and selected_location.get() == "Select a location":
        messagebox.showerror("Error", "Invalid location")
        return False
    else:
        return True



def create_future_dataframe(start_date, end_date):
    # Define the features and target variable
    FEATURES = [
        "is_holiday",
        "Hour",
        "day_of_month",
        "day_of_week",
        "day_of_year",
        "quarter",
        "month",
        "week_of_year"
    ]
    
    # Generate the date range with hourly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create the DataFrame with the date range as the index
    df_future_dates = pd.DataFrame(index=date_range)
    df_future_data = create_features(df_future_dates)
    
    # Prepare the future data with the same features
    X_future = df_future_data[FEATURES]
    return X_future



def predict_button_function():
    user_start_date = LineEdit_01.get() #'2023-05-02'
    user_end_date = LineEdit_02.get() #'2023-07-03
    
    try:
        datetime.strptime(user_start_date, '%Y-%m-%d')
        datetime.strptime(user_end_date, '%Y-%m-%d')
    except ValueError:
        messagebox.showerror("Error", "Invalid format for start date or end date")
        return False
    if selected_data_type.get() == "LBMP ($/MWHr)":
        messagebox.showerror("Error", "Cannot makes predictions with LBMP")
    elif user_start_date > user_end_date:
        messagebox.showerror("Error", "Invalid dates : start date must be before end date")
        return False
    elif datetime.strptime(user_start_date, "%Y-%m-%d") < datetime.today():
        messagebox.showerror("Error", "Invalid dates : start date must be today or after")
        return False
    elif (selected_data_type.get() == "LBMP ($/MWHr)" or selected_data_type.get() == "Energy Bid Load") and selected_location.get() == "Select a location":
        messagebox.showerror("Error", "Invalid location")
        return False
    
    if selected_data_type.get() == "Energy Bid Load":
        model, feature_scaler, target_scaler = load_trained_model()
        
        if model is None:
            model, feature_scaler, target_scaler = train_model()
        
        # Create new dataframe with future data  
        X_future = create_future_dataframe(user_start_date, user_end_date)
        
        # Scale the future data using the same scaler
        X_future_scaled = feature_scaler.transform(X_future)
        
        # Reshape the future data for LSTM
        X_future_reshaped = X_future_scaled.reshape(X_future_scaled.shape[0], 1, X_future_scaled.shape[1])
        
        # Make predictions
        scaled_predictions = model.predict(X_future_reshaped)
        # Inverse transform the predictions to get the actual values
        pred_lstm = target_scaler.inverse_transform(scaled_predictions)
        
        # Create final dataframe to plot
        # Create a new dataframe with the same index as original_df
        result_frame = pd.DataFrame(index=X_future.index)
        result_frame["pred_lstm"] = pred_lstm
        
        # Plot final results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_frame.index, result_frame["pred_lstm"], "o")
        ax.set_title("Predictions", fontsize=20)
        ax.set_ylabel(f"{selected_data_type.get()} [MW]", fontsize=20, labelpad=0)
        ax.set_xlabel("Time / 1 hour", fontsize=20);
        
        ax.tick_params(axis='both', which='major', labelsize=16)  # Ticks fontsize
        ax.yaxis.set_major_formatter(FuncFormatter(format_y_values))  # Set custom formatter
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        #ax.legend("Predictions", fontsize=16)  # Legend fontsize
        # plt.tight_layout()  # Adjust layout to make room for the rotated labels
        
        # Create a canvas for the figure
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        
        # Place the canvas widget in the grid layout
        canvas.get_tk_widget().place(relx=0.7, rely=0.725, anchor=tk.CENTER, width=647.5, height=447.5)
    else:
        df= pd.read_csv("HQ_LOAD_hourly_2019_2023.csv", header=0, index_col=0, parse_dates=True)
        
        # Remove duplicates
        df=df.groupby(level=0).mean()
         
        new_date_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df=df.reindex(new_date_range)
        df[selected_data_type.get()] = df[selected_data_type.get()].interpolate(method='linear')
        
        # Create random forest model
        ran_model=RandomForestRegressor(n_estimators=100,max_features=3, random_state=1)
        
        # Generate data to train the model
        df2 = create_features(df)
        
        # Reorganise dataframe for future processing
        # Specify the column to move
        column_to_move = 'Moyenne (MW)'
        # Get a list of columns excluding the one to move
        cols = [col for col in df2.columns if col != column_to_move]
        # Append the column to move to the end of the list
        cols.append(column_to_move)
        # Reorder the DataFrame columns
        df2 = df2[cols]
        # Remove the date column
        df2 = df2.drop(['Date'], axis=1)

        X_train = df2.iloc[:, :-1]
        
        y_train = df2.iloc[:, -1]
        
        # Convert to NumPy arrays
        X_train = X_train.values
        y_train = y_train.values
        
        # Fit model
        ran_model.fit(X_train,y_train)
        
        # Create dataframe with future indexes
        future_df = create_future_dataframe(user_start_date, user_end_date)

        X_test = future_df.values
        
        # Use model to make predictions
        pred_ran=ran_model.predict(X_test)
        
        # Create final dataframe to plot
        # Create a new dataframe with the same index as original_df
        result_frame = pd.DataFrame(index=future_df.index)
        result_frame["pred_ran"] = pred_ran
        
        # Plot final results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_frame.index, result_frame["pred_ran"], "o")
        ax.set_title("Predictions", fontsize=20)
        ax.set_ylabel("Energy Load [MW]", fontsize=20, labelpad=0)
        ax.set_xlabel("Time / 1 hour", fontsize=20)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(format_y_values))  # Set custom formatter
        ax.tick_params(axis='both', which='major', labelsize=16)  # Ticks fontsize
        #plt.tight_layout()  # Adjust layout to make room for the rotated labels
        
        # Create a canvas for the figure
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        
        # Place the canvas widget in the grid layout
        canvas.get_tk_widget().place(relx=0.7, rely=0.725, anchor=tk.CENTER, width=647.5, height=447.5)


        
def create_features(df):
    """
    Create time series features based on time series index

    Args:
        - df: time series dataframe

    Returns:
        - df: time series dataframe with new features
    """
    us_holidays = holidays.US()
    
    df = df.copy()
    df["Date"] = df.index.date
    df["Hour"] = df.index.hour
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Extract unique dates from the "Date" column
    unique_dates = df['Date'].unique()
    # Create a dictionary to map each unique date to 1 if it's a holiday, 0 otherwise
    holiday_dict = {date: 1 if date in us_holidays else 0 for date in unique_dates}   
    # Map the dictionary to the original DataFrame to create the "holidays" column
    df['is_holiday'] = df['Date'].map(holiday_dict)
    
    return df



def train_model():
    # Read the input dataset
    df= pd.read_csv("NYISO_LOAD_hourly_2012_2023.csv", header=0, index_col=0, parse_dates=True)
    
    # Filter dataframe with user input location
    location_column_index = df.columns.get_loc("Name")
    mask = df.iloc[:, location_column_index] == selected_location.get()
    df = df[mask]
    
    df=df[selected_data_type.get()].to_frame()
    
    df=df.groupby(level=0).mean()
    
    new_date_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df=df.reindex(new_date_range)
    df[selected_data_type.get()] = df[selected_data_type.get()].interpolate(method='linear')
    
    train_data = create_features(df)
    
    # Define the features and target variable
    FEATURES = [
        "is_holiday",
        "Hour",
        "day_of_month",
        "day_of_week",
        "day_of_year",
        "quarter",
        "month",
        "week_of_year"
    ]
    TARGET = selected_data_type.get()
    
    # Split the data into features and target
    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    
    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = feature_scaler.fit_transform(X_train)
    
    # Scale the target variable
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_keras = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    
    # Reshape the features for LSTM
    X_train_keras = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    
    # Create and compile neural network
    model = Sequential()
    model.add(Input(shape=(X_train_keras.shape[1], X_train_keras.shape[2])))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(32))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.compile(loss = custom_loss, optimizer="adam")
    
    # Fit model
    model.fit(
        X_train_keras,
        y_train_keras,
        epochs=10,
        batch_size=32)
    
    # Save the model and scalers
    model.save(f"model_{selected_location.get()}.keras")
    with open(f"feature_scaler_{selected_location.get()}.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(f"target_scaler_{selected_location.get()}.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    
    return model, feature_scaler, target_scaler



def load_trained_model():
    model_path = f"model_{selected_location.get()}.keras"
    feature_scaler_path = f"feature_scaler_{selected_location.get()}.pkl"
    target_scaler_path = f"target_scaler_{selected_location.get()}.pkl"
    
    if os.path.exists(model_path) and os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
        # Load the model
        model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
        
        # Load the scalers
        with open(feature_scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
        
        return model, feature_scaler, target_scaler
    else:
        return None, None, None



@keras.saving.register_keras_serializable()
def custom_loss(y_true, y_pred):
    # Calculate the basic mean squared error
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # Calculate the underestimation penalty
    underestimation_penalty = tf.square(tf.maximum(0., y_true - y_pred))
    # Amplify the underestimation penalty
    amplified_underestimation_penalty = 10 * underestimation_penalty
    # Combine the two parts
    return mse + amplified_underestimation_penalty





# Create the main window
root = tk.Tk()
root.title("New York ISO and HQ")
#setting window size
width=750
height=750
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
root.geometry(alignstr)
root.resizable(width=True, height=True)

### FRAMES
# Create frame 1
frame_NYISO = tk.Frame(root, bd=1, relief=tk.SOLID)
frame_NYISO.place(relx=0.3, rely=0.15, anchor=tk.CENTER, width=300, height=250)

# Create frame 2
frame_HQ = tk.Frame(root, bd=1, relief=tk.SOLID)
frame_HQ.place(relx=0.7, rely=0.15, anchor=tk.CENTER, width=300, height=250)

# Create frame 3
frame_data = tk.Frame(root, bd=1, relief=tk.SOLID)
frame_data.place(relx=0.3, rely=0.725, anchor=tk.CENTER, width=650, height=450)

# Create frame 4
frame_forecast = tk.Frame(root, bd=1, relief=tk.SOLID)
frame_forecast.place(relx=0.7, rely=0.725, anchor=tk.CENTER, width=650, height=450)


### BUTTONS
# Plot button
GButton_01=tk.Button(root)
GButton_01["bg"] = "#e9e9ed"
ft = tkFont.Font(family='Times',size=20)
GButton_01["font"] = ft
GButton_01["fg"] = "#000000"
GButton_01["justify"] = "center"
GButton_01["text"] = "Plot"
GButton_01.place(relx=0.4, rely=0.45, anchor=tk.CENTER, width=150, height=50)
GButton_01["command"] = plot_button_function

# Download button
GButton_02=tk.Button(root)
GButton_02["bg"] = "#e9e9ed"
ft = tkFont.Font(family='Times',size=20) 
GButton_02["font"] = ft
GButton_02["fg"] = "#000000"
GButton_02["justify"] = "center"
GButton_02["text"] = "Download"
GButton_02.place(relx=0.5, rely=0.45, anchor=tk.CENTER, width=150, height=50)
GButton_02["command"] = download_button_function

# Predict button
GButton_03=tk.Button(root)
GButton_03["bg"] = "#e9e9ed"
ft = tkFont.Font(family='Times',size=20) 
GButton_03["font"] = ft
GButton_03["fg"] = "#000000"
GButton_03["justify"] = "center"
GButton_03["text"] = "Predict"
GButton_03.place(relx=0.6, rely=0.45, anchor=tk.CENTER, width=150, height=50)
GButton_03["command"] = predict_button_function


### LABELS
# New York ISO label
Label_01=tk.Label(root)
ft = tkFont.Font(family='Times',size=24, weight='bold')
Label_01["font"] = ft
Label_01["fg"] = "#333333"
Label_01["justify"] = "center"
Label_01["text"] = "New York ISO"
Label_01.place(relx=0.3,rely=0.08, anchor=tk.CENTER, width=225,height=50)

# Data type label 1
Label_02=tk.Label(root)
ft = tkFont.Font(family='Times',size=20)
Label_02["font"] = ft
Label_02["fg"] = "#333333"
Label_02["justify"] = "center"
Label_02["text"] = "Data type"
Label_02.place(relx=0.3, rely=0.12, anchor=tk.CENTER, width=150, height=50)

# start date label
Label_03=tk.Label(root)
ft = tkFont.Font(family='Times',size=20)
Label_03["font"] = ft
Label_03["fg"] = "#333333"
Label_03["justify"] = "center"
Label_03["text"] = "Start date (YYYY-MM-DD) :"
Label_03.place(relx=0.4, rely=0.325, anchor=tk.CENTER, width=350, height=50)

# end date label
Label_04=tk.Label(root)
ft = tkFont.Font(family='Times',size=20)
Label_04["font"] = ft
Label_04["fg"] = "#333333"
Label_04["justify"] = "center"
Label_04["text"] = "End date (YYYY-MM-DD) :"
Label_04.place(relx=0.4, rely=0.375, anchor=tk.CENTER, width=350, height=50)

# Data type label 2
Label_06=tk.Label(root)
ft = tkFont.Font(family='Times',size=20)
Label_06["font"] = ft
Label_06["fg"] = "#333333"
Label_06["justify"] = "center"
Label_06["text"] = "Data type"
Label_06.place(relx=0.7, rely=0.12, anchor=tk.CENTER, width=150, height=50)

# HQ label
Label_07=tk.Label(root)
ft = tkFont.Font(family='Times',size=24, weight='bold')
Label_07["font"] = ft
Label_07["fg"] = "#333333"
Label_07["justify"] = "center"
Label_07["text"] = "Hydro Quebec"
Label_07.place(relx=0.7, rely=0.08, anchor=tk.CENTER, width=200 ,height=50)

# Location label
Label_08=tk.Label(root)
ft = tkFont.Font(family='Times',size=20)
Label_08["font"] = ft
Label_08["fg"] = "#333333"
Label_08["justify"] = "center"
Label_08["text"] = "Location"
Label_08.place(relx=0.3, rely=0.18, anchor=tk.CENTER, width=150 ,height=50)


### EDIT FIELDS
# Start date edit field
LineEdit_01=tk.Entry(root)
LineEdit_01["borderwidth"] = "1px"
ft = tkFont.Font(family='Times',size=20)
LineEdit_01["font"] = ft
LineEdit_01["fg"] = "#333333"
LineEdit_01["justify"] = "center"
LineEdit_01.place(relx=0.6, rely=0.325, anchor=tk.CENTER, width=250, height=50)

# End date edit field
LineEdit_02=tk.Entry(root)
LineEdit_02["borderwidth"] = "1px"
ft = tkFont.Font(family='Times',size=20)
LineEdit_02["font"] = ft
LineEdit_02["fg"] = "#333333"
LineEdit_02["justify"] = "center"
LineEdit_02.place(relx=0.6, rely=0.375, anchor=tk.CENTER, width=250, height=50)


### DROPDOWN MENU
# Create a StringVar to hold the current selection
selected_data_type = tk.StringVar()
selected_data_type.set("LBMP ($/MWHr)")  # Set default selection

# Define the options for the dropdown menu
selected_location = tk.StringVar(root)
location_options = get_locations()

# Create a StringVar to hold the current selection
selected_location.set("Select a location")  # Set default selection
ft = tkFont.Font(family='Times', size=20)

# Create an OptionMenu widget (dropdown menu)  
dropdown_menu_locations = tk.OptionMenu(root, selected_location, *location_options)#, command=option_selected)
dropdown_menu_locations["bg"] ="#e9e9ed"
dropdown_menu_locations["font"] = ft
dropdown_menu_locations["fg"] = "#000000"
dropdown_menu_locations["justify"] = "center"
dropdown_menu_locations.place(relx=0.3, rely=0.225, anchor=tk.CENTER, width=250, height=50)


### RADIO BUTTONS
# Create the first radio button
RButton_01 = tk.Radiobutton(root, text="LBMP", variable=selected_data_type, value="LBMP ($/MWHr)")
ft1 = tkFont.Font(family='Times', size=20)
RButton_01["font"] = ft1
RButton_01["fg"] = "#333333"
RButton_01["justify"] = "center"
RButton_01["command"] = get_locations
RButton_01.place(relx=0.27, rely=0.15, anchor=tk.CENTER, width=125, height=25)

# Create the second radio button
RButton_02 = tk.Radiobutton(root, text="LOAD", variable=selected_data_type, value="Energy Bid Load")
ft2 = tkFont.Font(family='Times', size=20)
RButton_02["font"] = ft2
RButton_02["fg"] = "#333333"
RButton_02["justify"] = "center"
RButton_02.place(relx=0.33, rely=0.15, anchor=tk.CENTER, width=125, height=25)
RButton_02["command"] = get_locations

# Create the third radio button (HQ)
RButton_03 = tk.Radiobutton(root, text="LOAD", variable=selected_data_type, value="Moyenne (MW)")
ft2 = tkFont.Font(family='Times', size=20)
RButton_03["font"] = ft2
RButton_03["fg"] = "#333333"
RButton_03["justify"] = "center"
RButton_03.place(relx=0.7, rely=0.15, anchor=tk.CENTER, width=150, height=25)


root.mainloop()




