import requests
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection  import train_test_split, GridSearchCV
import cv2
import pickle
from datetime import datetime





# Fetch country coordinates from a CSV file online
def fetch_coordinates():
    url = "https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/dist/countries.csv"
    coords = {}
    # Download and save the CSV if it's not already present
    path = os.path.join('files','countries.csv')
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(requests.get(url).content)
    # Read the downloaded CSV
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            country_name, lat, lon = row[2], float(row[1]), float(row[0])
            coords[country_name] = (lat, lon)
    return coords

def fetch_map_image():
    map_file = os.path.join('files','Equirectangular-projection-topographic-world.jpg')
    map_url = 'https://upload.wikimedia.org/wikipedia/commons/3/3e/Equirectangular-projection-topographic-world.jpg'

    if not os.path.exists(map_file):
        with open(map_file, 'wb') as f:
            f.write(requests.get(map_url).content)

    return cv2.imread(map_file)

# Fetch country descriptions from Wikipedia
def fetch_descriptions(coords):
    desc = {}
    # A mapping for country names that differ on Wikipedia
    name_mapper = {
        'Saba': 'Saba_(island)',
        'Saint Martin': 'Saint_Martin_(island)',
        'Congo': 'Republic_of_the_Congo',
        'Congo DRC': 'Democratic_Republic_of_the_Congo',
        'Canarias': 'Canary_Islands'
    }
    # Iterate through each country and download or read its description
    for country_name in coords:
        file_path = os.path.join('files', f"{country_name}.html")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                desc[country_name] = soup.get_text()
        else:
            url_name = name_mapper.get(country_name, country_name)
            url = f"https://en.wikipedia.org/wiki/{url_name}"
            html = requests.get(url).text
            soup = BeautifulSoup(html, 'html.parser')
            with open(file_path, 'w') as f:
                f.write(soup.get_text())
            desc[country_name] = soup.get_text()
    return desc

# Function for training one model configuration
def train_model_with_params(X_fit, y_fit, model_type='MultiTaskElasticNetCV'):
    # Define the model
    if model_type == 'MultiTaskElasticNetCV':
        model = MultiTaskElasticNetCV(l1_ratio=np.linspace(0, 1, 22)[1:-1], max_iter=50000, cv=20, n_jobs=7)
        model.fit(X_fit.toarray(), y_fit)
    elif model_type == 'SVR':
        param_grid = {
            'estimator__C': np.logspace(-3,3,10),
            'estimator__epsilon': np.logspace(-1,1,3),
            'estimator__gamma': ['auto', 'scale'] + list(np.logspace(-3, 3, 10))
            }
        svr = SVR(kernel='rbf')
        multi_svr = MultiOutputRegressor(svr)
        grid = GridSearchCV(multi_svr, param_grid, refit=True, verbose=3, cv=10, n_jobs=7)
        model = grid.fit(X_fit.toarray(), y_fit)
    elif model_type == 'GPR':
        model = GaussianProcessRegressor(ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBF(1.0, length_scale_bounds=(1e-16,1e2)))
        param_grid = {
            'alpha': np.logspace(-3,3,10),
            }
        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=10, n_jobs=7)
        model = grid.fit(X_fit.toarray(), y_fit)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    date_time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pickle.dump(model, open(f'model_{model_type}_{date_time_string}.sav' , 'wb'))
    return model

def evaluate_model(model, X_holdout, y_holdout, i_holdout, coords):
    prediction = model.predict(X_holdout.toarray())

    # Output CSV format results
    print("Country,Actual_Lat,Actual_Long,Predicted_Lat,Predicted_Long")
    holdout_coords = []
    predicted_coords = []
    country_names = []
    countries = list(coords.keys())
    for j, index in enumerate(i_holdout):
        country = countries[index]
        lat = y_holdout[j][0]
        lon = y_holdout[j][1]
        print(f"{country},{lat},{lon},{prediction[j][0]},{prediction[j][1]}")
        holdout_coords.append((lat, lon))
        predicted_coords.append((prediction[j][0], prediction[j][1]))
        country_names.append(country)

    # Load the background map image
    background_img  = fetch_map_image()
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

    # Plot the map
    plot_map(background_img, holdout_coords, predicted_coords, country_names)


def plot_map(background_img, holdout_coords, predicted_coords, country_names):
    """
    Plot the map using matplotlib.
    :param background_img: The background image (numpy array) to plot.
    :param holdout_coords: Actual coordinates [(lat, lon), ...]
    :param predicted_coords: Predicted coordinates [(lat, lon), ...]
    :param country_names: Names of the countries corresponding to the coordinates
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(18, 9))

    # Display background
    ax.imshow(background_img, extent=[-180, 180, -90, 90])

    # Plot the actual and predicted coordinates
    for (lat, lon), (pred_lat, pred_lon), country in zip(holdout_coords, predicted_coords, country_names):
        ax.scatter(lon, lat, c='blue')  # Actual
        ax.scatter(pred_lon, pred_lat, c='red')  # Predicted
        ax.annotate(f"{country}", (lon, lat), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.annotate(f"{country}", (pred_lon, pred_lat), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.plot([lon, pred_lon], [lat, pred_lat], 'k--')  # Line connecting actual to predicted

    # Customize plot
    ax.set_title("Country Centroid Prediction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save the figure as a PNG file
    plt.savefig("Country_Centroid_Prediction.png")
    print('foo')

# Convert latitude and longitude to x and y for plotting
def lon_to_x(lon):
    return int((lon + 180) * 5.1)

def lat_to_y(lat):
    return int((-lat + 90) * 5.1)

# Main part of the code
if __name__ == "__main__":
    coords = fetch_coordinates()
    desc = fetch_descriptions(coords)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.1)
    descriptions = [desc[country] for country in coords]
    X = vectorizer.fit_transform(descriptions)

    # Latitude and longitude
    lat_longs = np.array([[lat, lon] for lat, lon in coords.values()])

    # Train/test split
    X_fit, X_holdout, y_fit, y_holdout, i_fit, i_holdout = train_test_split(
        X, lat_longs, range(len(lat_longs)), test_size=0.1, random_state=12345)

    # Model training
    model = train_model_with_params(X_fit, y_fit, model_type='MultiTaskElasticNetCV')

    # Model evaluation
    evaluate_model(model, X_holdout, y_holdout, i_holdout, coords)