# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name:
# Collaborators:
# Time:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
from sklearn.cluster import KMeans

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

# KMeans class not required until Problem 7
class KMeansClustering(KMeans):

    def __init__(self, data, k):
        super().__init__(n_clusters=k, random_state=0)
        self.fit(data)
        self.labels = self.predict(data)

    def get_centroids(self):
        'return np array of shape (n_clusters, n_features) representing the cluster centers'
        return self.cluster_centers_

    def get_labels(self):
        'Predict the closest cluster each sample in data belongs to. returns an np array of shape (samples,)'
        return self.labels

    def total_inertia(self):
        'returns the total inertia of all clusters, rounded to 4 decimal points'
        return round(self.inertia_, 4)



class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        results = []
        for year in years:
            vals = []
            # for every city
            for city in cities:
                vals += list(self.get_daily_temps(city, year))
            results.append(sum(vals)/len(vals))

        return results
def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    x_bar = sum(x)/len(x)
    y_bar = sum(y)/len(y)

    m_numerator = 0
    m_denominator = 0

    for i in range(len(x)):
        m_numerator += (x[i] - x_bar) * (y[i] - y_bar)
        m_denominator += (x[i] - x_bar)**2
    m = m_numerator/m_denominator

    return (m, y_bar - m*x_bar)


def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    squared_e = 0
    for i in range(len(x)):
        squared_e += (y[i] - (m*x[i] + b))**2
    return squared_e


def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models = []
    for degree in degrees:
        models.append(np.polyfit(x, y, degree))
    
    return models


def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    r_squared_values = []
    if display_graphs:
            
        for model in models:
            # get the polynomial
            p = np.poly1d(model)
            # get the predicted values
            y_pred = [ np.polyval(p, x_i) for x_i in x]
            r_squared = r2_score(y, y_pred)
            title = "Degree " + str( len(model) - 1 ) + " with " + str(round(r_squared, 4)) + " R^2 value"
            # if model is linear...
            if len(model) <= 2:
                # seos is standard error over slope
                seos = round((np.std(y)/len(y)**0.5) / model[0], 4)
                title += " and " + str(seos) + " slope test value"
            
            r_squared_values.append(r_squared)

            plt.scatter(x, y, s = 1)
            plt.plot(x, y_pred, label = "Predicted Values")
            plt.legend()
            plt.ylabel("Temperature in Degrees Celsius")
            plt.xlabel("Year")
            plt.title(title)
            plt.show()
    else:
        for model in models:
            # get the polynomial
            p = np.poly1d(model)
            # get the predicted values
            y_pred = [ np.polyval(p, x_i) for x_i in x]
            r_squared_values.append(r2_score(y, y_pred))

    return r_squared_values


def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    Max, start, end, m, b = [0]*5
    if length > len(x):
        return None
    for i in range(length-1, len(x)):
        m, b = linear_regression(x[i - (length - 1): i + 1 ], y[i - (length - 1): i + 1])

        # if looking for positive slope...
        if positive_slope:
            # if greater positive slope has been found...
            if m > Max:
                Max = m
                start, end = i - (length - 1), i + 1
        else:
            # finding most extreme negative slope
            if m < Max:
                Max = m
                start, end = i - (length - 1), i + 1
    # if no interval was found, return None
    if start == end:
        return None
    return (start, end, Max)


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    if len(x) < 2:
        return []
    results = []
    for length in range(2, len(x) + 1):
        p_tuple = get_max_trend(x, y, length, True)
        n_tuple = get_max_trend(x, y, length, False)
        result = 0
        if p_tuple == None and n_tuple != None:
            result = n_tuple
        elif n_tuple == None and p_tuple != None:
            result = p_tuple
        elif n_tuple == None and p_tuple == None:
            result = (0, length, None)
        elif p_tuple[2] > abs(n_tuple[2]):
            result = p_tuple
        else:
            result = n_tuple
        results.append(result)
    return results



def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    Sum = 0
    for i in range(len(y)):
        Sum += (y[i] - estimated[i])**2

    return (Sum/len(y) )**0.5



def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    rmse_values = []
    if display_graphs:
            
        for model in models:
            # get the polynomial
            p = np.poly1d(model)
            # get the predicted values
            y_pred = [ np.polyval(p, x_i) for x_i in x]
            rmse = calculate_rmse(y, y_pred)
            title = "Degree " + str( len(model) - 1 ) + " with " + str(round(rmse, 4)) + " RMSE value"
            
            rmse_values.append(rmse)

            plt.scatter(x, y, s = 1)
            plt.plot(x, y_pred, label = "Predicted Values")
            plt.legend()
            plt.ylabel("Temperature in Degrees Celsius")
            plt.xlabel("Year")
            plt.title(title)
            plt.show()
    else:
        for model in models:
            # get the polynomial
            p = np.poly1d(model)
            # get the predicted values
            y_pred = [ np.polyval(p, x_i) for x_i in x]
            rmse_values.append(calculate_rmse(y, y_pred))

    return rmse_values


def cluster_cities(cities, years, data, n_clusters):
    '''
    Clusters cities into n_clusters clusters using their average daily temperatures
    across all years in years. Generates a line plot with the average daily temperatures
    for each city. Each cluster of cities should have a different color for its
    respective plots.

    Args:
        cities: a list of the names of cities to include in the average
                daily temperature calculations
        years: a list of years to include in the average daily
                temperature calculations
        data: a Dataset instance
        n_clusters: an int representing the number of clusters to use for k-means

    Note that this part has no test cases, but you will be expected to show and explain
    your plots during your checkoff
    '''
    vectors = []
    for city in cities:
        vector = np.array([0.0]*365)
        for year in years:
            # sum all the daily temps for the current city
            vector += np.array(data.get_daily_temps(city, year)[:365])
        # divide all daily temps by amount of years to get the average daily temp
        vector = vector/365
        vectors.append(vector)
    # calculate kmeans
    kmeans = KMeans(n_clusters).fit(vectors)
    # dictionary where label is key and color is value
    colors = {0:"#228B22", 1:"#0000FF", 2:"#FF0000", 3:"#FFA500"}
    for i in range(len(cities)):
        plt.plot(list(range(1, 366)), vectors[i], label = cities[i], color = colors[kmeans.labels_[i]])
    # plot the feature vectors
    plt.legend()
    plt.ylabel("Temperature in Degrees Celsius")
    plt.xlabel("Days of the Years, 1-365")
    plt.title("Clustering Climates")
    plt.show()


if __name__ == '__main__':
    ##################################################################################
    # # Problem 4A: DAILY TEMPERATURE
    dataset = Dataset("data.csv")
    x_vals = np.array(range(1961, 2017))
    # y_vals = []
    # for x in x_vals:
    #     y_vals.append(dataset.get_temp_on_date("BOSTON", 12, 1, x))
    # y_vals = np.array(y_vals)
    # # model it to one degree polynomial
    # models = generate_polynomial_models(x_vals, y_vals, [1])
    # # plot it
    # r_squared_values = evaluate_models(x_vals, y_vals, models, True)

    # ##################################################################################
    # # Problem 4B: ANNUAL TEMPERATURE
    # mean_annual_temperatures = dataset.calculate_annual_temp_averages(["BOSTON"], x_vals)
    # models = generate_polynomial_models(x_vals, mean_annual_temperatures, [1])
    # r_squared_values = evaluate_models(x_vals, mean_annual_temperatures, models, True)

    # ##################################################################################
    # # Problem 5B: INCREASING TRENDS
    # mean_annual_temperatures = dataset.calculate_annual_temp_averages(["SEATTLE"], x_vals)
    # start, end, slope = get_max_trend(x_vals, mean_annual_temperatures, 30, True)
    # models = generate_polynomial_models(x_vals[start:end], mean_annual_temperatures[start:end], [1])
    # # plot it
    # r_squared_values = evaluate_models(x_vals[start:end], mean_annual_temperatures[start:end], models, True)
    # ##################################################################################
    # # Problem 5C: DECREASING TRENDS
    # mean_annual_temperatures = dataset.calculate_annual_temp_averages(["SEATTLE"], x_vals)
    # start, end, slope = get_max_trend(x_vals, mean_annual_temperatures, 15, False)
    # models = generate_polynomial_models(x_vals[start:end], mean_annual_temperatures[start:end], [1])
    # # plot it
    # r_squared_values = evaluate_models(x_vals[start:end], mean_annual_temperatures[start:end], models, True)

    # ##################################################################################
    # # Problem 5D: ALL EXTREME TRENDS
    # # Your code should pass test_get_max_trend. No written answer for this part, but
    # # be prepared to explain in checkoff what the max trend represents.

    # ##################################################################################
    # # Problem 6B: PREDICTING
    # mean_annual_temperatures = dataset.calculate_annual_temp_averages(CITIES, TRAIN_INTERVAL)
    # models = generate_polynomial_models(TRAIN_INTERVAL, mean_annual_temperatures, [2, 10])
    # # plot it
    # r_squared_values = evaluate_models(TRAIN_INTERVAL, mean_annual_temperatures, models, True)

    # mean_annual_temperatures = dataset.calculate_annual_temp_averages(CITIES, TEST_INTERVAL)
    # models = generate_polynomial_models(TEST_INTERVAL, mean_annual_temperatures, [2, 10])
    # # plot it
    # r_squared_values = evaluate_models(TEST_INTERVAL, mean_annual_temperatures, models, True)

    ##################################################################################
    # Problem 7: KMEANS CLUSTERING (Checkoff Question Only)
    cluster_cities(CITIES, x_vals, dataset, 4)

    ####################################################################################