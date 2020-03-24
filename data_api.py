"""Data module"""
import re
from datetime import datetime
from itertools import tee
import pandas as pd
import numpy as np
import xlrd


class Country:
    """Country data"""
    def __init__(self, name: str, population: int, days: np.array,
                 confirmed: np.array, deaths: np.array):
        self.name = name
        self.population = population
        self.days = days
        self.confirmed = confirmed
        self.deaths = deaths

    def days_to_ind(self):
        """Generates an array of indices from the country's reported days"""
        day_diff = self.days - self.days[0]
        return np.array([int(x.days) for x in day_diff])

    def _delim_str(self):
        return "-" * len(self.name)

    def _time_interval(self, format_="%y-%m-%d"):
        start = self.days[0].strftime(format_)
        end = self.days[-1].strftime(format_)
        return "{} - {}".format(start, end)

    def __str__(self):
        return "{}\n{}\n\tPopulation: {}\n\tInterval: {}\n{}".format(
            self._delim_str(), self.name.upper(), self.population,
            self._time_interval(), self._delim_str())


def country_data(country_name: str):
    """Get country datastructure from external sources"""
    raw_data = _get_data(CONFIRMED_URL)
    return _country_from_df(country_name, raw_data)


CONFIRMED_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

POPULATION_FILENAME = "PopulationByCountry.xlsx"


def list_countries(full_data):
    """List all countries in a dataframe.
    Nice to have to see the proper ID spelling
    """
    # List/set shenanigans give unique entries
    countries = list(set(full_data["Country/Region"]))
    return sorted(countries)


def read_population_file(filename):
    """Returns dictionary on the form {'Sweden' : int(population size)}"""
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    dictionary = {}
    for row in range(1, worksheet.nrows):
        dictionary[worksheet.cell_value(row,
                                        0)] = int(worksheet.cell_value(row, 1))
    return dictionary


def _country_from_df(country_name: str, df: pd.DataFrame):
    """Populate country datastruct from raw pandas df

    Currently sums all provinces in a country.
    """
    day_columns = _get_days(df)
    str_dates, to_datetime = tee(day_columns)

    population = read_population_file(POPULATION_FILENAME)[country_name]
    days = np.array([_parse_date(x) for x in to_datetime])

    provinces = df[df["Country/Region"] == country_name]
    confirmed = provinces[np.array(list(str_dates))].sum()

    return Country(country_name, population, days, confirmed, None)


def _get_data(url: str):
    """Main entry point"""
    return pd.read_csv(url)


def _get_days(df):
    """Return array of dates from df columns"""
    return filter(_is_date, df.columns)


def _parse_date(date: str):
    return datetime.strptime(date, "%m/%d/%y").date()


DATE_REGEX = re.compile(r"\d{1,2}/\d{1,2}/\d{1,2}")


def _is_date(date: str):
    return DATE_REGEX.match(date)


def _quick_test():
    """"Convenience access for quick testing"""
    country = country_data("United Kingdom")
    print(country.days_to_ind())


if __name__ == "__main__":
    _quick_test()
