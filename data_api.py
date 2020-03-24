"""Data module"""
import re
from datetime import datetime
import urllib.request
import pandas as pd
import numpy as np
from functional import seq

CONFIRMED_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"


def get_data(url: str):
    """Main entry point"""
    return pd.read_csv(url)


def list_countries(full_data):
    """List all countries in a dataframe.
    Nice to have to see the proper ID spelling
    """
    # List/set shenanigans give unique entries
    countries = sorted(list(set(full_data["Country/Region"])))
    return countries


def get_days(df):
    """Return array of dates from df columns"""
    return seq(df.columns).filter(_is_date)


def _parse_date(date: str):
    return datetime.strptime(date, "%m/%d/%y").date()


DATE_REGEX = re.compile(r"\d{1,2}/\d{1,2}/\d{1,2}")


def _is_date(date: str):
    return DATE_REGEX.match(date)


class Country:
    """Coronoa data by country"""
    def __init__(self, name: str):
        self.name = name
        self.population = None
        self.days = None
        self.confirmed = None
        self.deaths = None

    def __str__(self):
        return "{}\n{}\n\tPopulation: {}\n\tInterval: {}\n{}".format(
            self._delim_str(), self.name.upper(), self.population,
            self.time_interval(), self._delim_str())

    def _delim_str(self):
        return "-" * len(self.name)

    def time_interval(self, format_="%y-%m-%d"):
        start = self.days[0].strftime(format_)
        end = self.days[-1].strftime(format_)
        return "{} - {}".format(start, end)

    def add_data(self, df: pd.DataFrame):
        """Populate datastruct from raw pandas df

        Currently sums all provinces in a country.
        """
        day_columns = get_days(df)
        self.days = np.array(list(day_columns.map(_parse_date)))

        provinces = df[df["Country/Region"] == self.name]

        self.confirmed = provinces[day_columns].sum()

    def days_to_ind(self):
        """Generates an array of indices from the country's reported days"""
        return np.array(
            list(seq(self.days - self.days[0]).map(lambda x: int(x.days))))


def main():
    df = get_data(CONFIRMED_URL)
    country = Country("United Kingdom")
    country.add_data(df)
    print(country.days_to_ind())


if __name__ == "__main__":
    main()
