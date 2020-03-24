"""Test script"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from SEQIJR import SEQIJR
import data_api
import utils


def main():
    """Main entrypoint"""
    config = utils.read_config("config.json")
    print(config)
    country = data_api.country_data("Italy")
    print(country)
    x_confirmed = country.days_to_ind()
    y_confirmed = country.confirmed

    model = init_model(country.population, config)
    print(model)

    y_0 = init_y(model.N)
    resolution = 0.5

    x, y_1, aj_p, j_p = predictions(model, country, y_0, 0, 365, resolution)

    plot_L = True
    if plot_L:
        plot_l(x, aj_p, x_confirmed, y_confirmed)
    else:
        plot_not_l(country, model, x, y_1, x_confirmed, y_confirmed,
                   resolution, t_end, j_p)


def predictions(model, country, y_0, start, end, resolution):
    """Step-wise prediction"""
    _, s_p_1, e_p_1, q_p_1, i_p_1, j_p_1, r_p_1, aj_p_1, aij_p_1, ad_p_1 = model.prediction(
        y_0, start, end, resolution)

    m = np.argmax(aj_p_1 > min(country.confirmed))

    y_0 = np.array([
        s_p_1[m], e_p_1[m], q_p_1[m], i_p_1[m], j_p_1[m], r_p_1[m], aj_p_1[m],
        aij_p_1[m], ad_p_1[m]
    ],
                   dtype=float).transpose()

    days_ahead = 1

    t_start = min(country.days_to_ind())
    t_end = max(country.days_to_ind()) + days_ahead

    x, s_p, e_p, q_p, i_p, j_p, r_p, aj_p, aij_p, ad_p = model.prediction(
        y_0, t_start, t_end, resolution)

    y_1 = np.array([
        s_p[m], e_p[m], q_p[m], i_p[m], j_p[m], r_p[m], aj_p[m], aij_p[m],
        ad_p[m]
    ],
                   dtype=float).transpose()
    return x, y_1, aj_p, j_p


def init_model(population: int, config: dict):
    """Create model
    """

    return SEQIJR(N=config["population_factor"] * population,
                  Pi=config["Pi"],
                  mu=config["inv_avg_age_in_days"],
                  b=config["b"],
                  e_E=config["e_E"],
                  e_Q=config["e_Q"],
                  e_J=config["e_J"],
                  g_1=config["g_1"],
                  g_2=config["g_2"],
                  s_1=config["s_1"],
                  s_2=config["s_2"],
                  k_1=config["k_1"],
                  k_2=config["k_2"],
                  d_1=config["d_1"],
                  d_2=config["d_1"])


def init_y(population: int):
    """Initialise"""
    s_0 = population
    e_0 = 0
    q_0 = 0
    i_0 = 1
    j_0 = 0
    r_0 = 0
    aj_0 = 0
    aij_0 = 1
    ad_0 = 0

    return np.array([s_0, e_0, q_0, i_0, j_0, r_0, aj_0, aij_0, ad_0],
                    dtype=float).transpose()


def plot_l(x, aJ_p, x_confirmed, y_confirmed):
    plt.plot(x, aJ_p)
    plt.plot(x_confirmed, y_confirmed, "bo", fillstyle="none")
    plt.show()


def plot_not_l(country, model, x, y_1, x_confirmed, y_confimed, h, t_end, J_p):
    m = int(max(country.days_to_ind()) / h)
    model.g_1 = 1
    t_start2 = max(country.days_to_ind())
    x2, S_p2, E_p2, Q_p2, I_p2, J_p2, R_p2, aJ_p2, aIJ_p2, aD_p2 = model.prediction(
        y_1, t_start2, t_end, h)
    plt.plot([min(x), max(x)], [571 * 1.5, 571 * 1.5])
    plt.plot(x, 0.05 * J_p)
    plt.plot(x2, 0.05 * J_p2)
    plt.show()


if __name__ == "__main__":
    main()
