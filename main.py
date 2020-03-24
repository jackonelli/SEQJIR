"""Test script"""
import numpy as np
import matplotlib.pyplot as plt
from SEQIJR import SEQIJR
from FileReader import FileReader
import data_api


def main():
    """Main entrypoint"""
    covid19_filename = "COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx"
    population_filename = "PopulationByCountry.xlsx"

    country = data_api.Country("Italy")
    raw_data = data_api.get_data(data_api.CONFIRMED_URL)
    country.add_data(raw_data)
    files = FileReader(covid19_filename, population_filename)
    country.population = files.population(country.name)
    print(country)

    x_confirmed = country.days_to_ind()
    y_confirmed = country.confirmed

    N = .5 * country.population
    Pi = 0
    mu = 1 / (80 * 365)  # 1/(80*365) (average age in days)^(-1)
    b = .8
    e_E = 0
    e_Q = 0
    e_J = 0.05
    g_1 = 1 / 19
    g_2 = 1 / 7
    s_1 = 1 / 14
    s_2 = 1 / 7
    k_1 = 1 / 3
    k_2 = 1 / 14
    d_1 = 0.05 / 7
    d_2 = 0.03 / 7

    model = SEQIJR(N, Pi, mu, b, e_E, e_Q, e_J, g_1, g_2, s_1, s_2, k_1, k_2,
                   d_1, d_2)

    print("R_0 = " + str(model.R_0()))
    print("R_c = " + str(model.R_c()))

    S_0 = N
    E_0 = 0
    Q_0 = 0
    I_0 = 1
    J_0 = 0
    R_0 = 0
    aJ_0 = 0
    aIJ_0 = 1
    aD_0 = 0

    y_0 = np.array([S_0, E_0, Q_0, I_0, J_0, R_0, aJ_0, aIJ_0, aD_0],
                   dtype=float).transpose()

    h = 1 / 2
    x_1, S_p_1, E_p_1, Q_p_1, I_p_1, J_p_1, R_p_1, aJ_p_1, aIJ_p_1, aD_p_1 = model.prediction(
        y_0, 0, 365, h)

    m = np.argmax(aJ_p_1 > min(y_confirmed))

    y_0 = np.array([
        S_p_1[m], E_p_1[m], Q_p_1[m], I_p_1[m], J_p_1[m], R_p_1[m], aJ_p_1[m],
        aIJ_p_1[m], aD_p_1[m]
    ],
                   dtype=float).transpose()

    P = 1

    t_start = min(country.days_to_ind())
    t_end = max(country.days_to_ind()) + P

    x, S_p, E_p, Q_p, I_p, J_p, R_p, aJ_p, aIJ_p, aD_p = model.prediction(
        y_0, t_start, t_end, h)

    plot_L = True

    if plot_L:
        plt.plot(x, aJ_p)
        plt.plot(x_confirmed, y_confirmed, "bo", fillstyle="none")
    else:
        m = int(max(country.days_to_ind()) / h)
        y_1 = np.array([
            S_p[m], E_p[m], Q_p[m], I_p[m], J_p[m], R_p[m], aJ_p[m], aIJ_p[m],
            aD_p[m]
        ],
                       dtype=float).transpose()
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
