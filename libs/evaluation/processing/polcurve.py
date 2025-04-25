"""
Scripts for analysis of polarization curves
Created in 2023
@author: Nico Röttcher
"""

import datetime as dt
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sql
from IPython.display import clear_output
from ipywidgets import *
from matplotlib import gridspec

from evaluation.processing import tools_ec
# own modules
from evaluation.utils import db
import evaluation.utils.db_config as db_config
from evaluation.visualization import plot


# from importlib import reload
# reload(plot)
print(
    "\x1b[31m",
    "evaluation.processing.polcurve will be depracated. Please use tools_ec.polcurve_analysis",
    "\x1b[0m",
)


def polcurve_analysis(**kwargs):
    print(
        "\x1b[31m",
        "evaluation.processing.polcurve will be depracated. Please use tools_ec.polcurve_analysis",
        "\x1b[0m",
    )
    return tools_ec.polcurve_analysis(**kwargs)


def polcurve_analysis_old(
    name_user,
    name_setup_sfc,
    date,
    id_ML,
    add_cond="",
    number_datapoints_in_tail=40,
    j_geo_col="j__mA_cm2geo_fc_bottom_PTL",
    skip_id_data_eis_greater=2,
):
    """

    :param name_user: choose username
    :param name_setup_sfc: choose sfc setup
    :param date: chosse date
    :param id_ML: choose id_ML
    :param add_cond: add another condition to exp_sfc select statement
    :param number_datapoints_in_tail: number of datapoints at the of the ghold used to average current and potential
    :param j_geo_col: current column displayed in analysis plot, in database always absolute currents are stored
    :param skip_id_data_eis_greater: int
        it was observed that usually the first one or two datapoints of EIS measurement are not usable
        which is why they are discarded by default.
    :return: None
    """
    engine = db.connect(user="hte_polcurve_inserter")
    export_folder = db_config.DIR_REPORTS() / Path("02_polcurve_reports/")



    with engine.begin() as con:
        add_cond = ("AND " + add_cond) if add_cond != "" else ""
        exp_ec = pd.read_sql(
            '''SELECT  *
                            FROM hte_data.exp_ec_expanded 
                             WHERE name_user="'''
            + name_user
            + '''"
                             AND name_setup_sfc = "'''
            + name_setup_sfc
            + '''"
                             AND DATE(t_start__timestamp) = "'''
            + date
            + """"
                             AND id_ML ="""
            + str(id_ML)
            + """ 
                             """
            + add_cond
            + """ 
                             ;""",
            con=con,
            index_col="id_exp_sfc",
        )
        data_eis = exp_ec.dataset.get_data(
            con,
            "data_eis_analysis",
            add_cond="id_data_eis> " + str(skip_id_data_eis_greater),
        )
        data_ec = exp_ec.dataset.get_data(
            con,
            "data_ec_analysis",
        )

        polcurve_existing = con.execute(
            '''SELECT id_exp_ec_polcurve
                                    FROM exp_ec_polcurve_expanded
                                    WHERE name_user="'''
            + name_user
            + '''"
                                         AND name_setup_sfc = "'''
            + name_setup_sfc
            + '''"
                                         AND DATE(t_start__timestamp) = "'''
            + date
            + """"
                                         AND id_ML ="""
            + str(id_ML)
            + """
                                         ;"""
        ).fetchall()
    if len(polcurve_existing) == 0:
        id_exp_ec_polcurve = "new"
    elif len(polcurve_existing) == 1:
        id_exp_ec_polcurve = polcurve_existing[0][0]
    else:
        id_exp_ec_polcurve = ""
        sys.exit(
            "Error multiple polarization curve experiments found. Please inform Admin."
        )

    if not all(exp_ec.ec_name_technique.isin(["exp_ec_ghold", "exp_ec_geis"])):
        warnings.warn(
            "techniques other than exp_ec_ghold and exp_ec_geis will be ignored"
        )

    # Check for equality of experimental parameters throughgout polarization curve
    cols_value_difference_accepted = [
        "t_start__timestamp",
        "id_ML_technique",
        "t_duration__s",
        "t_end__timestamp",
        "ec_name_technique",
        "geis_I_dc__A",
        "ghold_I_hold__A",
        "ghold_t_hold__s",
    ]

    exp_ec_differences = exp_ec.loc[
        :, [col for col in exp_ec.columns if col not in cols_value_difference_accepted]
    ].loc[
        :, exp_ec.nunique() > 1
    ]  # None + 1 unique value is accepted (geis/ghold parameter for example)

    if len(exp_ec_differences.columns) > 0:
        display(exp_ec_differences)
        print(
            "\x1b[31m",
            " Experimental parameters have been changed during the measurement! "
            "This is not implemented in database. \n",
            " The VIEW exp_ec_polcurve_expanded will display the experimental parameters of the first technique. \n",
            " Consider that in your analysis!",
            "\x1b[0m",
        )

        # ['ghold_I_hold__A', 'ghold_t_hold__s', 'ghold_t_samplerate__s']
    exp_ec_ghold = (
        exp_ec.loc[
            exp_ec.ec_name_technique.isin(["exp_ec_ghold"]),
            [
                "ghold_I_hold__A",
            ],
        ]
        .reset_index()
        .rename(columns={"id_exp_sfc": "id_exp_sfc_ghold"})
    )

    exp_ec_geis = (
        exp_ec.loc[exp_ec.ec_name_technique.isin(["exp_ec_geis"]), :]
        .reset_index()
        .rename(columns={"id_exp_sfc": "id_exp_sfc_geis"})
    )
    exp_ec_sgeis = exp_ec_ghold.join(exp_ec_geis, rsuffix="geis")

    display_exp_ec_sgeis_last_current_step = False
    if not all(exp_ec_sgeis.geis_I_dc__A == exp_ec_sgeis.ghold_I_hold__A):
        if np.isnan(exp_ec_sgeis.geis_I_dc__A.iloc[-1]):
            print(
                "\x1b[33m",
                " EIS measurment on last current step is not found. "
                "This current step will not be considered but displayed in orange",
                "\x1b[0m",
            )
            display_exp_ec_sgeis_last_current_step = True
            exp_ec_sgeis_last_current_step = exp_ec_sgeis.iloc[
                [
                    -1,
                ]
            ]
            exp_ec_sgeis = exp_ec_sgeis.iloc[:-1]
        else:
            display(exp_ec_sgeis.loc[:, ["geis_I_dc__A", "ghold_I_hold__A"]])

    exp_ec_sgeis["scan_direction"] = exp_ec_sgeis.groupby("ghold_I_hold__A")[
        "ghold_I_hold__A"
    ].transform(lambda x: 1 + np.arange(len(x)))
    exp_ec_sgeis["next_ghold_I_hold__A"] = exp_ec_sgeis["ghold_I_hold__A"].shift(-1)
    exp_ec_sgeis["prev_ghold_I_hold__A"] = exp_ec_sgeis["ghold_I_hold__A"].shift(1)
    exp_ec_sgeis["scan_direction_2"] = exp_ec_sgeis.apply(
        lambda x: 1
        if (
            x.loc["ghold_I_hold__A"] <= x.loc["next_ghold_I_hold__A"]
            or np.isnan(x.loc["next_ghold_I_hold__A"])
        )
        and (
            x.loc["ghold_I_hold__A"] >= x.loc["prev_ghold_I_hold__A"]
            or np.isnan(x.loc["prev_ghold_I_hold__A"])
        )
        else 2,
        axis=1,
    )

    if not all(exp_ec_sgeis["scan_direction"].isin([1, 2])):
        display(exp_ec_sgeis.loc[:, ["scan_direction"]])
        warnings.warn(
            "something went wrong with merging forward and backward scan, third scan_direction found?"
        )

    if not all(exp_ec_sgeis.scan_direction == exp_ec_sgeis.scan_direction_2):
        display(exp_ec_sgeis.loc[:, ["scan_direction", "scan_direction_2"]])
        warnings.warn(
            "something went wrong with merging forward and backward scan, difference in two calculation methods"
        )

    exp_ec_sgeis["id_current_step"] = (
        exp_ec_sgeis.sort_values(by=["ghold_I_hold__A"])
        .groupby("scan_direction")["scan_direction"]
        .transform(lambda x: 1 + np.arange(len(x)))
    )

    # derive list of overload errors appeared during measurement
    exp_ec_sgeis = exp_ec_sgeis.join(
        data_ec.groupby("id_exp_sfc")["overload"].unique().astype(str),
        on="id_exp_sfc_ghold",
    ).rename(columns={"overload": "overload_list"})

    if id_exp_ec_polcurve == "new":
        # derive R_u guess form minimum
        exp_ec_sgeis = exp_ec_sgeis.join(
            data_eis.loc[
                data_eis.loc[:, "minusZ_img__ohm"].groupby(level=0).idxmin(), :
            ]
            .reset_index(level=1)
            .id_data_eis,
            on="id_exp_sfc_geis",
        ).rename(columns={"id_data_eis": "id_data_eis_chosen_Ru"})
        # calculate default gooddata
        exp_ec_sgeis.loc[:, "gooddata"] = (
            exp_ec_sgeis.loc[:, "overload_list"]
            .apply(lambda x: True if len(x.split(" ")) == 1 else False)
            .astype(np.bool_)
        )
    else:
        with engine.begin() as con:
            # Get R_u__ohm and gooddata from database
            existing_data = pd.read_sql(
                """SELECT id_exp_sfc_geis, id_data_eis_chosen_Ru, gooddata
                   FROM data_ec_polcurve
                   WHERE id_exp_ec_polcurve="""
                + str(id_exp_ec_polcurve)
                + """
                                                  ;""",
                con=con,
                index_col="id_exp_sfc_geis",
            )
        exp_ec_sgeis = exp_ec_sgeis.join(existing_data, on="id_exp_sfc_geis")
        exp_ec_sgeis.loc[
            ~exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"
        ] = exp_ec_sgeis.loc[
            ~exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"
        ].astype(
            np.bool_
        )
        exp_ec_sgeis.loc[exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"] = (
            exp_ec_sgeis.loc[exp_ec_sgeis.loc[:, "gooddata"].isna(), "overload_list"]
            .apply(lambda x: True if len(x.split(" ")) == 1 else False)
            .astype(np.bool_)
        )
        # print(type(exp_ec_sgeis.gooddata.iloc[0]))

    # time synchronisattion of eis data
    data_eis = data_eis.join(exp_ec.t_start__timestamp, on="id_exp_sfc")
    data_eis.loc[data_eis.Timestamp.isna(), "Timestamp"] = data_eis.loc[
        data_eis.Timestamp.isna(), "t_start__timestamp"
    ]
    data_eis.loc[:, "t__s"] = (
        pd.to_datetime(data_eis.Timestamp) - pd.to_datetime(data_ec.Timestamp).min()
    ).dt.total_seconds()
    data_ec.loc[:, "t__s"] = (
        pd.to_datetime(data_ec.Timestamp) - pd.to_datetime(data_ec.Timestamp).min()
    ).dt.total_seconds()  # .dtRu_calculated.loc[:, 't_start__timestamp'].to_timestamp().min()

    # init exp_ec_polcurve
    exp_ec_polcurve = pd.DataFrame.from_dict(
        {
            id_exp_ec_polcurve: [
                number_datapoints_in_tail,
                str(exp_ec_differences.columns.tolist()),
            ]
        },
        columns=["number_datapoints_in_tail", "changed_exp_parameters"],
        orient="index",
    )
    exp_ec_polcurve.index.name = "id_exp_ec_polcurve"

    # exp_ec_polcurve.loc[:, 'id_exp_sfc_first'] = exp_ec_test.reset_index().id_exp_sfc.iloc[0]

    def calculate_data_ec_polcurve(
        data_ec_old,
        data_eis_old,
        exp_ec_sgeis_old,
        number_datapoints_in_tail_new=number_datapoints_in_tail,
    ):
        # cut data used for averaging the potential
        data_ec_polcurve_raw_new = (
            data_ec_old.groupby("id_exp_sfc", as_index=False)
            .apply(
                lambda x: x
                if len(x) <= number_datapoints_in_tail_new
                else x.iloc[-number_datapoints_in_tail_new:]
            )
            .reset_index(level=0, drop=True)
        )

        data_ec_polcurve_new = (
            exp_ec_sgeis_old.set_index("id_exp_sfc_ghold")
            .join(
                data_ec_polcurve_raw_new.loc[
                    :, ~data_ec_old.columns.isin(["overload", "cycle", "t__s", "Timestamp"])
                ]
                .groupby("id_exp_sfc")
                .mean()
            )
            .join(
                data_ec_polcurve_raw_new.loc[
                    :, ~data_ec_old.columns.isin(["overload", "cycle", "t__s", "Timestamp"])
                ]
                .groupby("id_exp_sfc")
                .std(),
                rsuffix="_std",
            )
            .join(
                data_eis_old.Z_real__ohm.rename("R_u__ohm"),
                on=["id_exp_sfc_geis", "id_data_eis_chosen_Ru"],
            )
        )

        # calculate compenmsated potential:
        data_ec_polcurve_new.loc[:, "E_WE__VvsRHE"] = data_ec_polcurve_new.loc[
            :, "E_WE_uncompensated__VvsRHE"
        ] - (data_ec_polcurve_new.loc[:, "I__A"] * data_ec_polcurve_new.loc[:, "R_u__ohm"])

        # write corresponding id_exp_ec_polcurve
        data_ec_polcurve_new.loc[:, "id_exp_ec_polcurve"] = exp_ec_polcurve.reset_index().id_exp_ec_polcurve.iloc[0]

        data_ec_polcurve_new = (
            data_ec_polcurve_new.reset_index()
            .reset_index()
            .rename(columns={"index": "id_data_ec_polcurve"})
            .set_index(["id_exp_ec_polcurve", "id_data_ec_polcurve"])
        )

        return data_ec_polcurve_new, data_ec_polcurve_raw_new

    data_ec_polcurve, data_ec_polcurve_raw = calculate_data_ec_polcurve(
        data_ec,
        data_eis,
        exp_ec_sgeis,
        number_datapoints_in_tail_new=number_datapoints_in_tail,
    )

    # return exp_ec_sgeis, data_ec_polcurve_raw, data_eis
    with plt.rc_context(
        plot.get_style(
            style="singleColumn",
            interactive=True,
            increase_fig_height=2,
            add_margins_and_figsize={"right": 0.8, "left": 0.2},
            add_margins_between_subplots={"hspace": 3},
        )
    ):
        plot_storage = plot.PlotDataStorage(
            "statistics_number_of_experiments", overwrite_existing=True
        )
        fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        gs = gridspec.GridSpec(3, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.export_name = "ax1_top"
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.export_name = "ax2_middle_leftyaxis"
        ax2r = ax2.twinx()
        ax2r.export_name = "ax2_middle_rightyaxis"
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.export_name = "ax3_bottom"
        #
        #
        # exp_ec_geis =
        exp_ec_sgeis_label_helper = exp_ec_sgeis.set_index("id_exp_sfc_geis").join(
            exp_ec, lsuffix="pol_curve"
        )
        exp_ec_sgeis = (
            exp_ec_sgeis.set_index("id_exp_sfc_geis")
            .dataset.add_column(
                "label",
                values="id in ML: "
                + exp_ec_sgeis_label_helper.loc[:, "id_ML"].astype(str)
                + ",   Spot: "
                + exp_ec_sgeis_label_helper.loc[:, "id_spot"].astype(str)
                + ",   @ "
                + (exp_ec_sgeis_label_helper.loc[:, "geis_I_dc__A"] * 1000).astype(str)
                + " mA",
            )
            .add_column("color", values="pub_blues")
            .plot(
                x_col="Z_real__ohm",
                y_col="minusZ_img__ohm",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_eis,
                ax=ax1,
                # alpha=0.2
            )
            .return_dataset()
            .reset_index()
        )
        # display(exp_ec_sgeis)
        (Ru_dots,) = ax1.plot(
            data_eis.loc[
                exp_ec_sgeis.set_index(
                    ["id_exp_sfc_geis", "id_data_eis_chosen_Ru"]
                ).index,
                "Z_real__ohm",
            ],
            data_eis.loc[
                exp_ec_sgeis.set_index(
                    ["id_exp_sfc_geis", "id_data_eis_chosen_Ru"]
                ).index,
                "minusZ_img__ohm",
            ],
            marker=".",
            linestyle="",
            color="tab:orange",
        )

        exp_ec_sgeis_ghold_label_helper = exp_ec_sgeis.set_index(
            "id_exp_sfc_ghold"
        ).join(exp_ec, lsuffix="pol_curve")
        exp_ec_sgeis = (
            exp_ec_sgeis.set_index("id_exp_sfc_ghold")
            .dataset.add_column(
                "label",
                values="id in ML: "
                + exp_ec_sgeis_ghold_label_helper.loc[:, "id_ML"].astype(str)
                + ",   Spot: "
                + exp_ec_sgeis_ghold_label_helper.loc[:, "id_spot"].astype(str)
                + ",   @ "
                + (
                    exp_ec_sgeis_ghold_label_helper.loc[:, "geis_I_dc__A"] * 1000
                ).astype(str)
                + " mA",
            )
            .add_column("color", values="grey")
            .plot(
                x_col="t__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax2,
            )
            .add_column("color", values="pub_blues")
            .plot(
                x_col="t__s",
                y_col="E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax2r,
            )
            .plot(
                x_col="t__s",
                y_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax2,
                color="tab:green",
            )
            .plot(
                x_col="t__s",
                y_col="E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax2r,
                color="tab:green",
            )
            .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label="E_WE_uncompensated__VvsRHE",
                x_col="t__s",
                y_col="E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax2r,
                color="tab:green",
                alpha=0.5,
            )
            .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label="I__A",
                display_fit=False,
                x_col="t__s",
                y_col="I__A",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax2,
                color="tab:green",
                alpha=0.5,
            )
            .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label=j_geo_col,
                display_fit=True,
                x_col="t__s",
                y_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax2,
                color="tab:green",
                alpha=0.5,
            )
            .return_dataset()
            .reset_index()
        )
        if display_exp_ec_sgeis_last_current_step:
            exp_ec_sgeis_last_current_step.set_index(
                "id_exp_sfc_ghold"
            ).dataset.add_column("label", values="hold without EIS").add_column(
                "color", values="grey"
            ).plot(
                x_col="t__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax2,
            ).add_column(
                "color", values="tab:orange"
            ).plot(
                x_col="t__s",
                y_col="E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax2r,
            )
        exp_ec_sgeis = (
            exp_ec_sgeis.set_index("id_exp_sfc_geis")
            .dataset.add_column("color", values="grey")
            .plot(
                x_col="t__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_eis,
                ax=ax2,
            )
            .add_column("color", values="pub_blues")
            .plot(
                x_col="t__s",
                y_col="E_WE_uncompensated__VvsRHE",
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_eis,
                ax=ax2r,
            )
            .return_dataset()
            .reset_index()
        )
        exp_ec_polcurve = (
            exp_ec_polcurve.dataset.add_column("color", values="grey")
            .plot(
                x_col="E_WE_uncompensated__VvsRHE",
                y_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax3,
                marker="s",
                markersize=2,
                linestyle="-",
            )
            .add_column("color", values="tab:blue")
            .plot(
                x_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                y_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax3,
                marker="s",
                markersize=2,
                linestyle="-",
            )
            .add_column("color", values="tab:red")
            .plot(
                x_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                y_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve.loc[~data_ec_polcurve.loc[:, "gooddata"], :],
                ax=ax3,
                marker="s",
                markersize=2,
                linestyle="",
            )
            .return_dataset()
        )

        def update_Ru_marker():
            Ru_dots.set_xdata(
                data_eis.loc[
                    exp_ec_sgeis.set_index(
                        ["id_exp_sfc_geis", "id_data_eis_chosen_Ru"]
                    ).index,
                    "Z_real__ohm",
                ]
            )
            Ru_dots.set_ydata(
                data_eis.loc[
                    exp_ec_sgeis.set_index(
                        ["id_exp_sfc_geis", "id_data_eis_chosen_Ru"]
                    ).index,
                    "minusZ_img__ohm",
                ]
            )
            # ax1.autoscale() # will lead to rescale of acxis upon every Ru marker change
            fig.canvas.draw()  # does the update of axis

        def update_polcurve_raw_tail(data_ec_polcurve_raw_new):
            ax1.set_title("changed")
            for index, row in (
                pd.DataFrame.from_dict(
                    dict(
                        zip(
                            exp_ec_sgeis.id_exp_sfc_ghold.values,
                            exp_ec_sgeis.ax_plot_objects.values,
                        )
                    )
                )
                .transpose()
                .iterrows()
            ):
                row.loc[3][0].set_xdata(
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), "t__s"]
                )
                row.loc[3][0].set_ydata(
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), j_geo_col]
                )
                row.loc[4][0].set_xdata(
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), "t__s"]
                )
                row.loc[4][0].set_ydata(
                    data_ec_polcurve_raw_new.loc[
                        (index, slice(None)), "E_WE_uncompensated__VvsRHE"
                    ]
                )

        def update_polcurve(data_ec_polcurve_new):
            # blue compensated pol curve
            exp_ec_polcurve.ax_plot_objects.iloc[0][1][0].set_xdata(
                data_ec_polcurve_new.loc[:, "E_WE__VvsRHE"]
            )
            exp_ec_polcurve.ax_plot_objects.iloc[0][1][0].set_ydata(
                data_ec_polcurve_new.loc[:, j_geo_col]
            )
            # red not gooddata
            exp_ec_polcurve.ax_plot_objects.iloc[0][2][0].set_xdata(
                data_ec_polcurve_new.loc[
                    ~data_ec_polcurve_new.loc[:, "gooddata"], "E_WE__VvsRHE"
                ]
            )
            exp_ec_polcurve.ax_plot_objects.iloc[0][2][0].set_ydata(
                data_ec_polcurve_new.loc[~data_ec_polcurve_new.loc[:, "gooddata"], j_geo_col]
            )
            ax3.autoscale()

        # Sliders
        init_id = 0
        intSlider_choose_R_u = widgets.IntSlider(
            value=exp_ec_sgeis.id_data_eis_chosen_Ru.iloc[init_id],
            # min=0, #init by radio
            # max=len(x), #init by radio
            step=1,
            description="Choose point",
            continuous_update=False,
            disabled=True,
        )

        def update_on_slider(changed_value):
            exp_ec_sgeis.loc[
                radio_text_to_df_row(radiobuttons.value, exp_ec_sgeis).name,
                "id_data_eis_chosen_Ru",
            ] = changed_value.new
            # print(radiobuttons.value)
            # output.update('test ' + str(radiobuttons.value) + ' - ' + str('i'))
            # display(('test ' + str(radiobuttons.value) + ' - ' + str('i')))
            update_Ru_marker()
            # display(data_ec_polcurve)
            data_ec_polcurve_new, data_ec_polcurve_raw_new = calculate_data_ec_polcurve(
                data_ec,
                data_eis,
                exp_ec_sgeis,
                number_datapoints_in_tail_new=number_datapoints_in_tail,
            )
            # data_ec_polcurve.loc[:,'R_u__ohm'] = data_eis.loc[exp_ec_sgeis.set_index(['id_exp_sfc_geis',
            # 'id_data_eis_chosen_Ru']).index, 'Z_real__ohm'].to_numpy()
            # calculate_R_u_ohm()
            update_polcurve(data_ec_polcurve_new)

        intSlider_choose_R_u.observe(update_on_slider, "value")

        # Checkbox
        def update_on_checkbox_gooddata(changed_value):
            exp_ec_sgeis.loc[
                radio_text_to_df_row(radiobuttons.value, exp_ec_sgeis).name, "gooddata"
            ] = changed_value.new
            data_ec_polcurve_new, data_ec_polcurve_raw_new = calculate_data_ec_polcurve(
                data_ec,
                data_eis,
                exp_ec_sgeis,
                number_datapoints_in_tail_new=number_datapoints_in_tail,
            )
            update_polcurve(data_ec_polcurve_new)

        checkbox_gooddata = widgets.Checkbox(
            value=False, description="gooddata", disabled=True
        )
        checkbox_gooddata.observe(update_on_checkbox_gooddata, "value")

        # Radiobuttons
        def df_to_radio_text(exp_ec_sgeis_new):
            return ["all"] + (
                exp_ec_sgeis_new.index.astype(str).to_numpy()
                + ": "
                + (exp_ec_sgeis_new.ghold_I_hold__A * 1000).astype(str)
                + " mA, \t initial_id_chosen: "
                + exp_ec_sgeis_new.id_data_eis_chosen_Ru.astype(str)
                + " overload: "
                + exp_ec_sgeis_new.overload_list.astype(str).str.replace("8191", "")
            ).tolist()

        def radio_text_to_df_row(text, exp_ec_sgeis_new):
            return exp_ec_sgeis_new.loc[int(text.split(": ")[0]), :]

        radiobuttons = widgets.RadioButtons(
            value="all",  # df_to_radio_text(exp_ec_sgeis).iloc[init_id],
            options=df_to_radio_text(exp_ec_sgeis),  # exp_ec_sgeis.ghold_I_hold__A,
            description="Choose EIS measurement",
            layout=Layout(width="100%"),
        )
        upload_output = widgets.Output()

        def update_on_radio(change):
            # Update slider
            if change.new == "all":
                intSlider_choose_R_u.disabled = True
                checkbox_gooddata.disabled = True
            else:
                intSlider_choose_R_u.disabled = False
                intSlider_choose_R_u.min = data_eis.loc[
                    radio_text_to_df_row(change.new, exp_ec_sgeis).loc[
                        "id_exp_sfc_geis"
                    ],
                    :,
                ].index.min()
                intSlider_choose_R_u.max = data_eis.loc[
                    radio_text_to_df_row(change.new, exp_ec_sgeis).loc[
                        "id_exp_sfc_geis"
                    ],
                    :,
                ].index.max()
                intSlider_choose_R_u.value = radio_text_to_df_row(
                    change.new, exp_ec_sgeis
                ).loc[
                    "id_data_eis_chosen_Ru"
                ]
                checkbox_gooddata.disabled = False
                checkbox_gooddata.value = bool(
                    radio_text_to_df_row(change.new, exp_ec_sgeis).loc["gooddata"]
                )
            # with upload_output:
            #    clear_output()
            #    print('hallo_')

            # Update alpha
            if change.old == "all":
                [
                    row.ax_plot_objects[0][0].set_alpha(0.2)
                    for index, row in exp_ec_sgeis.iterrows()
                ]
            else:
                radio_text_to_df_row(change.old, exp_ec_sgeis).loc["ax_plot_objects"][
                    0
                ][0].set_alpha(0.2)

            if change.new == "all":
                [
                    row.ax_plot_objects[0][0].set_alpha(1)
                    for index, row in exp_ec_sgeis.iterrows()
                ]
            else:
                radio_text_to_df_row(change.new, exp_ec_sgeis).loc["ax_plot_objects"][
                    0
                ][0].set_alpha(1)

            # change.owner.options=df_to_radio_text(exp_ec_sgeis),#exp_ec_sgeis.ghold_I_hold__A,
            # change.owner.value=df_to_radio_text(exp_ec_sgeis).iloc[init_id]

        radiobuttons.observe(update_on_radio, "value")

        # Field to enter tail
        intslider_tail_datapoints = widgets.IntSlider(
            value=number_datapoints_in_tail,
            min=10,
            max=100,
            step=1,
            description="tail datapoints:",
            disabled=True,  # befare enable --> adjustements of fit of selected segments need to be updated as well
            # continuous_update=False,
            orientation="horizontal",
            # readout=True,
            # readout_format='d'
        )

        def update_on_slider_tail(change):
            exp_ec_polcurve.loc[
                :, "number_datapoints_in_tail"
            ] = number_datapoints_in_tail

            data_ec_polcurve_new, data_ec_polcurve_raw_new = calculate_data_ec_polcurve(
                data_ec, data_eis, exp_ec_sgeis, number_datapoints_in_tail_new=change.new
            )

            update_polcurve_raw_tail(data_ec_polcurve_raw_new)
            update_polcurve(data_ec_polcurve_new)

        intslider_tail_datapoints.observe(update_on_slider_tail, "value")

        # Uplpoad Button
        def on_upload_button_clicked():
            radiobuttons.value = "all"
            with upload_output:
                clear_output()
                print("Processing ...")
                exp_ec_polcurve_tosql = exp_ec_polcurve.loc[
                    :, ["number_datapoints_in_tail", "changed_exp_parameters"]
                ]
                exp_ec_polcurve_tosql.loc[:, "t_inserted_data__timestamp"] = str(
                    dt.datetime.now()
                )
                id_exp_ec_polcurve_final = (
                    exp_ec_polcurve_tosql.reset_index().id_exp_ec_polcurve.iloc[0]
                )

                with engine.begin() as conn:
                    # begin() not connect()
                    # --> this way everythin is handled as transaction and will be rollbacked in case of error
                    db.call_procedure(
                        engine,
                        "Reset_Autoincrement",
                        ["exp_ec_polcurve", "id_exp_ec_polcurve"],
                    )
                    try:
                        if id_exp_ec_polcurve_final == "new":
                            # Insert into exp_ec_polcurve
                            exp_ec_polcurve_tosql = (
                                db.insert_into(
                                    conn, "exp_ec_polcurve", exp_ec_polcurve_tosql
                                )
                                .reset_index(drop=True)
                                .rename(
                                    columns={
                                        "inserted_primary_key": "id_exp_ec_polcurve"
                                    }
                                )
                                .set_index("id_exp_ec_polcurve")
                            )
                            print(
                                "\x1b[32m",
                                "Successfully inserted to exp_ec_polcurve",
                                "\x1b[0m",
                            )

                            # Determine auto-increment id_exp_ec_polcurve given by db
                            id_exp_ec_polcurve_final = exp_ec_polcurve_tosql.reset_index().id_exp_ec_polcurve.iloc[
                                0
                            ]
                            # Update global exp_ec_polcurve
                            exp_ec_polcurve.rename(
                                index={
                                    exp_ec_polcurve.reset_index().id_exp_ec_polcurve.iloc[
                                        0
                                    ]: id_exp_ec_polcurve_final
                                },
                                inplace=True,
                            )  # Overwrite id_exp_ec_polcurve in global exp_ec_polcurve
                        else:
                            # update exp (not delete+reinsert --> won't work with auto increment)
                            conn.execute(
                                '''UPDATE exp_ec_polcurve
                                    SET number_datapoints_in_tail = "'''
                                + str(
                                    exp_ec_polcurve_tosql.number_datapoints_in_tail.iloc[
                                        0
                                    ]
                                )
                                + '''",
                                            changed_exp_parameters = "'''
                                + str(
                                    exp_ec_polcurve_tosql.changed_exp_parameters.iloc[0]
                                )
                                + '''",
                                            t_inserted_data__timestamp = "'''
                                + str(
                                    exp_ec_polcurve_tosql.t_inserted_data__timestamp.iloc[
                                        0
                                    ]
                                )
                                + """"
                                        WHERE id_exp_ec_polcurve = """
                                + str(int(id_exp_ec_polcurve_final))
                                + """
                                ;"""
                            )
                            print(
                                "\x1b[32m",
                                "Successfully updated exp_ec_polcurve",
                                "\x1b[0m",
                            )
                            # delete data and reinsert
                            db.call_procedure(
                                engine,
                                "delete_data_ec_polcurve",
                                [int(id_exp_ec_polcurve_final)],
                            )

                        # Save figure and update file path in db
                        export_file_path = str(
                            export_folder
                            / str(
                                str(int(id_exp_ec_polcurve_final))
                                + "_"
                                + str(date)
                                + "_"
                                + str(name_user)
                                + "_"
                                + str(name_setup_sfc)
                                + "_"
                                + str(id_ML)
                            )
                        )
                        plt.savefig(export_file_path + ".svg")

                        conn.execute(
                            '''UPDATE exp_ec_polcurve
                                        SET file_path_processing_plot = "'''
                            + str(export_file_path)
                            + """"
                                        WHERE id_exp_ec_polcurve = """
                            + str(int(id_exp_ec_polcurve_final))
                            + """
                                ;"""
                        )
                        exp_ec_polcurve_tosql.loc[:, "file_path_processing_plot"] = str(
                            export_file_path
                        )  # just for display resaons
                        display(exp_ec_polcurve_tosql)

                        # recalculate data_ec_polcurve with chosen intslider_tail and updated id_ecp_ec_polcurve
                        data_ec_polcurve_new, data_ec_polcurve_raw_new = calculate_data_ec_polcurve(
                            data_ec,
                            data_eis,
                            exp_ec_sgeis,
                            number_datapoints_in_tail_new=intslider_tail_datapoints.value,
                        )
                        # Insert into data_ec_polcurve
                        # .rename(index={id_exp_ec_polcurve:exp_ec_polcurve_tosql.reset_index().id_exp_ec_polcurve.iloc[0]}, level=0)\

                        # display(
                        data_ec_polcurve_new.rename(
                            columns={
                                "I__A_linear_fit_m": "I_drift__A_s",
                                "E_WE_uncompensated__VvsRHE_linear_fit_m": "E_WE_uncompensated_drift__V_s",
                            }
                        ).loc[
                            :,
                            [
                                "id_exp_sfc_geis",
                                "id_exp_sfc_ghold",
                                "id_current_step",
                                "scan_direction",
                                "id_data_eis_chosen_Ru",
                                # 'R_u__ohm',
                                "overload_list",  # min
                                "gooddata",
                                # Timestamp't__s', #min
                                "I__A",  # avg
                                "I__A_std",  # std
                                "I_drift__A_s",
                                "E_WE_uncompensated__VvsRHE",  # avg
                                "E_WE_uncompensated__VvsRHE_std",  # std
                                "E_WE_uncompensated_drift__V_s",
                            ],
                        ].to_sql(
                            "data_ec_polcurve", con=conn, if_exists="append"
                        )

                        print(
                            "\x1b[32m",
                            "Successfully inserted to data_ec_polcurve",
                            "\x1b[0m",
                        )

                        upload_button.widget.children[
                            0
                        ].description = "Updating entry in database"

                    except sql.exc.IntegrityError as error:
                        if "Duplicate entry" in str(error.orig) and "UNIQUE" in str(
                            error.orig
                        ):
                            clear_output()
                            print(
                                "\x1b[31m",
                                "Duplicate entry error. Check uniqueness of: id_exp_sfc_ghold, id_exp_sfc_geis",
                                "\x1b[0m",
                            )
                        else:
                            print("\x1b[31m", "An error appeared", "\x1b[0m")
                        print(str(error.orig), type(error.orig))
                        display(exp_ec_polcurve)
                        sys.exit("error")

        upload_button = interact_manual(on_upload_button_clicked)
        upload_button.widget.children[0].description = (
            "Insert into database"
            if id_exp_ec_polcurve == "new"
            else "Updating entry in database"
        )

        # Sorting  widgets
        widgets_box = widgets.HBox(
            [
                widgets.VBox(
                    [
                        intslider_tail_datapoints,
                        radiobuttons,
                        intSlider_choose_R_u,
                        checkbox_gooddata,
                        upload_button.widget,
                        upload_output,
                    ],
                    layout=Layout(width="100%"),
                ),
            ]
        )

        display(widgets_box)

        plt.show()

