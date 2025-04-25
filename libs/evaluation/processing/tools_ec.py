"""
Scripts for processing and analyzing electrochemical data
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

import evaluation.utils.db as db
import evaluation.utils.db_config as db_config

from evaluation.utils import user_input, tools  # import user_input, round_digits, truncate
from evaluation.visualization import plot

# def _all_j_geo_cols(data, j_gelo_col):
#    return [col for col in data.columns if col[:12] == 'j__mA_cm2geo'] if j_geo_col is None else [j_geo_col]


geo_columns = plot.geo_columns


def ohmic_drop_correction(exp_ec, data_ec):
    """
    Perform ohmic drop correction based on exp_ec values for R_u__ohm and perecentage of compensation during experiment
    :param exp_ec: pd.DataFrame
        EC experimental dataframe
    :param data_ec: pd.DataFrame
        EC experimental data dataframe
    :return: exp_ec, data_ec
    """

    exp_ec_req_cols = ['ec_R_u__ohm', 'ec_iR_corr_in_situ__percent',  'ec_R_u_postdetermined__ohm',]

    if (any([col not in exp_ec.columns for col in exp_ec_req_cols])
            or any([col not in data_ec.columns for col in ['E_WE_raw__VvsRHE',
                                                           'I__A']])
        ):
        print("\x1b[31m", "Missing columns to perform partial ohmic drop correction"
                          " Required: %s" % exp_ec_req_cols,
              "\x1b[0m")
        return exp_ec, data_ec

    # initialize column with nans
    data_ec.loc[:, 'E_WE__VvsRHE'] = np.nan

    # partial post-correction
    data_ec_partial_post_corrected = data_ec.index.get_level_values('id_exp_sfc').isin(
        exp_ec.loc[(((exp_ec.ec_R_u__ohm * exp_ec.ec_iR_corr_in_situ__percent) != 0)
                      & ((exp_ec.ec_R_u__ohm * exp_ec.ec_iR_corr_in_situ__percent).notna())
                      )].index.get_level_values('id_exp_sfc'))
    if data_ec_partial_post_corrected.any():
        data_ec.loc[data_ec_partial_post_corrected,
                    'E_WE__VvsRHE'] = data_ec.loc[data_ec_partial_post_corrected, 'E_WE_raw__VvsRHE'] \
                                              - ((data_ec.loc[data_ec_partial_post_corrected, 'I__A']
                                                  * exp_ec.ec_R_u__ohm)
                                                 * (1 - exp_ec.ec_iR_corr_in_situ__percent / 100))

    # complete post-correction
    data_ec_complete_post_corrected = data_ec.index.get_level_values('id_exp_sfc').isin(
        exp_ec.loc[((exp_ec.ec_R_u_postdetermined__ohm != 0)
                    & (exp_ec.ec_R_u_postdetermined__ohm.notna())
                    )].index.get_level_values('id_exp_sfc'))
    if data_ec_complete_post_corrected.any():
        data_ec.loc[data_ec_complete_post_corrected,
                    'E_WE__VvsRHE'] = data_ec.loc[data_ec_complete_post_corrected, 'E_WE_raw__VvsRHE'] \
                                              - ((data_ec.loc[data_ec_complete_post_corrected, 'I__A']
                                                  * exp_ec.ec_R_u_postdetermined__ohm)
                                                 * 1)  # 100% correction
    return exp_ec, data_ec



def add_j_geo(
    exp_ec,
    data,
    I__A_col="I__A",
    suffix="",
    A_geo_col=None,
):
    """
    add j_geo to data_ec or data_eis, usually calculated by database view,
    however necessary if user-specific transformations are requested
    :param exp_ec: exp_ec with list of experiments
    :param data: data_ec or data_eis
    :param I__A_col: 'I__A' for data_ec and  data_eis 'I_dc__A' for data_eis or any other user-specific current column
    :param suffix: suffix for the resulting j_geo_column(s)
    :param A_geo_col: list or None, optional, Default None
        list of A_geo_col in exp_ec to derive j_geo
        None: will calculate j_geo for all A_geo columns defined in tools_ec.geo_columns.A_geo
    :return: data_ec
        including columns for j_geo
    """
    # get requested j_geo_cols
    if A_geo_col is None:
        A_geo_cols = geo_columns.A_geo.tolist()
    else:
        A_geo_cols = A_geo_col if type(A_geo_col) == list else [A_geo_col]
        for A_geo_col in A_geo_cols:
            if A_geo_col not in geo_columns.j_geo.tolist():
                print(
                    A_geo_col
                    + " is not a current density column check geo_columns.j_geo"
                )

    for A_geo_col in A_geo_cols:
        if A_geo_col in exp_ec.columns.tolist():
            j_geo_col = geo_columns.set_index("A_geo").loc[A_geo_col, "j_geo"]

            # check whteher j_geo_col exists in data_ec
            if j_geo_col in data.columns:
                print(
                    j_geo_col
                    + " already contained in given data. This column is skipped."
                )
                continue
            # print('add', j_geo_col)
            data.loc[:, j_geo_col + suffix] = (
                data.loc[:, I__A_col] * 1000 / (exp_ec.loc[:, A_geo_col] / 100)
            )

    return data


def find_subsequent_ec_technique(exp_ec, row, name_column):
    """
    find value in given column (name_column) of subsequent ec technique in the same EC batch (id_ML) given in exp_ec
    by id_ML_technique
    :param exp_ec: EC experimental DataFrame
    :param row: row of the experiment from whcih to find the subsequent experiment
    :param name_column: name of the column from which the value should be returned
    :return: value of name_column for the subsequent ec technique
    """
    # row = exp_ec.iloc[1]
    df_all_subsequent_techniques = exp_ec.loc[
        (
            (exp_ec.id_ML == row.id_ML)
            & (exp_ec.name_user == row.name_user)
            & (exp_ec.name_setup_sfc == row.name_setup_sfc)
            & (
                pd.to_datetime(exp_ec.t_start__timestamp).dt.date
                == pd.to_datetime(row.t_start__timestamp).date()
            )
            & (exp_ec.id_ML == row.id_ML)
            & (exp_ec.id_ML_technique > row.id_ML_technique)
        ),
        :,
    ]  # .min()
    # print(row.name_user, row.name_setup_sfc, pd.to_datetime(row.t_start__timestamp).date(),
    # row.id_ML, row.id_ML_technique)
    if len(df_all_subsequent_techniques.index) == 0:
        return np.nan
    return df_all_subsequent_techniques.loc[
        df_all_subsequent_techniques.id_ML_technique.idxmin(), name_column
    ]


def update_exp_sfc_t_end__timestamp():
    """
    Calculates timestamp of the end of an ec technique and upload it to exp_sfc.t_end__timestamp. This was necessary
    as the end timestamp was not uploaded to database in eCat <V4.5. Not required for new data.
    :return: None
    """
    con = db.connect("hte_write")
    name_user = db.current_user()

    # update_exp_sfc_t_end__timestamp for dc techniques using stored procedure
    db.call_procedure(con, "update_exp_sfc_t_end__timestamp", params=[name_user])
    # will only update t_end__timestamp of experiments belonging to the current user

    # update_exp_sfc_t_end__timestamp for ac techniques using start time of subsequent experiment
    exp_ec = db.query_sql(
        """SELECT  *
                            FROM exp_ec_expanded 
                            WHERE name_user= %s
                                AND ec_name_technique IN ('exp_ec_peis', 'exp_ec_geis')
                                AND t_end__timestamp IS NULL
                         ;""", # removed hte_data.
        params=[name_user],
        con=con,
        method="pandas",
        index_col="id_exp_sfc",
    )
    # data_eis = exp_ec.dataset.get_data(con, 'data_eis_analysis',

    # only consider eis experiments inserted directly by Jonas Software, because otherwise start timestamp is unknown
    # start time for ac techniques via python insert are build by id_ML and id_ML_technique
    exp_ec_selected = exp_ec.loc[
        [
            row.t_start__timestamp[-6:]
            != "000"[len(str(row["id_ML"])):]
            + str(row["id_ML"])
            + "000"[len(str(row["id_ML_technique"])):]
            + str(row["id_ML_technique"])
            for index, row in exp_ec.iterrows()
        ],
        :,
    ].copy()

    # if no eis techniques to update selected quit
    if len(exp_ec_selected.index) == 0:
        return True

    exp_ec_selected.loc[:, "Date"] = pd.to_datetime(
        exp_ec_selected.t_start__timestamp
    ).dt.date

    # select all techniques within id_MLs with ac techniques (necessary to find subsequent experiment)
    sql_query = "SELECT * FROM exp_ec_expanded WHERE "
    params = []
    for counter, (index, row) in enumerate(
        exp_ec_selected.loc[:, ["name_user", "name_setup_sfc", "Date"]]
        .drop_duplicates()
        .iterrows()
    ):
        id_MLs = (
            exp_ec_selected.loc[
                (
                    (exp_ec_selected.name_user == row.name_user)
                    & (exp_ec_selected.name_setup_sfc == row.name_setup_sfc)
                    & (
                        pd.to_datetime(exp_ec_selected.t_start__timestamp).dt.date
                        == row.Date
                    )
                ),
                "id_ML",
            ]
            .unique()
            .tolist()
        )
        id_MLs_str = ("%s, " * len(id_MLs))[:-2]
        # print(id_MLs_str)
        sql_query += " OR " if counter > 0 else ""
        sql_query += (
            """( name_user = %s
                        AND name_setup_sfc = %s 
                        AND DATE(t_start__timestamp) = %s 
                        AND id_ML IN ("""
            + id_MLs_str
            + """)
                   )"""
        )
        params += row.tolist() + id_MLs
    # print(' '.join([sql+str(params) for sql_query, params in zip(sql_query.split('%s'), params+[''])]))
    exp_ec_all = db.query_sql(
        sql_query, params=params, con=con, method="pandas", index_col="id_exp_sfc"
    )

    # derive t_end__timestamp
    exp_ec_selected.loc[:, "t_end__timestamp"] = exp_ec_selected.apply(
        lambda x: find_subsequent_ec_technique(exp_ec_all, x, "t_start__timestamp"),
        axis=1,
    ).tolist()  # tools_data_ec.find_subsequent_ec_technique(x, 't_start__timestamp', exp_ec)

    # update
    df_update = exp_ec_selected.loc[
        :,
        [
            "t_end__timestamp",
        ],
    ].dropna()
    if len(df_update.index) > 0:
        display(df_update)
        if user_input.user_input(
            text="Update the following experiments?\n",
            dtype="bool",
        ):
            db.sql_update(df_update, table_name="exp_sfc")
        else:
            print("Not updated")


def data_eis_avg_to_data_ec(exp_ec, data_ec, data_eis):
    """
    add weighted average value for dc current and potential during EIS measurement to data_ec.
    Useful for integration of current during eis experiments
    Per eis experiment two datapoints are added. One with the start time of the experiment.
    Another one with the start time of the next experiment as end time is not recorded (yet)
    :param exp_ec: pd.DataFrame
        EC experimental dataframe
    :param data_ec: pd.DataFrame
        EC data dataframe
    :param data_eis: pd.DataFrame
        EIS data dataframe
    :return: concatted data_ec and data_eis values transformed
    """
    if not data_eis.Timestamp.isna().all():
        print(
            "\x1b[33m",
            "At least some of the EIS data has a Timestamp (measured with eCat >4.9).",
            "This is not developed yet.",
            "\x1b[0m",
        )

    # data_eis.f__Hz / data_eis.f__Hz.groupby(level=0).sum()
    data_eis_ec_begin = pd.DataFrame({})
    data_eis_ec_begin.loc[:, "Timestamp"] = pd.to_datetime(
        exp_ec.loc[
            exp_ec.ec_name_technique.isin(["exp_ec_geis", "exp_ec_peis"]),
            "t_start__timestamp",
        ]
    )
    data_eis_ec_begin.loc[:, "id_data_ec"] = 0
    # data_eis_ec_begin.set_index('id_data_ec', append=True, inplace=True)
    data_eis_ec_end = pd.DataFrame({})
    data_eis_ec_end.loc[:, "Timestamp"] = pd.to_datetime(
        exp_ec.loc[
            exp_ec.ec_name_technique.isin(["exp_ec_geis", "exp_ec_peis"]),
            "t_end__timestamp",
        ]
    )
    # exp_ec.loc[exp_ec.ec_name_technique.isin(['exp_ec_geis', 'exp_ec_peis']), :]\
    #    .apply(lambda row: find_subsequent_ec_technique(exp_ec,
    #                                                    row,
    #                                                    't_start__timestamp'),
    #            axis=1)
    data_eis_ec_end.loc[:, "id_data_ec"] = 1
    # data_eis_ec_end.set_index('id_data_ec', append=True, inplace=True)
    data_eis_ec = pd.concat([data_eis_ec_begin, data_eis_ec_end])

    level = list(
        range(len(data_eis.index[0]) - 1)
    )  # usually just id_exp_sfc (level=0) but necessary for overlap with multiple indexes

    data_eis_ec.loc[:, "E_WE_raw__VvsRE"] = (
        (
            1
            / data_eis.f__Hz
            / (1 / data_eis.f__Hz).groupby(level=level).sum()
            * data_eis.E_dc__VvsRE
        )
        .groupby(level=level)
        .sum()
    )
    data_eis_ec.loc[:, "E_WE_uncompensated__VvsRHE"] = (
        (
            1
            / data_eis.f__Hz
            / (1 / data_eis.f__Hz).groupby(level=level).sum()
            * data_eis.E_dc__VvsRE
        )
        .groupby(level=level)
        .sum()
    )
    data_eis_ec.loc[:, "I__A"] = (
        (
            1
            / data_eis.f__Hz
            / (1 / data_eis.f__Hz).groupby(level=level).sum()
            * data_eis.I_dc__A
        )
        .groupby(level=level)
        .sum()
    )
    data_eis_ec.set_index("id_data_ec", append=True, inplace=True)
    data_eis_ec.loc[:, "t__s"] = pd.to_timedelta(
        data_eis_ec.loc[:, "Timestamp"] - data_eis_ec_begin.loc[:, "Timestamp"]
    ).dt.total_seconds()

    data_eis_ec = add_j_geo(
        exp_ec,
        data_eis_ec,
    )
    data_eis_ec.sort_index()  # .dataset.display()

    return pd.concat([data_ec, data_eis_ec]).sort_index()


def derive_HFR(
    exp_eis,
    data_eis,
    on="id_exp_sfc",
    method="minimumorintercept",
    show_control_plot=True,
    suffix="",
    append_cols=None,
):
    """
    Derive a guess for high frequency resistance for a set of experiments
    :param exp_eis: list of eis experiments to be analyzed
    :param data_eis: corresponding data of eis experiments
    :param on: index to match experiments to data
    :param method: method to derive the HFR , choose from: 'minimum', 'intercept', 'minimumorintercept'
    :param show_control_plot: creates a figure to visually control the quality of the HFR extraction
    :param suffix: str, optional default ''
        suffix to append to the name of columns being appended to exp_eis (for example if applied mutliple times)
    :param append_cols: list of columns to be appended to exp_eis, default only essential columns
    :return: exp_eis with derived HFR the experimental set with
    """

    def derive_HFR_single_exp(group):
        """
        Derive a guess for high frequency resistance for a single experiment
        """
        # display( group.sort_values(by=['id_exp_sfc',  'f__Hz'],
        # ascending=[True, False]).shift(-1).sort_values(by=['id_exp_sfc',  'id_data_eis']).minusZ_img__ohm)

        # determine possible x-axis intercept points by comparing
        # with neighbouring values with lower and higher frequency
        group.loc[:, "next_lf_minusZ_img__ohm"] = (
            group.sort_values(by=["id_exp_sfc", "f__Hz"], ascending=[True, False])
            .shift(-1)
            .sort_values(by=["id_exp_sfc", "id_data_eis"])
            .minusZ_img__ohm
        )
        group.loc[:, "next_hf_minusZ_img__ohm"] = (
            group.sort_values(by=["id_exp_sfc", "f__Hz"], ascending=[True, False])
            .shift(1)
            .sort_values(by=["id_exp_sfc", "id_data_eis"])
            .minusZ_img__ohm
        )
        possible_intercept_points = (
            (group.minusZ_img__ohm < 0) & (group.next_lf_minusZ_img__ohm > 0)
        ) | ((group.minusZ_img__ohm > 0) & (group.next_hf_minusZ_img__ohm < 0))

        if method == "minimumorintercept":
            method_local = (
                "intercept"
                if group.minusZ_img__ohm.min() < 0 and possible_intercept_points.any()
                else "minimum"
            )
        else:
            method_local = method

        if method_local == "intercept":
            # intercept if negative values and x-axis intercept values
            id_data_eis_chosen_Ru = (
                group.loc[possible_intercept_points, "minusZ_img__ohm"]
                .abs()
                .idxmin()[-1]
            )
            # R_u_derived_by = 'intercept'
        elif method_local == "minimum":
            # minimum value if all values positive (or negative --> this should not be the case)
            id_data_eis_chosen_Ru = group.minusZ_img__ohm.idxmin()[-1]
            # R_u_derived_by = 'minimum'
        else:
            raise Exception(
                'method must be one of ["minimumorintercept", "minimum", "intercept"'
            )
        #if np.isnan(id_data_eis_chosen_Ru):
        #    print(
        #        "\x1b[31m",
        #        "HFR could not be detected. First datapoint is selected. Be aware this might not be corrected"
        #        "\x1b[0m",
        #    )
        #    id_data_eis_chosen_Ru = 0

        return pd.concat(
            [
                pd.Series(
                    {
                        "id_data_eis_chosen_Ru": id_data_eis_chosen_Ru,
                        "R_u_derived_by": method_local,  # R_u_derived_by,
                        "R_u__ohm": group.loc[
                            (group.index[0][0], id_data_eis_chosen_Ru), "Z_real__ohm"
                        ],
                    }
                ),
                None
                if append_cols is None
                else group.loc[
                    group.index.get_level_values(level=-1) == id_data_eis_chosen_Ru,
                    append_cols,
                ].iloc[0],
            ]
        )

    # Apply derive_HFR_single on each experiment in the list
    exp_eis = exp_eis.join(
        data_eis.groupby(level=0).apply(derive_HFR_single_exp), on=on, rsuffix=suffix
    )  # 'id_exp_sfc_geis')

    if show_control_plot:
        print("\x1b[33m", "Control quality of HFR extraction:", "\x1b[0m")
        with plt.rc_context(
            plot.get_style(
                style="singleColumn",
                add_params={
                    "figure.dpi": 150,
                },
            )
        ):
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            exp_eis = (
                exp_eis.dataset.add_column("color", values="tab10")
                .plot(
                    x_col="Z_real__ohm",
                    y_col="minusZ_img__ohm",
                    data=data_eis,
                    ax=ax1,
                    marker="s",
                    markersize=2,
                    alpha=0.3,
                    label="",
                )
                .add_column(
                    "label",
                    values="R$_\mathrm{u}$: "
                    + exp_eis.assign(dummy_index=exp_eis.index).dummy_index.astype(str),
                )
                .plot(
                    x_col="R_u__ohm",
                    y_col=0,
                    axlabel_auto=False,
                    marker="|",
                    markersize=15,
                    linestyle="",
                )
                .return_dataset()
            )
            ax1.legend(fontsize=5)
            plt.show()

    return exp_eis


def update_R_u__ohm(
    exp_ec,
    match_eis_id_ML=None,
    match_eis_timedelta__h=3,
    user_control=True,
    skip_id_data_eis_upto=0,
    **kwargs_derive_HFR
):
    """
    Loops through all exp_ec and searches for matching EIS data.
    For found EIS experiments the HFR will be derived and updated in exp_ec
    :param exp_ec: experimental dataframe received from exp_ec_expanded
    :param match_eis_id_ML: int or None, optional, default None
        match eis from given id_ML given as single int or list of ints with length of exp_ec
    :param match_eis_timedelta__h: search for matched eis n hours before and after start timestamp of ec experiment
    :param user_control: bool, optional, Default True
        whether user has to control the selection of EIS experiment and derivation of uncompensated resistance
    :param skip_id_data_eis_upto: Ignore the first data points of EIS spectrum.
        Why? It was observed that usually the first one or two datapoints of EIS measurement are not usable.
    :param kwargs_derive_HFR: keyword arguments for the derive_HFR method
    :return: exp_ec with updated columns ec_R_u__ohm, ec_R_u_determining_exp_ec
    """

    if (
        exp_ec.index.name != "id_exp_sfc"
        or "ec_R_u__ohm" not in exp_ec.columns
        or "ec_R_u_determining_exp_ec" not in exp_ec.columns
    ):
        raise Exception("Only works with exp_ec derived from exp_ec_expanded!")

    exp_ec.loc[:, "match_eis_id_ML"] = match_eis_id_ML
    for index, row in exp_ec.iterrows():
        # if row.name_user != db.current_user():
        #    print("\x1b[33m", 'View-only: This is data from ',
        #          row.name_user,
        #          '. Updates are restricted!',
        #          "\x1b[0m")

        if (
            row.ec_R_u__ohm != 0
            and row.ec_R_u__ohm is not None
            or row.ec_R_u_determining_exp_ec is not None
        ):
            # Either R_u_ohm already given when performed experiment
            # or updated by using this routine (ec_R_u_determining_exp_ec)
            print(
                "\x1b[33m",
                "Already updated exp_ec with index:",
                index,
                "with R_u__ohm=",
                row.ec_R_u__ohm,
                " from id_exp_sfc=",
                row.ec_R_u_determining_exp_ec,
                "\x1b[0m",
            )
            continue
        # print(index, ', '.join(exp_ec.index.names))
        sql_query = (
            '''
                    SELECT  *
                    FROM exp_ec_expanded 
                    WHERE name_user= "'''
            + row.name_user
            + '''"
                        AND name_setup_sfc = "'''
            + row.name_setup_sfc
            + """"
                        AND id_sample = """
            + str(row.id_sample)
            + """
                        AND id_spot = """
            + str(row.id_spot)
            + (
                ''' AND t_start__timestamp < "'''
                + str(
                    pd.to_datetime(exp_ec.iloc[0].t_start__timestamp)
                    + pd.Timedelta(hours=match_eis_timedelta__h)
                )
                + '''"
                         AND t_start__timestamp > "'''
                + str(
                    pd.to_datetime(exp_ec.iloc[0].t_start__timestamp)
                    - pd.Timedelta(hours=match_eis_timedelta__h)
                )
                + '"'
                if match_eis_id_ML is None
                else """ AND id_ML=""" + str(row.match_eis_id_ML)
            )
            + """ AND ec_name_technique IN ("exp_ec_peis", "exp_ec_geis")
                        ;
                     """
        )

        exp_eis = db.get_exp(sql_query)

        if len(exp_eis.index) == 0:
            print(
                "\x1b[33m",
                "No EIS experiments matched to given ec experiment with "
                + ", ".join(exp_ec.index.names)
                + ": "
                + str(index),
                "\x1b[0m",
            )
            print(sql_query)
            continue

        data_eis = db.get_data(exp_eis,
                               "data_eis_analysis",
                               add_cond="id_data_eis> " + str(skip_id_data_eis_upto),)
        exp_eis = derive_HFR(exp_eis, data_eis, **kwargs_derive_HFR)

        exp_eis.loc[:, "time_diff_to_ec__min"] = (
            pd.to_datetime(exp_eis.t_start__timestamp)
            - pd.to_datetime(row.t_start__timestamp)
        ).dt.total_seconds() / 60
        exp_eis.loc[:, "time_diff_to_ec_abs__min"] = exp_eis.time_diff_to_ec__min.abs()

        if len(exp_eis.index) == 1:
            print(
                "Matched experiment performed ",
                exp_eis.iloc[0].time_diff_to_ec__min,
                "min ",
                "earlier" if exp_eis.iloc[0].time_diff_to_ec__min > 0 else "later",
            )
            if user_control:
                if not user_input.user_input(
                    text="Transfer extracted R_u to given exp_ec with index: "
                    + str(index)
                    + "?\n",
                    dtype="bool",
                ):
                    continue
            index_selected_eis = exp_eis.index[0]
        else:
            print("\x1b[33m", "Multiple EIS experiments matched. Choose:", "\x1b[0m")
            exp_eis = exp_eis.sort_values(by="time_diff_to_ec_abs__min")
            # print((pd.to_datetime(exp_eis.t_start__timestamp)
            #        - pd.to_datetime(row.t_start__timestamp)).abs().idxmin())
            iloc_index_selected_eis = user_input.user_input(
                text="Select: \n",
                dtype="int",
                optional=False,
                options=pd.DataFrame(
                    {
                        "values": {
                            no: str(val)
                            for no, val in enumerate(exp_eis.reset_index().index)
                        },
                        "dropdown": {
                            no: ", ".join(
                                row_eis[exp_eis.index.names].values.astype(str)
                            )
                            + ", "
                            + row_eis.ec_name_technique
                            + ", "
                            + "id_ML="
                            + str(row_eis.id_ML)
                            + ", "
                            + "R_u="
                            + str(user_input.round_digits(row_eis.R_u__ohm, digits=3))
                            + " \u03A9, "
                            + "time difference to ec="
                            + str(
                                user_input.truncate(
                                    row_eis.time_diff_to_ec__min, decimals=2
                                )
                            )
                            + " min"
                            for no, (index_eis, row_eis) in enumerate(
                                exp_eis.reset_index().iterrows()
                            )
                        },
                    }
                ),
            )
            index_selected_eis = exp_eis.index[int(iloc_index_selected_eis)]
        # print(exp_eis.iloc[int(index_selected_eis)])
        print(
            "Updated exp_ec with index:",
            index,
            "with ec_R_u_postdetermined__ohm=",
            exp_eis.loc[index_selected_eis, "R_u__ohm"],
            " from id_exp_sfc=",
            index_selected_eis,
        )  # exp_eis.loc[index_selected_eis, :].index.get_level_values(level='id_exp_sfc'))
        # to link experiment to publication
        db.get_exp(
            by="SELECT * FROM exp_ec_expanded WHERE id_exp_sfc=%s",
            params=[int(index_selected_eis)],
        )

        exp_ec.loc[index, "ec_R_u_postdetermined__ohm"] = exp_eis.loc[index_selected_eis, "R_u__ohm"]
        # ec_R_u_postdetermined__ohm was before None, thus has dtype object,
        # but needs to be float for further operations
        exp_ec.loc[:, "ec_R_u_postdetermined__ohm"] = exp_ec.ec_R_u_postdetermined__ohm.astype(float)
        exp_ec.loc[index, "ec_R_u_determining_exp_ec"] = index_selected_eis

    return exp_ec


def compensate_R_u(exp_ec, data_ec):
    """
    Derive compensated potential based on given potential, current, and uncompensated resistance by columns
    ec_R_u__ohm or ec_R_u_postdetermined__ohm and ec_iR_corr_in_situ__percent in exp_ec.
    Also referencing potential to RHE by column ec_E_RE__VvsRHE on exp_ec
    :param exp_ec: experimental dataframe as received by exp_ec_expanded
    :param data_ec: experimental dataframe as received by data_ec
    :return: data_ec with new columns for compensated potential E_WE__VvsRHE
    """
    if "ec_R_u_postdetermined__ohm" in exp_ec.columns:
        # R_u as determined by tools_ec.update_Ru__ohm where exists
        # else R_u as given by user and stored in db as R_u__ohm (or ec_R_u__ohm)
        exp_ec.loc[
            ~exp_ec.ec_R_u_postdetermined__ohm.isna(), "ec_R_u__ohm"
        ] = exp_ec.loc[
            ~exp_ec.ec_R_u_postdetermined__ohm.isna(), "ec_R_u_postdetermined__ohm"
        ]

    exp_ec_index_col = exp_ec.index.names  # id_exp_sfc or added overlay_index_cols

    ec_R_u__ohm = data_ec.join(exp_ec.ec_R_u__ohm, on=exp_ec_index_col).ec_R_u__ohm
    ec_iR_corr_in_situ__percent = data_ec.join(
        exp_ec.ec_iR_corr_in_situ__percent, on=exp_ec_index_col
    ).ec_iR_corr_in_situ__percent
    ec_E_RE__VvsRHE = data_ec.join(
        exp_ec.ec_E_RE__VvsRHE, on=exp_ec_index_col
    ).ec_E_RE__VvsRHE

    data_ec = data_ec.assign(
        E_WE__VvsRHE=(data_ec.E_WE_raw__VvsRE + ec_E_RE__VvsRHE)
        - ((data_ec.I__A * ec_R_u__ohm) * (1 - (ec_iR_corr_in_situ__percent / 100))),
        E_WE_uncompensated__VvsRHE=(data_ec.E_WE_raw__VvsRE + ec_E_RE__VvsRHE)
        + data_ec.Delta_E_WE_uncomp__V,
        E_WE_raw__VvsRHE=(data_ec.E_WE_raw__VvsRE + ec_E_RE__VvsRHE),
    )
    # ((data_ec.E_WE__VvsRHE_test - data_ec.E_WE__VvsRHE) > 1e-7).any()
    # deviations from SQL result only due to FLOAT nature of value
    return data_ec


def geometric_current(
    exp_ec, data_ec, geo_cols=None, j_geo_col_subscript="", I_col__A="I__A"
):
    """
    Derive geometric current density based on metadata sample area, column name(s) specified by geo_cols
    :param exp_ec: experimental dataframe as received by exp_ec_expanded
    :param data_ec: experimental dataframe as received by data_ec
    :param geo_cols: optional, str or list of column names in exp_ec to derive geometric current density,
                    only values in plot.geo_columns allowed, default all values in plot.geo_column
    :param j_geo_col_subscript: add a subscript to create j_geo columns
    :param I_col__A: name of the colum from whic to derive the geometric current
    :return: data_ec with new columns for geometric current density
    """
    if geo_cols is None:
        geo_cols = plot.geo_columns.loc[:, "A_geo"]
    elif type(geo_cols) == str:
        geo_cols = [geo_cols]
    for geo_col in geo_cols:
        j_geo_col = plot.geo_columns.set_index("A_geo").loc[geo_col, "j_geo"]
        if j_geo_col_subscript != "":
            j_geo_col = j_geo_col.replace("j__", "j_" + j_geo_col_subscript + "__")
        geo_value__mm2 = data_ec.join(exp_ec.loc[:, geo_col], on="id_exp_sfc").loc[
            :, geo_col
        ]
        data_ec.loc[:, j_geo_col] = (
            data_ec.loc[:, I_col__A] * 1000 / (geo_value__mm2 / 100)
        )

    return data_ec


def gravimetric_current(exp_ec,
                        data_ec,
                        j_geo_cols=None,
                        j_geo_col=None):
    """
    Add loading and composition columns to exp_ec_expanded and mass normalized current to data_ec.
    Using loading column in spots/sample table and composition given in samples_composition/spots_composition
    :param exp_ec: experimental dataframe as received by exp_ec_expanded
    :param data_ec: experimental dataframe as received by data_ec
    :param j_geo_cols: str of list of str or None
        Selected a specific column for geometric area which will be used to normalize by mass.
        Optional, if None procedure is done for all geometric columns.
    :param j_geo_col: str of list of str or None
        Deprecated, use j_geo_cols
    :return: exp_ec, data_ec
        with appended columns for loading and gravimetric current density
    """
    #return plot.get_j__mA_mg(
    #    exp_ec,
    #    data_ec,
    #    j_geo_col,
    #)
    if j_geo_col is not None and j_geo_cols is None:
        print("\x1b[33m", "j_geo_col is deprecated. Please use the parameter j_geo_cols.", "\x1b[0m")
        j_geo_cols = j_geo_col
    j_geo_cols = tools.check_type(
        "j_geo_cols",
        j_geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    con = db.connect()

    # Work-around for: id_sample or id_spot in exp_ec index columns (in overlay example)
    id_sample_col = 'id_sample'
    id_spot_col = 'id_spot'
    temporary_cols = []

    def add_col_from_index_col(col_name, temporary_cols):
        if col_name in exp_ec.index.names:
            new_col = 'copy_'+col_name
            if new_col in exp_ec.columns:
                raise Exception('Column '+ new_col +
                                ' already exists in exp_ec. This can lead to undesired behavior, please remove.')
            exp_ec.loc[:, new_col] = exp_ec.index.get_level_values(level=col_name)
            temporary_cols += [new_col]
            return new_col, temporary_cols
        else:
            return col_name, temporary_cols

    id_sample_col, temporary_cols = add_col_from_index_col(id_sample_col, temporary_cols)
    id_spot_col, temporary_cols = add_col_from_index_col(id_spot_col, temporary_cols)

    # total loading from spots (preferred) or samples table
    for index, row in exp_ec.iterrows():
        exp_ec.loc[index, "total_loading__mg_cm2"] = (
            row.spots_total_loading__mg_cm2
            if row.spots_total_loading__mg_cm2 is not None
            else row.samples_total_loading__mg_cm2
        )
    # exp_ec.loc[:, ['total_loading__mg_cm2', 'spots_total_loading__mg_cm2','samples_total_loading__mg_cm2']]

    # get compositions
    sample_spot_composition = pd.read_sql(
        """ SELECT id_sample AS '""" + id_sample_col + """', 
                    NULL AS '""" + id_spot_col + """',
                    material,
                    wt_percent 
            FROM samples_composition 
            WHERE id_sample IN ("""
        + str(list(exp_ec.loc[:, id_sample_col].to_list()))[1:-1]
        + """)
            UNION
            SELECT id_sample AS '""" + id_sample_col + """',
                    id_spot AS '""" + id_spot_col + """',
                    material, 
                    wt_percent  
            FROM spots_composition 
            WHERE (id_sample, id_spot) IN ("""
        + str([tuple([row[id_sample_col], row[id_spot_col]]) for index, row in exp_ec.iterrows()])[1:-1]
        + """)
                               ;""",
        con=con,
        index_col=[id_sample_col, id_spot_col],
    )
    # display(sample_spot_composition)

    # for each material given, get values and write composition and loading into exp_ec
    materials = sample_spot_composition.material.unique()
    material_composition_cols = "wt_percent_" + materials
    material_loading_cols = "loading__mg_" + materials + "_cm2"
    for material, material_colname in zip(materials, material_composition_cols):
        # exp_ec.loc[:, material_colname] = None #default #Not necessary?

        # composition of materials for each experiment for each spot
        exp_ec_spots_composition = exp_ec.loc[:, [id_sample_col, id_spot_col]].join(
            sample_spot_composition.loc[
                (sample_spot_composition.loc[:, "material"] == material)
                & (sample_spot_composition.index.get_level_values(id_spot_col).notna()),
                :,
            ]
            .reset_index()
            .set_index([id_sample_col, id_spot_col])
            .loc[
                :,
                [
                    "wt_percent",
                ],
            ]
            .rename(columns={"wt_percent": material_colname}),
            on=[id_sample_col, id_spot_col],
        )
        # display(exp_ec_spots_composition)

        # composition of materials for each experiment for each sample
        exp_ec_samples_composition = exp_ec.loc[:, [id_sample_col, id_spot_col]].join(
            sample_spot_composition.loc[
                (sample_spot_composition.loc[:, "material"] == material)
                & (sample_spot_composition.index.get_level_values(id_spot_col).isna()),
                :,
            ]
            .reset_index()
            .set_index([id_sample_col])
            .loc[
                :,
                [
                    "wt_percent",
                ],
            ]
            .rename(columns={"wt_percent": material_colname}),
            on=[id_sample_col],
        )
        # display(exp_ec_samples_composition)

        # combine spots and sample-specific composition, thereby spots-specific is preferred over sample-specific
        # display(exp_ec_spots_composition.combine_first(exp_ec_samples_composition))
        exp_ec = exp_ec.join(
            exp_ec_spots_composition.combine_first(exp_ec_samples_composition).drop(
                columns=[id_sample_col, id_spot_col]
            ),
            on=exp_ec.index.names,
        )  # ['id_exp_sfc'])
        exp_ec.loc[:, material_loading_cols] = (
            exp_ec.loc[:, "total_loading__mg_cm2"]
            * exp_ec.loc[:, material_colname]
            / 100
        )

    # get all j_geo columns available in given exp_ec and j_geo_col not given
    if j_geo_cols is None:
        j_geo_cols = [col for col in data_ec.columns if col[:12] == "j__mA_cm2geo"]

    # print(j_geo_cols)

    # calculate mass normalized current
    data_ec = data_ec.join(
        exp_ec.loc[:, ["total_loading__mg_cm2"] + list(material_loading_cols)],
        on=exp_ec.index.names,
    )
    # display(data_ec.total_loading__mg_cm2)
    for col in j_geo_cols:
        if col not in data_ec.columns:
            warnings.warn(f'Column: "{col}" not available in data_ec. Will be skipped.')
            continue
        # print(col, '_geo' + col[12:])
        #print(col)
        #display(data_ec.loc[:, col], data_ec.total_loading__mg_cm2)
        data_ec.loc[:, "j__mA_mg_total" + "_geo" + col[12:]] = (
            data_ec.loc[:, col] / data_ec.total_loading__mg_cm2
        )
        for material, material_loading_colname in zip(materials, material_loading_cols):
            data_ec.loc[:, "j__mA_mg_" + material + "_geo" + col[12:]] = (
                data_ec.loc[:, col] / data_ec.loc[:, material_loading_colname]
            )
    data_ec = data_ec.drop(columns=["total_loading__mg_cm2"])

    # Remove temporary cols defined at the beginning of function
    for temporary_col in temporary_cols:
        exp_ec.drop(columns=temporary_col,inplace=True)

    return exp_ec, data_ec


def derive_ECSA(
    exp_ec,
    data_ec,
    method="Pt_Hupd_horizontal",
    geo_cols=None,
    Q_spec__C_cm2=None,
    display_result_plot=True,
):
    """
    Derive the ECSA from experimental data by different methods.
    :param exp_ec: experimental dataframe as received by exp_ec_expanded
    :param data_ec: experimental dataframe as received by data_ec
    :param method: method of the derivation to be used
    :param geo_cols: geometric colum to be used to derive roughness factor
    :param Q_spec__C_cm2: specific charge to be used to evaluate the specific area.
        On Pt for Hupd default of 210e-6 is used. (e.g.: https://doi.org/10.20964/2016.06.71)
    :param display_result_plot: bool
        whether to display result plot
    :return: exp_ec, data_ec
        with appended columns for surface-specific charge, specific electrode area, ECSA,
        and surface specific current density
    """
    if geo_cols is None:
        geo_cols = plot.geo_columns.loc[:, "A_geo"]
    elif type(geo_cols) == str:
        geo_cols = [geo_cols]

    if len(exp_ec.index) > 1:
        # raise Exception('derivation of ECSA for multiple experiments not yet developed.')
        exp_ec_return = pd.DataFrame()
        data_ec_return = pd.DataFrame()
        for index, row in exp_ec.iterrows():
            exp_ec_single = exp_ec.loc[
                [
                    index,
                ]
            ]
            data_ec_single = data_ec.loc[
                [
                    index,
                ]
            ]
            exp_ec_single, data_ec_single = derive_ECSA(
                exp_ec_single,
                data_ec_single,
                method=method,
                geo_cols=geo_cols,
                Q_spec__C_cm2=Q_spec__C_cm2,
            )

            exp_ec_return = pd.concat([exp_ec_return, exp_ec_single])
            data_ec_return = pd.concat([data_ec_return, data_ec_single])
        return exp_ec_return, data_ec_return

    if len(data_ec.loc[(exp_ec.index, slice(None)), "cycle"].unique()) > 1:
        raise Exception("derivation of ECSA for multiple cycles not yet developed.")

    if method == "Pt_Hupd_horizontal":
        print(
            "\x1b[33m",
            "This is just a basic implementation of ECSA calculation for single CV cycle on "
            "Pt with constant capacitive current correction in acidic electrolyte.",
            "Be aware analysis procedure need to be adjusted for different electrode material, electrolyte, ...",
            "\x1b[0m",
        )
        selection_forward_scan = ((data_ec.E_WE__VvsRHE.diff() > 0)
                                  # E_WE__VvsRHE can fail with too high compensation
                                  & (data_ec.E_Signal__VvsRE.diff() > 0)
                                  & (~data_ec.index.isin(list(data_ec.reset_index().groupby(['id_exp_sfc']).first().id_data_ec.items())))
                                 )
        data_ec.loc[:, "selection_derive_minimum"] = (selection_forward_scan
                                                      & (data_ec.E_WE__VvsRHE > 0.2)
                                                      )

        data_ec.loc[:, "selection_derive_intersection"] = (selection_forward_scan
                                                           & (data_ec.E_WE__VvsRHE < 0.2)
                                                           )

        index_minimum = data_ec.loc[
            data_ec.selection_derive_minimum, :
        ].I__A.idxmin()
        iloc_minimum = (
            data_ec.reset_index().loc[data_ec.index == index_minimum, :].index[0]
        )
        number_datapoints_avg_minimum = 11
        data_ec.loc[:, "selection_minimum"] = data_ec.index.isin(
            data_ec.iloc[
                iloc_minimum - int((number_datapoints_avg_minimum - 1) / 2):
                iloc_minimum + int((number_datapoints_avg_minimum - 1) / 2 + 1)
            ].index
        )
        I_capacitive__A = data_ec.loc[data_ec.selection_minimum, :].I__A.mean()

        index_intersect = data_ec.loc[
            (
                selection_forward_scan
                & (data_ec.E_WE__VvsRHE < 0.2)
                & (data_ec.I__A > I_capacitive__A)
            ),
            :,
        ].index[0]
        # first datapoint above capacitive current
        data_ec.loc[:, "selection_derive_ecsa"] = data_ec.index.isin(
            data_ec.loc[index_intersect:index_minimum, :].index
        )
        data_ec.loc[
            data_ec.selection_derive_ecsa, "I_capacitive__A"
        ] = I_capacitive__A

        data_ec = geometric_current(
            exp_ec,
            data_ec,
            geo_cols=geo_cols,
            j_geo_col_subscript="capacitive",
            I_col__A="I_capacitive__A",
        )  # calculate geometric current density

        exp_ec.loc[:, "Q_ECSA__C"] = np.trapz(
            data_ec.loc[data_ec.selection_derive_ecsa, :].I__A,
            x=data_ec.loc[data_ec.selection_derive_ecsa, :].t__s,
        ) - np.trapz(
            data_ec.loc[data_ec.selection_derive_ecsa, :].I_capacitive__A,
            x=data_ec.loc[data_ec.selection_derive_ecsa, :].t__s,
        )
        Q_spec_Hupd__C_cm2 = 210 * 10**-6  # default value
        exp_ec.loc[:, "Q_spec__C_cm2"] = (
            Q_spec_Hupd__C_cm2 if Q_spec__C_cm2 is None else Q_spec__C_cm2
        )

        exp_ec.loc[:, "ECSA_method"] = method
        exp_ec.loc[:, "A_spec__m2"] = exp_ec.Q_ECSA__C / exp_ec.Q_spec__C_cm2 / 10**4
        exp_ec.loc[:, "A_spec__mm2"] = exp_ec.A_spec__m2 * 1e6
        data_ec.loc[:, "j__mA_cm2spec_Pt_hupd"] = (
            data_ec.loc[:, "I__A"] * 1000 / (exp_ec.A_spec__mm2.iloc[0] / 100)
        )

        # print('A_spec = ', A_spec__m2, ' m²')
        for geo_col in geo_cols:
            exp_ec.loc[:, "roughness_factor_" + geo_col] = (
                exp_ec.A_spec__m2 / exp_ec.loc[:, geo_col] * 1e6
            )

        if 'loading__mg_Pt_cm2' in exp_ec.columns:
            if ~exp_ec.loading__mg_Pt_cm2.isna().all():
                exp_ec.loc[:, "ECSA__m2_gPt"] = exp_ec.A_spec__m2 / (
                    exp_ec.loading__mg_Pt_cm2 / 1000 * exp_ec.spots_spot_size__mm2 / 100
                )
                # print('ECSA-Hupd = ', ECSA, ' m²/gPt')

        if display_result_plot:
            data_ec_selection_derive_minimum = (
                data_ec.loc[data_ec.selection_derive_minimum, :]
                .sort_values(by="E_WE__VvsRHE")
                .reset_index()
                .reset_index()
                .rename(columns={"index": "id_data_ec_sorted"})
                .set_index(["id_exp_sfc", "id_data_ec_sorted"])
                .copy()
            )
            data_ec_selection_minimum = (
                data_ec.loc[data_ec.selection_minimum, :]
                .sort_values(by="E_WE__VvsRHE")
                .reset_index()
                .reset_index()
                .rename(columns={"index": "id_data_ec_sorted"})
                .set_index(["id_exp_sfc", "id_data_ec_sorted"])
                .copy()
            )
            data_ec_selection_derive_ecsa = (
                data_ec.loc[data_ec.selection_derive_ecsa, :]
                .sort_values(by="E_WE__VvsRHE")
                .reset_index()
                .reset_index()
                .rename(columns={"index": "id_data_ec_sorted"})
                .set_index(["id_exp_sfc", "id_data_ec_sorted"])
                .copy()
            )

            with plt.rc_context(
                plot.get_style(
                    style="singleColumn",
                    fig_size={"width": 6, "height": 4},
                    add_margins_and_figsize={
                        "left": -0.3,
                        "bottom": -0.3,
                    },
                )
            ):
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                # ax1.yaxis.set_tick_params(which='both', labelleft=False)

                exp_ec.dataset.plot(
                    x_col="E_WE__VvsRHE",
                    y_col="I__A",
                    data=data_ec,
                    ax=ax1,
                    label="raw data",
                    color="tab:blue",
                ).plot(
                    x_col="E_WE__VvsRHE",
                    y_col="I__A",
                    data=data_ec_selection_derive_minimum,
                    ax=ax1,
                    label="datapoints to derive minimum",
                    color="tab:orange",
                ).plot(
                    x_col="E_WE__VvsRHE",
                    y_col="I__A",
                    data=data_ec_selection_minimum,
                    ax=ax1,
                    label="datapoints of minimum",
                    color="tab:red",
                ).plot(
                    x_col="E_WE__VvsRHE",
                    y_col="I__A",
                    data=data_ec_selection_derive_ecsa,
                    ax=ax1,
                    label="datapoints to derive ecsa",
                    color="tab:red",
                    linestyle="--",
                ).plot(
                    x_col="E_WE__VvsRHE",
                    y_col="I_capacitive__A",
                    data=data_ec_selection_derive_ecsa,
                    ax=ax1,
                    label="capacitive current",
                    color="tab:red",
                    linestyle="-.",
                    axlabel_auto=False,
                ).fill_between(
                    x_col="E_WE__VvsRHE",
                    y_col="I_capacitive__A",
                    y2_col="I__A",
                    data=data_ec_selection_derive_ecsa,
                    ax=ax1,
                    label="$Q_\mathrm{ECSA}\ =$ %.2E C"
                    % (exp_ec.loc[:, "Q_ECSA__C"].iloc[0]),
                    color="tab:blue",
                    alpha=0.6,
                    axlabel_auto=False,
                ).return_dataset()

                # ax1.set_ylabel("$j_\\mathrm{geo}$ / mA cm$^{-2}$")
                # ax1.set_xlabel("$E$ / V vs. RHE")

                legend = ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.2))
                legend.get_frame().set_alpha(0)
                # plot_storage.export(fig)

                plt.show()

    else:
        raise Exception("Method: " + method + " not yet implemented")

    return exp_ec, data_ec


def get_derived_ECSA(
    sql_ec,
    cycle=2,
    match_eis_id_ML=None,
    match_eis_timedelta__h=3,
    user_control_eis=True,
    geo_cols="fc_top_name_flow_cell_A_opening_ideal__mm2",
    method="Pt_Hupd_horizontal",
    Q_spec__C_cm2=None,
    display_result_plot=True,
):
    """
    Get ECSA for an exp_ec derived from an other ec experiment defined via sql_ec
    :param sql_ec: define experiment from which to derive ECSA from
    :param cycle: cycle of the experiment from whcih to derive ECSA from
    :param match_eis_id_ML: update_R_u__ohm parameter to derive compensated potentials
    :param match_eis_timedelta__h: update_R_u__ohm parameter to derive compensated potentials
    :param user_control_eis: update_R_u__ohm parameter to derive compensated potentials
    :param geo_cols: derive_ECSA parameter
    :param method: derive_ECSA parameter
    :param Q_spec__C_cm2: derive_ECSA parameter
    :param display_result_plot: derive_ECSA parameter
    :return: exp_ec, data_ec of the experiments from which ECSA is derived
    """
    exp_ec = db.get_exp(sql_ec)

    exp_ec = update_R_u__ohm(
        exp_ec,
        match_eis_id_ML=match_eis_id_ML,
        match_eis_timedelta__h=match_eis_timedelta__h,
        user_control=user_control_eis,
    )  # derive uncompensated resistance from EIS experiemtn performed in id_ML=61

    data_ec = db.get_data(exp_ec, "data_ec", add_cond="cycle IN (%s)" % (int(cycle)))
    data_ec = compensate_R_u(exp_ec, data_ec)  # calculate compensated potentials
    data_ec = geometric_current(
        exp_ec, data_ec, geo_cols=geo_cols
    )  # calculate geometric current density
    exp_ec, data_ec = gravimetric_current(
        exp_ec,
        data_ec,
        # j_geo_cols='j_geo__mA_cm2geo_spot_size',
    )

    exp_ec, data_ec = derive_ECSA(
        exp_ec,
        data_ec,
        method=method,
        geo_cols=geo_cols,
        Q_spec__C_cm2=Q_spec__C_cm2,
        display_result_plot=display_result_plot,
    )

    return exp_ec, data_ec


def polcurve_analysis(
        name_user,
        name_setup_sfc,
        date,
        id_ML,
        add_cond="",
        number_datapoints_in_tail=40,
        j_geo_col="j__mA_cm2geo_fc_bottom_PTL",
        skip_id_data_eis_greater=2,
        figure_dpi=150,
        ignore_ghold_geis_not_matching=False,
        ignore_overload_n_first_datapoints=0,
        tafel_fit_method='scipy.odr',
        display_compression=True,
        display_hysteresis=True,
):
    """
    Perform analysis of a galvanostatic step polarization curve including
        - HFR correction
        - derivation of stabilized potential and current (using number_datapoints_in_tail)
        - plotting of polarization curve and tafel plot
        - tafel slop determination (using tafel_fit_method)
    :param name_user: choose username
    :param name_setup_sfc: choose sfc setup
    :param date: chosse date
    :param id_ML: choose id_ML
    :param add_cond: add another condition to exp_sfc select statement
    :param number_datapoints_in_tail: number of datapoints at the of the ghold used to average current and potential
    :param j_geo_col: current column displayed in analysis plot, in database always absolute currents are stored
    :param skip_id_data_eis_greater: it was observed that usually the first one or two datapoints of EIS measurement
    are not usable which is why they are discarded by default.
    :param ignore_overload_n_first_datapoints: int, default 0
        In eCat 5.7 the first datapoints of ghold have an overload error.
        To keep automatic detection of gooddata these overload errors can be ignored.
    :param figure_dpi: dpi of plotted figure
    :param ignore_ghold_geis_not_matching: bool
        ignore although ghold and geis not matching. Set to True, if your protocol does not follow ghold+geis for each
        step. However functionalities cannot be guaranteed
    :param tafel_fit_method: str, one of ['scipy.optimize.curve_fit', 'scipy.odr'], default 'scipy.odr'
        Select the method used to fit the tafel plot. Parameter as panda.DataFrame().dataset.fit(method).
    :param display_compression: bool, default True
        whether to search for and if recorded add z-position and force data during polcurve measurement
    :return: None
    """
    engine = db.connect(user="hte_polcurve_inserter")
    export_folder = db_config.DIR_REPORTS() / Path("02_polcurve_reports/")

    add_cond = ("AND " + add_cond) if add_cond != "" else ""
    exp_ec = db.get_exp(
        by="""SELECT  *
                              FROM exp_ec_expanded
                              WHERE name_user= %s
                                AND name_setup_sfc = %s
                                AND DATE(t_start__timestamp) = %s
                                AND id_ML = %s
                                AND (geis_I_dc__A > 0 
                                    OR ghold_I_hold__A > 0
                                    OR id_exp_sfc IN (SELECT id_exp_sfc_ghold AS id_exp_sfc FROM data_ec_polcurve
                                                        UNION
                                                       SELECT id_exp_sfc_geis AS id_exp_sfc FROM data_ec_polcurve)
                                    )
                            """
           + add_cond
           + """ 
                            ;""",
        params=[name_user, name_setup_sfc, date, id_ML],
    )  # , con=con, index_col='id_exp_sfc')
    if len(exp_ec.index) == 0:
        raise Exception(
            "No experiments found for given name_user, name_setup_sfc, date, id_ML"
        )

    if exp_ec.name_user.iloc[0] != db.current_user():
        print(
            "\x1b[33m",
            "View-only: This is data from ",
            exp_ec.name_user.iloc[0],
            ". Updates are restricted!",
            "\x1b[0m",
        )

    data_eis = db.get_data(
        exp_ec,
        "data_eis_analysis",
        add_cond="id_data_eis> " + str(skip_id_data_eis_greater),
    )
    data_ec = db.get_data_ec(
        exp_ec,
        # "data_ec_analysis",
    )

    # Compression data
    if display_compression:
        exp_compression = db.get_exp(exp_ec, name_table='exp_compression_expanded')
    if len(exp_compression.index) == 0:
        print(
            "\x1b[33m",
            "No compression data found."
            "\x1b[0m",
        )
        display_compression = False
    if display_compression:
        data_compression = db.get_data(exp_compression, name_table='data_compression')

    polcurve_existing = db.query_sql(
        """SELECT id_exp_ec_polcurve
                                FROM exp_ec_polcurve_expanded
                                WHERE name_user= %s
                                     AND name_setup_sfc = %s
                                     AND DATE(t_start__timestamp) = %s
                                     AND id_ML = %s
                                     ;""",
        params=[name_user, name_setup_sfc, date, id_ML],
    ).values
    if len(polcurve_existing) == 0:
        id_exp_ec_polcurve = "new"
    elif len(polcurve_existing) == 1:
        id_exp_ec_polcurve = int(polcurve_existing[0][0])  # database cant handle np.int64
        print('Polcurve already inserted with id_exp_ec_polcurve = %s' % id_exp_ec_polcurve)
    else:
        sys.exit(
            "Error multiple polarization curve experiments found. Please inform Nico."
        )

    if not all(exp_ec.ec_name_technique.isin(["exp_ec_ocp", "exp_ec_ghold", "exp_ec_geis"])):
        warnings.warn(
            "techniques other than exp_ec_ocp ,exp_ec_ghold, and exp_ec_geis will be ignored"
        )

    # Check for equality of experimental parameters throughout polarization curve
    cols_value_difference_accepted = [
        "t_start__timestamp",
        "id_ML_technique",
        "t_duration__s",
        "t_end__timestamp",
        "ec_name_technique",
        "geis_I_dc__A",
        "ghold_I_hold__A",
        "ghold_t_hold__s",
        "linaxis_z__mm",
        "ec_id_ie_range",
        "geis_I_amplitude__A",
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
    """ # old algorithms fails for MLs with gholds without geis (activation ICP-MS)
    exp_ec_ghold = (
        exp_ec.loc[
            exp_ec.ec_name_technique.isin(["exp_ec_ghold", "exp_ec_ocp"]),
            [
                "ghold_I_hold__A",
            ],
        ]
        .reset_index()
        .rename(columns={"id_exp_sfc": "id_exp_sfc_ghold"})
    )

    # exp_ec_ocp technique has ghold_I_hold__A = nan
    exp_ec_ghold.loc[:, 'ghold_I_hold__A'] = exp_ec_ghold.ghold_I_hold__A.fillna(0)

    exp_ec_geis = (
        exp_ec.loc[exp_ec.ec_name_technique.isin(["exp_ec_geis"]), :]
        .reset_index()
        .rename(columns={"id_exp_sfc": "id_exp_sfc_geis"})
    )
    exp_ec_sgeis = exp_ec_ghold.join(exp_ec_geis, rsuffix="geis")
    """
    exp_ec_match_ghold_geis = exp_ec.reset_index().copy()
    exp_ec_match_ghold_geis = exp_ec_match_ghold_geis.join(
        exp_ec_match_ghold_geis.loc[:, ['id_exp_sfc', 'ec_name_technique']].shift(-1),
        rsuffix='_next') \
                                  .loc[:,
                              ['id_exp_sfc', 'ec_name_technique', 'id_exp_sfc_next', 'ec_name_technique_next']]
    exp_ec_sgeis = exp_ec_match_ghold_geis.loc[((exp_ec_match_ghold_geis.ec_name_technique == 'exp_ec_ghold')
                                                & (exp_ec_match_ghold_geis.ec_name_technique_next == 'exp_ec_geis')
                                                ), :] \
        .rename(columns={"id_exp_sfc": "id_exp_sfc_ghold"}) \
        .rename(columns={"id_exp_sfc_next": "id_exp_sfc_geis"}) \
        .join(exp_ec.ghold_I_hold__A, on='id_exp_sfc_ghold') \
        .join(exp_ec, on='id_exp_sfc_geis', rsuffix='_geis') \
        .copy()

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
            if not ignore_ghold_geis_not_matching:
                print(
                    "\x1b[31m",
                    "something went wrong with merging ghold and geis experiments. "
                    "Set ignore_ghold_geis_not_matching=True to ignore",
                    "\x1b[0m",
                )
                raise Exception("")
            else:
                print(
                    "\x1b[33m",
                    "something went wrong with merging ghold and geis experiments. But its ignored.",
                    "\x1b[0m",
                )

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
    if ignore_overload_n_first_datapoints != 0:
        df_ignored_overloads = data_ec.groupby("id_exp_sfc") \
            .head(n=ignore_overload_n_first_datapoints)[["overload", ]] \
            .loc[lambda row: row.overload != 8191]
        if len(df_ignored_overloads) > 0:
            print('Overloads during ignore_overload_n_first_datapoints:')
            display(df_ignored_overloads.dropna())
    exp_ec_sgeis = exp_ec_sgeis.join(
        data_ec.groupby("id_exp_sfc").apply(lambda x: x.iloc[ignore_overload_n_first_datapoints:]) \
            .groupby("id_exp_sfc")["overload"].unique().astype(str),
        on="id_exp_sfc_ghold",
    ).rename(columns={"overload": "overload_list"})

    if id_exp_ec_polcurve == "new":
        # derive R_u guess form minimum
        # exp_ec_sgeis = exp_ec_sgeis.join \
        #    (data_eis.loc[data_eis.loc[:, 'minusZ_img__ohm'].groupby(level=0).idxmin(), :].reset_index
        #     (level=1).id_data_eis, on='id_exp_sfc_geis').rename(columns={'id_data_eis': 'id_data_eis_chosen_Ru'})
        exp_ec_sgeis = derive_HFR(exp_ec_sgeis, data_eis, on="id_exp_sfc_geis", show_control_plot=False).drop(
            columns=["R_u__ohm"]
        )

        if exp_ec_sgeis.id_data_eis_chosen_Ru.isna().any():
            print(
                "\x1b[31m",
                "HFR could not be detected. First datapoint is selected. Be aware this might not be correct."
                "\x1b[0m",
            )
            exp_ec_sgeis.loc[exp_ec_sgeis.id_data_eis_chosen_Ru.isna(), 'id_data_eis_chosen_Ru'] = 0

        # calculate default gooddata
        exp_ec_sgeis.loc[:, "gooddata"] = (
            exp_ec_sgeis.loc[:, "overload_list"]
                .apply(lambda x: True if len(x.split(" ")) == 1 else False)
                .astype(np.bool_)
        )
        # derive tafel fit limits from ec data
        # (will give a bit larger window than from data_ec_polcurve as in that case only green data is considered)
        tafel_fit_limits = [
            data_ec.loc[:, j_geo_col].abs().min(),
            data_ec.loc[:, j_geo_col].abs().max(),
        ]
    else:
        with engine.begin() as con:
            # Get R_u__ohm and gooddata from database
            existing_data = db.query_sql(
                """SELECT id_exp_sfc_geis, id_data_eis_chosen_Ru, R_u_derived_by, gooddata
                                             FROM data_ec_polcurve
                                             WHERE id_exp_ec_polcurve=%s
                                                  ;""",
                con=con,
                params=[id_exp_ec_polcurve],
                method="pandas",
                index_col="id_exp_sfc_geis",
            )

            if existing_data.R_u_derived_by.isna().any():
                exp_ec_sgeis_dummy = (
                    derive_HFR(exp_ec_sgeis, data_eis, on="id_exp_sfc_geis")
                        .drop(columns=["R_u__ohm"])
                        .set_index("id_exp_sfc_geis")
                )
                equal_chosen_and_algorithm_Ru = (
                        existing_data.id_data_eis_chosen_Ru
                        == exp_ec_sgeis_dummy.id_data_eis_chosen_Ru
                )
                existing_data.loc[
                    ~equal_chosen_and_algorithm_Ru, "R_u_derived_by"
                ] = "manual"
                existing_data.loc[
                    equal_chosen_and_algorithm_Ru, "R_u_derived_by"
                ] = exp_ec_sgeis_dummy.loc[
                    equal_chosen_and_algorithm_Ru, "R_u_derived_by"
                ]

            exp_ec_sgeis = exp_ec_sgeis.join(existing_data, on="id_exp_sfc_geis")
            exp_ec_sgeis.loc[
                ~exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"
            ] = exp_ec_sgeis.loc[
                ~exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"
            ].astype(
                np.bool_
            )
            exp_ec_sgeis.loc[exp_ec_sgeis.loc[:, "gooddata"].isna(), "gooddata"] = (
                exp_ec_sgeis.loc[
                    exp_ec_sgeis.loc[:, "gooddata"].isna(), "overload_list"
                ]
                    .apply(lambda x: True if len(x.split(" ")) == 1 else False)
                    .astype(np.bool_)
            )

            # Get Tafel limits
            tafel_fit_limits = (
                db.query_sql(
                    """SELECT tafel_fit_left_limit__j_geo, tafel_fit_right_limit__j_geo
                                                FROM exp_ec_polcurve
                                                WHERE id_exp_ec_polcurve = %s
                                              ;""",
                    con=con,
                    params=[id_exp_ec_polcurve],
                    method="pandas",
                )
                    .iloc[0]
                    .values
            )
            # updating old values without tafel fit
            if tafel_fit_limits[0] is None:
                tafel_fit_limits[0] = data_ec.loc[:, j_geo_col].abs().min()
            if tafel_fit_limits[1] is None:
                tafel_fit_limits[1] = data_ec.loc[:, j_geo_col].abs().max()

        # display(exp_ec_sgeis)
        # print(type(exp_ec_sgeis.gooddata.iloc[0]))
    exp_ec_sgeis.loc[:, "id_data_eis_chosen_Ru_init"] = exp_ec_sgeis.loc[
                                                        :, "id_data_eis_chosen_Ru"
                                                        ]

    # time synchronisattion of eis data
    data_eis = data_eis.join(exp_ec.t_start__timestamp, on="id_exp_sfc")
    data_eis.loc[data_eis.Timestamp.isna(), "Timestamp"] = data_eis.loc[
        data_eis.Timestamp.isna(), "t_start__timestamp"
    ]
    data_eis.loc[:, "t_synchronized__s"] = (
            pd.to_datetime(data_eis.Timestamp) - pd.to_datetime(data_ec.Timestamp).min()
    ).dt.total_seconds()
    data_ec.loc[:, "t_synchronized__s"] = (
            pd.to_datetime(data_ec.Timestamp) - pd.to_datetime(data_ec.Timestamp).min()
    ).dt.total_seconds()  # .dtRu_calculated.loc[:, 't_start__timestamp'].to_timestamp().min()
    if display_compression:
        data_compression.loc[:, "t_synchronized__s"] = (
                pd.to_datetime(data_compression.Timestamp) - pd.to_datetime(data_ec.Timestamp).min()
        ).dt.total_seconds()

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
                :, ~data_ec_old.columns.isin(["overload", "cycle", "t_synchronized__s", "Timestamp"])
                ]
                    .groupby("id_exp_sfc")
                    .mean()
            )
                .join(
                data_ec_polcurve_raw_new.loc[
                :, ~data_ec_old.columns.isin(["overload", "cycle", "t_synchronized__s", "Timestamp"])
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

        # calculate compensated potential:
        data_ec_polcurve_new.loc[:, "E_WE__VvsRHE"] = data_ec_polcurve_new.loc[
                                                      :, "E_WE_uncompensated__VvsRHE"
                                                      ] - (data_ec_polcurve_new.loc[:,
                                                           "I__A"] * data_ec_polcurve_new.loc[:, "R_u__ohm"])
        # write corresponding id_exp_ec_polcurve
        data_ec_polcurve_new.loc[
        :, "id_exp_ec_polcurve"
        ] = exp_ec_polcurve.reset_index().id_exp_ec_polcurve.iloc[
            0
        ]  # id_exp_ec_polcurve

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

    # For hysteresis plot
    def avg_for_back_polcurve(data_ec_polcurve_input):
        # Error: joining on floats not possible --> transform to integer with specified resolution (nA should be sufficient)
        data_ec_polcurve_input.loc[:, 'ghold_I_hold__nA_integer'] = (
                    data_ec_polcurve_input.ghold_I_hold__A * 1e9).astype(int)

        on_for_back_cols = ['id_exp_ec_polcurve', 'ghold_I_hold__nA_integer']
        data_ec_polcurve_avg_for_back_new \
            = data_ec_polcurve_input.loc[data_ec_polcurve_input.scan_direction == 1] \
            .join(data_ec_polcurve_input.loc[data_ec_polcurve_input.scan_direction == 2]
                  .reset_index()
                  .set_index(on_for_back_cols)
                  .rename(columns={'id_data_ec_polcurve': 'id_data_ec_polcurve_backward', }),
                  on=on_for_back_cols,
                  lsuffix='_forward',
                  rsuffix='_backward', )
        data_ec_polcurve_avg_for_back_new = pd.concat([data_ec_polcurve_avg_for_back_new,
                                                       ((data_ec_polcurve_avg_for_back_new.loc[:,
                                                         j_geo_col + '_forward'] + data_ec_polcurve_avg_for_back_new.loc[
                                                                                   :,
                                                                                   j_geo_col + '_backward']) / 2).rename(
                                                           j_geo_col),
                                                       ((
                                                                    data_ec_polcurve_avg_for_back_new.E_WE__VvsRHE_backward - data_ec_polcurve_avg_for_back_new.E_WE__VvsRHE_forward) * 1000).rename(
                                                           'DeltaE_WE_avg_for_back__mV'),
                                                       (data_ec_polcurve_avg_for_back_new.loc[:,
                                                        ["gooddata_forward", "gooddata_backward"]].all(axis=1)).rename(
                                                           'gooddata'),
                                                       ],
                                                      axis=1)
        return data_ec_polcurve_avg_for_back_new

    # Add columns to data_ec: Normalize uncompensated potential
    j_geo_col_normalized = j_geo_col + '_normalized'
    j_geo_col_step_mean = j_geo_col + '_step_mean'
    data_ec_normalized = data_ec.loc[:, ['t_synchronized__s', 'E_WE_uncompensated__VvsRHE', j_geo_col]] \
        .join(data_ec_polcurve.set_index('id_exp_sfc_ghold') \
              .loc[:, ['E_WE_uncompensated__VvsRHE', j_geo_col]] \
              .rename(columns={'E_WE_uncompensated__VvsRHE': 'E_WE_uncompensated_step_mean__VvsRHE',
                               j_geo_col: j_geo_col_step_mean}),
              on='id_exp_sfc',
              ) \
        .assign(**{'E_WE_uncompensated_step_normalized__VvsRHE': lambda
        x: x.E_WE_uncompensated__VvsRHE - x.E_WE_uncompensated_step_mean__VvsRHE,
                   j_geo_col_normalized: lambda x: x[j_geo_col] - x[j_geo_col_step_mean]}
                )
    data_ec_normalized_selected = data_ec_normalized.groupby('id_exp_sfc').tail(n=number_datapoints_in_tail)

    plot.manual_col_to_axis_label['delta_tafel_fit_residuals__mV'] = '$\Delta E_\mathrm{tafel\ fit\ residuals}$ / mV'
    plot.manual_col_to_axis_label[
        'E_WE_uncompensated_step_normalized__VvsRHE'] = '$E_\mathrm{uncompensated,\ step\ norm.}$ / V'
    plot.manual_col_to_axis_label[j_geo_col_normalized] = plot.get_geo_column_label(j_geo_col).replace('j_\mathrm{geo}',
                                                                                                       'j_\mathrm{geo,\ norm.}')
    plot.manual_col_to_axis_label['E_WE_drift_plot__mV_min'] = '$E_\mathrm{uncompensated,\ drift}$ / mV min$^{-1}$'
    plot.manual_col_to_axis_label['DeltaE_WE_avg_for_back__mV'] = '$\Delta E_\mathrm{for\ back\ sweep}$ / mV'

    # '$\Delta E_\mathrm{step\ norm.}$ / V'

    # return exp_ec_sgeis, data_ec_polcurve_raw, data_eis
    with plt.rc_context(
            plot.get_style(
                style="doubleColumn",  # "singleColumn",
                interactive=True,
                increase_fig_height=3.5,
                add_margins_and_figsize={"right": 0, "left": 0, "bottom": 1},
                add_margins_between_subplots={"hspace": 6, "wspace": 6},
                add_params={"figure.dpi": figure_dpi},
                scale_fontsize=0.5,
            )
    ):
        plot_storage = plot.PlotDataStorage(
            "statistics_number_of_experiments", overwrite_existing=True
        )
        fig = plt.figure()
        # ax_EIS = fig.add_subplot(111)
        n_plots = 6
        n_plots += 2 if display_compression else 0
        n_plot = 0
        axs = []

        def twinx_to_left(ax):
            ax.spines["left"].set_position(("axes", -0.2))
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.yaxis.set_label_position('left')
            ax.yaxis.set_ticks_position('left')

        gs = gridspec.GridSpec(5, 3)  # n_plots, 1)

        ax_EIS = fig.add_subplot(gs[0, 0])  # n_plot, 0])
        ax_EIS.axis('equal')  # same magnitude for x and y axis - important for Nyquist plots
        ax_EIS.export_name = "ax_EIS"
        axs += [ax_EIS]
        n_plot += 1

        ax_IV_vs_t_potential = fig.add_subplot(gs[1, 0])  # n_plot, 0])
        ax_IV_vs_t_potential.export_name = "ax_IV_vs_t_leftyaxis"
        ax_IV_vs_t_current = ax_IV_vs_t_potential.twinx()
        # twinx_to_left(ax_IV_vs_t_current)
        ax_IV_vs_t_current.export_name = "ax_IV_vs_t_current_rightyaxis"
        axs += [ax_IV_vs_t_potential]
        n_plot += 1

        ax_IV_vs_t_normalized_potential = fig.add_subplot(gs[2, 0], sharex=ax_IV_vs_t_potential)  # n_plot, 0])
        ax_IV_vs_t_normalized_potential.export_name = "ax_IV_vs_t_normalized_potential"
        ax_IV_vs_t_normalized_current = ax_IV_vs_t_normalized_potential.twinx()
        # twinx_to_left(ax_IV_vs_t_current)
        ax_IV_vs_t_normalized_current.export_name = "ax_IV_vs_t_normalized_current_rightyaxis"
        axs += [ax_IV_vs_t_normalized_potential]
        n_plot += 1

        if display_compression:
            ax_compression_zpos = fig.add_subplot(gs[3, 0],  # [n_plot, 0],
                                                  sharex=ax_IV_vs_t_potential)
            ax_compression_zpos.export_name = "ax_compression_zpos"
            axs += [ax_compression_zpos]
            n_plot += 1

            ax_compression_force = fig.add_subplot(gs[4, 0],
                                                   sharex=ax_IV_vs_t_potential)
            ax_compression_force.export_name = "ax_compression_force"
            axs += [ax_compression_force]
            n_plot += 1

        ax_Ru_j = fig.add_subplot(gs[0, 1])
        ax_Ru_j.export_name = "ax_Ru_j"
        ax_Ru_j.set_xscale("log")
        axs += [ax_Ru_j]
        n_plot += 1

        ax_polcurve = fig.add_subplot(gs[1, 1])#, sharex=ax_Ru_j)
        ax_polcurve.export_name = "ax_polcurve"
        axs += [ax_polcurve]
        n_plot += 1

        ax_tafel = fig.add_subplot(gs[0, 2])
        ax_tafel.export_name = "ax_tafel"
        ax_tafel.set_xscale("log")
        axs += [ax_tafel]
        n_plot += 1

        ax_tafel_res = fig.add_subplot(gs[1, 2], sharex=ax_tafel)
        ax_tafel_res.export_name = "ax_tafel_res"
        ax_tafel_res.set_xscale("log")
        axs += [ax_tafel_res]
        n_plot += 1

        ax_drift_potential = fig.add_subplot(gs[2, 1], sharex=ax_tafel)
        ax_drift_potential.export_name = "ax_drift_potential"
        ax_drift_potential.set_xscale("log")
        # ax_drift_current = ax_drift_potential.twinx()
        # ax_drift_current.export_name = "ax_drift_current_rightyaxis"
        # ax_drift_potential.set_xscale("log")
        axs += [ax_drift_potential]
        n_plot += 1

        if display_hysteresis:
            display_hysteresis = data_ec_polcurve.scan_direction.max() == 2

        if display_hysteresis:
            ax_hysteresis_for_back = fig.add_subplot(gs[2, 2], sharex=ax_tafel)
            ax_hysteresis_for_back.export_name = "ax_hysteresis_for_back"
            ax_hysteresis_for_back.set_xscale("log")
            axs += [ax_hysteresis_for_back]
            n_plot += 1
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
                ax=ax_EIS,
                # alpha=0.2
            )
                .return_dataset()
                .reset_index()
        )
        # display(exp_ec_sgeis)
        (Ru_dots,) = ax_EIS.plot(
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

        # exp_ec_sgeis_ghold_label_helper = exp_ec_sgeis.set_index('id_exp_sfc_ghold').join(exp_ec, lsuffix='pol_curve')
        # .add_column('label',
        #                        values='id in ML: ' + exp_ec_sgeis_ghold_label_helper.loc[:, 'id_ML'].astype(str)
        #                               + ',   Spot: ' + exp_ec_sgeis_ghold_label_helper.loc[:, 'id_spot'].astype(str)
        #                               + ',   @ ' + (exp_ec_sgeis_ghold_label_helper.loc[:,
        #                                             'geis_I_dc__A'] * 1000).astype(str) + ' mA'
        #                        ) \
        exp_ec_sgeis = (
            exp_ec_sgeis.set_index("id_exp_sfc_ghold")
                .dataset.add_column(
                "label",
                values="$j_\mathrm{geo}}$",
                rowindexer_first_of_group="ec_name_technique",
            )
                .plot(
                x_col="t_synchronized__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax_IV_vs_t_current,
                color="gray",
            )
                .plot(
                x_col="t_synchronized__s",
                y_col=j_geo_col_normalized,
                data=data_ec_normalized,  #
                ax=ax_IV_vs_t_normalized_current,
                color="gray",
            )
                .add_column("color", values="pub_blues")
                .add_column(
                "label",
                values="$E_\mathrm{raw}}$",
                rowindexer_first_of_group="ec_name_technique",
            )
                .plot(
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated__VvsRHE",
                data=data_ec,
                ax=ax_IV_vs_t_potential,
            )
                .plot(
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated_step_normalized__VvsRHE",
                data=data_ec_normalized,
                ax=ax_IV_vs_t_normalized_potential,
            )
                .add_column(
                "label",
                values="points step average",
                rowindexer_first_of_group="ec_name_technique",
            )
                .plot(
                x_col="t_synchronized__s",
                y_col=j_geo_col_normalized,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_normalized_selected,  # data_ec_polcurve_raw,
                ax=ax_IV_vs_t_normalized_current,
                color="tab:green",
            )
                .add_column(
                "label", values="", rowindexer_first_of_group="ec_name_technique"
            )
                .plot(
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated_step_normalized__VvsRHE",
                # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_normalized_selected,  # data_ec_polcurve_raw,
                ax=ax_IV_vs_t_normalized_potential,
                color="tab:green",
            )
                .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label="E_WE_uncompensated_step_normalized__VvsRHE",  # "E_WE_uncompensated__VvsRHE",
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated_step_normalized__VvsRHE",
                # "E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_normalized_selected,  # data_ec_polcurve_raw,
                ax=ax_IV_vs_t_normalized_potential,
                color="tab:green",
                alpha=0.5,
                axlabel_auto=False,
                label_fit=False,
            )
                .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label="I__A",
                display_fit=False,
                x_col="t_synchronized__s",
                y_col="I__A",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve_raw,
                ax=ax_IV_vs_t_current,
                color="tab:green",
                alpha=0.5,
                axlabel_auto=False,
                label_fit=False,
            )
                .fit(
                model=plot.linear_func,
                method="scipy.optimize.curve_fit",
                beta0=[1, 1],
                exp_col_label=j_geo_col_normalized,  # j_geo_col,
                display_fit=True,
                x_col="t_synchronized__s",
                y_col=j_geo_col_normalized,  # j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_normalized_selected,  # data_ec_polcurve_raw,
                ax=ax_IV_vs_t_normalized_current,
                color="tab:green",
                alpha=0.5,
                axlabel_auto=False,
                label_fit=False,
            )
                .return_dataset()
                .reset_index()
        )
        if display_exp_ec_sgeis_last_current_step:
            exp_ec_sgeis_last_current_step.set_index("id_exp_sfc_ghold").dataset.plot(
                x_col="t_synchronized__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax_IV_vs_t_current,
                color="gray",
                zorder=0,
                label="",
            ).plot(
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated__VvsRHE",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec,
                ax=ax_IV_vs_t_potential,
                color="tab:orange",
                label="hold without EIS",
            )

        # Add eis to IV_vs_t
        exp_ec_sgeis = (
            exp_ec_sgeis.set_index("id_exp_sfc_geis")
                .dataset.plot(
                x_col="t_synchronized__s",
                y_col=j_geo_col,
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_eis,
                ax=ax_IV_vs_t_current,
                color="gray",
                zorder=0,
            )
                .add_column("color", values="pub_blues")
                .plot(
                x_col="t_synchronized__s",
                y_col="E_WE_uncompensated__VvsRHE",
                # 'j__mA_cm2geo_fc_top_cell_Aideal',#'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_eis,
                ax=ax_IV_vs_t_potential,
            )
                .return_dataset()
                .reset_index()
        )

        # Drift plot

        # Just for drift plot - will be overwritten once updated
        data_ec_polcurve = data_ec_polcurve.join(
            exp_ec_sgeis.set_index('id_exp_sfc_ghold').E_WE_uncompensated_step_normalized__VvsRHE_linear_fit_m.rename(
                'E_WE_drift_plot__mV_min') * 1000 * 60,
            on='id_exp_sfc_ghold')
        exp_ec_polcurve = (exp_ec_polcurve.dataset
                           .plot(
            y_col="E_WE_drift_plot__mV_min",  # 'E_WE__VvsRHE',
            x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
            data=data_ec_polcurve,
            ax=ax_drift_potential,
            marker="s",
            markersize=1.5,
            linestyle="-",
            label="$E_\mathrm{WE. drift}$",
            color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
            zorder=1,
            # ax_plot_object_name='polcurve_E_HFR_line',
        )
                           .scatter(
            y_col="E_WE_drift_plot__mV_min",  # 'E_WE__VvsRHE',
            x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
            data=data_ec_polcurve,
            ax=ax_drift_potential,
            marker="s",
            s=4,  # markersize=2,
            # linestyle='-',
            label="",  # $E_\mathrm{HFR}$',
            c=plot.get_colormap("pub_blues", len(exp_ec_sgeis.index)),
            zorder=2,
            # ax_plot_object_name='polcurve_E_HFR_points',
        )
                           .return_dataset()
                           )

        ax_IV_vs_t_normalized_potential.set_ylim([-0.002, 0.002])
        ax_IV_vs_t_normalized_current.set_ylim([-2, 1])

        # tafel_fit_limits = [data_ec_polcurve.loc[:, j_geo_col].abs().min(),
        #                     data_ec_polcurve.loc[:, j_geo_col].abs().max()]

        def tafel_fit_filter(data_ec_polcurve_new):
            return (
                # (data_ec_polcurve_new.j__mA_cm2geo_fc_bottom_PTL > 0.01)
                    (data_ec_polcurve_new.loc[:, j_geo_col] >= tafel_fit_limits[0])
                    & (data_ec_polcurve_new.loc[:, j_geo_col] <= tafel_fit_limits[1])
                    & (data_ec_polcurve_new.E_WE__VvsRHE > 1.42)
                    & (data_ec_polcurve_new.loc[:, "gooddata"])
                # &(data_ec_polcurve.scan_direction==2)
            )

        tafel_fit_params = dict(
            model=plot.tafel_func,
            x_col=j_geo_col,  # 'j__mA_cm2geo_fc_bottom_PTL',j__mA_mg_total_geo_fc_bottom_PTL
            y_col="E_WE__VvsRHE",
            # yerr_col='count_ratio_std',
            beta0=[1, 0],
            ifixb=[1, 1],
            method=tafel_fit_method,  # 'scipy.optimize.curve_fit',# "scipy.odr",
            # odr has poorer performance when high hysteresis in for and back sweep
            linestyle=(0, (5, 7)),  # '--',
            color="tab:orange",
            ax=ax_tafel,
            axlabel_auto=False,
        )

        exp_ec_polcurve = (
            exp_ec_polcurve.dataset.plot(
                y_col="E_WE_uncompensated__VvsRHE",
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_polcurve,
                marker="s",
                markersize=2,
                linestyle="-",
                label="$E_\mathrm{raw}$",
                axlabel_auto=False,
                color="gray",
                zorder=0,
                ax_plot_object_name='polcurve_E_raw_linepoints',
            )
                .plot(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_polcurve,
                marker="s",
                markersize=1.5,
                linestyle="-",
                label="$E_\mathrm{HFR}$",
                color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
                zorder=1,
                ax_plot_object_name='polcurve_E_HFR_line',
            )
                .scatter(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_polcurve,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='-',
                label="",  # $E_\mathrm{HFR}$',
                c=plot.get_colormap("pub_blues", len(data_ec_polcurve.index)),
                zorder=2,
                ax_plot_object_name='polcurve_E_HFR_points',
            )
                .scatter(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve.loc[~data_ec_polcurve.loc[:, "gooddata"], :],
                ax=ax_polcurve,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='',
                label="excluded",
                color="tab:red",
                zorder=3,
                ax_plot_object_name='polcurve_E_HFR_points_baddata',
            )
                .plot(
                y_col="E_WE_uncompensated__VvsRHE",
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel,
                marker="s",
                markersize=2,
                linestyle="-",
                label="",
                axlabel_auto=False,
                color="gray",
                zorder=0,
                ax_plot_object_name='tafel_E_raw_linepoints',
            )
                .plot(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel,
                # marker='s',
                # markersize=2,
                linestyle="-",
                label="",
                color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
                zorder=1,
                ax_plot_object_name='tafel_E_HFR_line',
            )
                .scatter(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel,
                s=4,  # markersize=2,
                marker="s",
                # linestyle='-',
                label="",
                c=plot.get_colormap("pub_blues", len(data_ec_polcurve.index)),
                zorder=2,
                ax_plot_object_name='tafel_E_HFR_points',
            )
                .scatter(
                y_col="E_WE__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve.loc[~data_ec_polcurve.loc[:, "gooddata"], :],
                ax=ax_tafel,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='',
                label="",
                color="tab:red",
                zorder=3,
                ax_plot_object_name='tafel_E_HFR_points_baddata',
            )
                .fit(
                **tafel_fit_params,
                data=data_ec_polcurve.loc[tafel_fit_filter(data_ec_polcurve), :],
                zorder=4,
                ax_plot_object_name='tafel_E_HFR_fit',
            )
                .plot(
                y_col="R_u__ohm",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_Ru_j,
                marker="s",
                markersize=1.5,
                linestyle="-",
                label="",
                color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
                zorder=1,
                ax_plot_object_name='Ru_j_line',
            )
                .scatter(
                y_col="R_u__ohm",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                s=4,
                c=plot.get_colormap("pub_blues", len(data_ec_polcurve.index)),
                # data_ec_polcurve.index.get_level_values(level=1).tolist(),
                ax=ax_Ru_j,
                marker="s",
                # markersize=2,
                # linestyle='-',
                label="",
                # color='tab:blue',
                # cmap='Greens',
                zorder=2,
                ax_plot_object_name='Ru_j_points',
            )
                .scatter(
                y_col="R_u__ohm",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve.loc[~data_ec_polcurve.loc[:, "gooddata"], :],
                ax=ax_Ru_j,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='',
                label="",
                color="tab:red",
                zorder=3,
                ax_plot_object_name='Ru_j_points_baddata',
            )
                .return_dataset()
        )

        def update_tafel_res(data_ec_polcurve_new):
            return data_ec_polcurve_new.assign(E_WE_fitted__VvsRHE
                                               =lambda x: tafel_fit_params["model"](
                exp_ec_polcurve.loc[:, ["tafel_fit_m", "tafel_fit_b"]]
                    .iloc[0].values,
                x.loc[:, j_geo_col],
            ),
                                               delta_tafel_fit_residuals__mV
                                               =lambda x: (x.E_WE__VvsRHE - x.E_WE_fitted__VvsRHE) * 1000)

        data_ec_polcurve = update_tafel_res(data_ec_polcurve)

        exp_ec_polcurve = (
            exp_ec_polcurve.dataset
                .plot(
                y_col="E_WE_fitted__VvsRHE",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel,
                # marker='s',
                # markersize=2,
                linestyle="-",
                label="",
                color='tab:orange',  # 'tab:blue',
                alpha=0.3,
                zorder=3,
                axlabel_auto=False,
                ax_plot_object_name='tafel_E_HFR_fit_all',
            )
                .plot(
                y_col="delta_tafel_fit_residuals__mV",  # 'E_WE__VvsRHE',
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel_res,
                # marker='s',
                # markersize=2,
                linestyle="-",
                label="",
                color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
                zorder=1,
                ax_plot_object_name='tafel_res_fit_line',
            )
                .scatter(
                y_col="delta_tafel_fit_residuals__mV",
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve,
                ax=ax_tafel_res,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='',
                label="",
                c=plot.get_colormap("pub_blues", len(data_ec_polcurve.index)),
                zorder=3,
                ax_plot_object_name='tafel_res_fit_points',
            )
                .scatter(
                y_col="delta_tafel_fit_residuals__mV",
                x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                data=data_ec_polcurve.loc[~data_ec_polcurve.loc[:, "gooddata"], :],
                ax=ax_tafel_res,
                marker="s",
                s=4,  # markersize=2,
                # linestyle='',
                label="",
                color="tab:red",
                zorder=3,
                ax_plot_object_name='tafel_res_fit_points_baddata',
            )
                .return_dataset()
        )

        # Hysteresis plot
        if display_hysteresis:
            data_ec_polcurve_avg_for_back = avg_for_back_polcurve(data_ec_polcurve)
            exp_ec_polcurve = (
                exp_ec_polcurve.dataset
                    .plot(
                    y_col="DeltaE_WE_avg_for_back__mV",  # 'E_WE__VvsRHE',
                    x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                    data=data_ec_polcurve_avg_for_back,
                    ax=ax_hysteresis_for_back,
                    # marker='s',
                    # markersize=2,
                    linestyle="-",
                    label="",
                    color=(0.00390625, 0.12109375, 0.29296875, 1.0),  # 'tab:blue',
                    zorder=1,
                    ax_plot_object_name='hysteresis_for_back_line',
                )
                    .scatter(
                    y_col="DeltaE_WE_avg_for_back__mV",
                    x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                    data=data_ec_polcurve_avg_for_back,
                    ax=ax_hysteresis_for_back,
                    marker="s",
                    s=4,  # markersize=2,
                    # linestyle='',
                    label="",
                    color=(0.00390625, 0.12109375, 0.29296875, 1.0),
                    # c=plot.get_colormap("pub_blues", len(data_ec_polcurve_avg_for_back.loc[data_ec_polcurve_avg_for_back.loc[:, "gooddata"], :].index)),
                    zorder=3,
                    ax_plot_object_name='hysteresis_for_back_points',
                )
                    .scatter(
                    y_col="DeltaE_WE_avg_for_back__mV",
                    x_col=j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                    data=data_ec_polcurve_avg_for_back.loc[~data_ec_polcurve_avg_for_back.loc[:, "gooddata"], :],
                    ax=ax_hysteresis_for_back,
                    marker="s",
                    s=4,  # markersize=2,
                    # linestyle='',
                    label="",
                    color="tab:red",
                    zorder=3,
                    ax_plot_object_name='hysteresis_for_back_points_baddata',
                )
                    .return_dataset()
            )

        ax_tafel_res.axhline(y=0,
                             linestyle='--',
                             zorder=-1,
                             color='black'
                             )
        vlines = []
        for ax in [ax_tafel, ax_tafel_res]:
            vlines = [[ax.axvline(x=data_ec_polcurve.loc[tafel_fit_filter(data_ec_polcurve), j_geo_col].min(),
                                  linestyle='--',
                                  zorder=-1,
                                  color='black'
                                  ),
                       ax.axvline(x=data_ec_polcurve.loc[tafel_fit_filter(data_ec_polcurve), j_geo_col].max(),
                                  linestyle='--',
                                  zorder=-1,
                                  color='black'
                                  )], ]

        # Compression plot
        if display_compression:
            exp_compression.dataset \
                .add_column('color', values='pub_blues') \
                .plot(x_col='t_synchronized__s',  # '',
                      y_col='linaxis_z__mm',
                      data=data_compression,  # 'data_ec_analysis', con=engine,
                      ax=ax_compression_zpos,
                      ) \
                .plot(x_col='t_synchronized__s',  # '',
                      y_col='force__N',
                      data=data_compression.loc[lambda row: row.force__N > 0],  # 'data_ec_analysis', con=engine,
                      ax=ax_compression_force,
                      )

        handle2, label2 = ax_IV_vs_t_current.get_legend_handles_labels()
        handle, label = ax_IV_vs_t_potential.get_legend_handles_labels()
        ax_IV_vs_t_potential.legend(handle2 + handle, label2 + label, fontsize=5)
        ax_polcurve.legend()  # fontsize=5)
        ax_tafel_legend = ax_tafel.legend()  # fontsize=5)

        # make secondary axis for ax_Ru_j ro show area normalized resistances
        col_A_geo = plot.geo_columns.loc[lambda row: row.j_geo == j_geo_col, 'A_geo'].iloc[0]
        # only take the area of first exp - assumption: should not change in cours of experiment
        A_active_catalyst__cm2 = exp_ec.loc[:, col_A_geo].iloc[0] / 100
        ax_Ru_j_secondary = ax_Ru_j.secondary_yaxis('right', functions=(lambda R_u__ohm:
                                                                        R_u__ohm * 1000 * A_active_catalyst__cm2,
                                                                        lambda R_u_geo_mohm_cm2:
                                                                        R_u_geo_mohm_cm2 / 1000 / A_active_catalyst__cm2,
                                                                        ))
        ax_Ru_j.yaxis.set_tick_params(which='both', right=False)
        ax_Ru_j_secondary.set_ylabel(
            plot.get_geo_column_label(
                plot.geo_columns.loc[lambda row: row.j_geo == j_geo_col, 'R_u_geo'].iloc[0]
            )
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
            # ax_EIS.autoscale() # will lead to rescale of acxis upon every Ru marker change
            fig.canvas.draw()  # does the update of axis

        def update_polcurve_raw_tail(data_ec_polcurve_raw_new):
            ax_EIS.set_title("changed")
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
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), "t_synchronized__s"]
                )
                row.loc[3][0].set_ydata(
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), j_geo_col]
                )
                row.loc[4][0].set_xdata(
                    data_ec_polcurve_raw_new.loc[(index, slice(None)), "t_synchronized__s"]
                )
                row.loc[4][0].set_ydata(
                    data_ec_polcurve_raw_new.loc[
                        (index, slice(None)), "E_WE_uncompensated__VvsRHE"
                    ]
                )

        def update_line_points_baddata(exp, data, ax_plot_object_name, x_col, y_col):
            # exp_ec_polcurve.ax_plot_objects.iloc[0][1][0]
            exp['ax_plot_object_' + ax_plot_object_name + '_line'].iloc[0].set_data(
                data.loc[:, x_col],
                data.loc[:, y_col],
            )
            exp['ax_plot_object_' + ax_plot_object_name + '_points'].iloc[0].set_offsets(
                data.loc[:, [x_col, y_col]].values
            )

            # red not gooddata
            exp['ax_plot_object_' + ax_plot_object_name + '_points_baddata'].iloc[0].set_offsets(
                data.loc[
                    ~data.loc[:, "gooddata"], [x_col, y_col]
                ].values
            )

        def update_polcurve(data_ec_polcurve_new):
            # blue compensated pol curve
            # exp_ec_polcurve.ax_plot_objects.iloc[0][1][0].set_ydata(data_ec_polcurve_new.loc[:, 'E_WE__VvsRHE'])
            # exp_ec_polcurve.ax_plot_objects.iloc[0][1][0].set_xdata(data_ec_polcurve_new.loc[:, j_geo_col])
            # exp_ec_polcurve.ax_plot_objects.iloc[0][1][0]
            update_line_points_baddata(exp=exp_ec_polcurve,
                                       data=data_ec_polcurve_new,
                                       ax_plot_object_name='polcurve_E_HFR',
                                       x_col=j_geo_col,
                                       y_col="E_WE__VvsRHE")
            update_line_points_baddata(exp=exp_ec_polcurve,
                                       data=data_ec_polcurve_new,
                                       ax_plot_object_name='tafel_E_HFR',
                                       x_col=j_geo_col,
                                       y_col="E_WE__VvsRHE")
            update_line_points_baddata(exp=exp_ec_polcurve,
                                       data=data_ec_polcurve_new,
                                       ax_plot_object_name='Ru_j',
                                       x_col=j_geo_col,
                                       y_col="R_u__ohm")
            if display_hysteresis:
                data_ec_polcurve_avg_for_back_new = avg_for_back_polcurve(data_ec_polcurve_new)
                update_line_points_baddata(exp=exp_ec_polcurve,
                                           data=data_ec_polcurve_avg_for_back_new,
                                           ax_plot_object_name='hysteresis_for_back',
                                           x_col=j_geo_col,
                                           y_col="DeltaE_WE_avg_for_back__mV")

        def update_tafel_fit(data_ec_polcurve_new):
            # tafel fit
            # display(data_ec_polcurve_new)
            # display(data_ec_polcurve_new.loc[tafel_fit_filter(data_ec_polcurve_new), :])
            exp_ec_polcurve_updated_fit = exp_ec_polcurve.dataset.fit(
                **tafel_fit_params,
                data=data_ec_polcurve_new.loc[tafel_fit_filter(data_ec_polcurve_new), :],
                display_fit=False,
            ).return_dataset()
            data_ec_polcurve_new = update_tafel_res(data_ec_polcurve_new)

            exp_ec_polcurve.ax_plot_object_tafel_E_HFR_fit.iloc[0].set_xdata(
                data_ec_polcurve_new.loc[tafel_fit_filter(data_ec_polcurve_new), j_geo_col]
            )
            exp_ec_polcurve.ax_plot_object_tafel_E_HFR_fit.iloc[0].set_ydata(
                data_ec_polcurve_new.loc[tafel_fit_filter(data_ec_polcurve_new), 'E_WE_fitted__VvsRHE']
                # tafel_fit_params["model"](
                #    exp_ec_polcurve.loc[:, ["tafel_fit_m", "tafel_fit_b"]]
                #   .iloc[0]
                #   .values,
                #    data_ec_polcurve_new.loc[tafel_fit_filter(data_ec_polcurve_new), j_geo_col],
                # )
            )
            exp_ec_polcurve.ax_plot_object_tafel_E_HFR_fit.iloc[0]._label = exp_ec_polcurve.label_fit.iloc[0]
            # display(exp_ec_polcurve.label_fit.iloc[0])
            ax_tafel_legend.get_texts()[0].set_text(exp_ec_polcurve.label_fit.iloc[0])

            exp_ec_polcurve.ax_plot_object_tafel_E_HFR_fit_all.iloc[0].set_data(
                data_ec_polcurve_new.loc[:, j_geo_col],
                data_ec_polcurve_new.loc[:, 'E_WE_fitted__VvsRHE'],
            )

            # residual fit

            update_line_points_baddata(exp=exp_ec_polcurve,
                                       data=data_ec_polcurve_new,
                                       ax_plot_object_name='tafel_res_fit',
                                       x_col=j_geo_col,
                                       y_col="delta_tafel_fit_residuals__mV")

            ax_polcurve.autoscale()
            return exp_ec_polcurve_updated_fit

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

            update_polcurve(data_ec_polcurve_new)
            exp_ec_polcurve = update_tafel_fit(data_ec_polcurve_new)

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
            exp_ec_polcurve = update_tafel_fit(data_ec_polcurve_new)

        checkbox_gooddata = widgets.Checkbox(
            value=False, description="gooddata", disabled=True
        )
        checkbox_gooddata.observe(update_on_checkbox_gooddata, "value")

        # Radiobuttons
        def df_to_radio_text(exp_ec_sgeis_df):
            return ["all"] + (
                    exp_ec_sgeis_df.index.astype(str).to_numpy()
                    + ": "
                    + (exp_ec_sgeis_df.ghold_I_hold__A * 1000).astype(str)
                    + " mA, \t initial_id_chosen: "
                    + exp_ec_sgeis_df.id_data_eis_chosen_Ru.astype(str)
                    + " overload: "
                    + exp_ec_sgeis_df.overload_list.astype(str).str.replace("8191", "")
            ).tolist()

        def radio_text_to_df_row(text, exp_ec_sgeis_df):
            return exp_ec_sgeis_df.loc[int(text.split(": ")[0]), :]

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
                                           ].index.min()  # .get_level_values(level=1)
                intSlider_choose_R_u.max = data_eis.loc[
                                           radio_text_to_df_row(change.new, exp_ec_sgeis).loc[
                                               "id_exp_sfc_geis"
                                           ],
                                           :,
                                           ].index.max()  # .get_level_values(level=1)
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
            update_tafel_fit(data_ec_polcurve_new)

        intslider_tail_datapoints.observe(update_on_slider_tail, "value")

        # tafel fit range
        # update_floattext_value = True
        floattext_tafel_leftlimit = widgets.FloatText(
            value=user_input.round_digits(
                value=tafel_fit_limits[0], digits=5, method="down"
            ),
            description="Left limit tafel fit:",
            disabled=False,
            style={"description_width": "initial"},
        )

        def update_floattext_tafel_leftlimit(changed_value):
            # if update_floattext_value:
            rounded = user_input.round_digits(
                value=changed_value.new, digits=5, method="half_up"
            )
            tafel_fit_limits[0] = rounded
            # update_floattext_value = False
            floattext_tafel_leftlimit.value = rounded
            # update_floattext_value = True
            exp_ec_polcurve = update_tafel_fit(data_ec_polcurve)

            # update left vlines
            for axs in vlines:
                axs[0].set_xdata(data_ec_polcurve.loc[tafel_fit_filter(data_ec_polcurve), j_geo_col].min())

        floattext_tafel_leftlimit.observe(update_floattext_tafel_leftlimit, "value")
        floattext_tafel_rightlimit = widgets.FloatText(
            value=user_input.round_digits(
                value=tafel_fit_limits[1], digits=3, method="up"
            ),
            description="Right limit tafel fit:",
            disabled=False,
            style={"description_width": "initial"},
        )

        def update_floattext_tafel_rightlimit(changed_value):
            # if update_floattext_value:
            rounded = user_input.round_digits(
                value=changed_value.new, digits=3, method="half_up"
            )
            tafel_fit_limits[1] = rounded
            # update_floattext_value = False
            floattext_tafel_rightlimit.value = rounded
            # update_floattext_value = True
            exp_ec_polcurve = update_tafel_fit(data_ec_polcurve)

            # update right vlines
            for axs in vlines:
                axs[1].set_xdata(data_ec_polcurve.loc[tafel_fit_filter(data_ec_polcurve), j_geo_col].max())

        floattext_tafel_rightlimit.observe(update_floattext_tafel_rightlimit, "value")

        # Uplpoad Button
        def on_upload_button_clicked():
            radiobuttons.value = "all"
            with upload_output:
                clear_output()
                if not db.user_is_owner("id_exp_sfc", index_value=int(exp_ec.index[0])):
                    print(
                        "\x1b[31m",
                        "You better not change data of other users",
                        "\x1b[0m",
                    )
                    return
                print("Processing ...")
                exp_ec_polcurve.loc[:, "chosen_j_geo_col"] = j_geo_col
                exp_ec_polcurve.loc[
                :, "tafel_fit_left_limit__j_geo"
                ] = floattext_tafel_leftlimit.value
                exp_ec_polcurve.loc[
                :, "tafel_fit_right_limit__j_geo"
                ] = floattext_tafel_rightlimit.value

                exp_ec_polcurve_tosql = exp_ec_polcurve.rename(
                    columns={
                        "tafel_fit_m": "tafel_fit_m__V_dec",
                        "tafel_fit_m_sd": "tafel_fit_m_sd__V_dec",
                        "tafel_fit_b": "tafel_fit_b__VvsRHE",
                        "tafel_fit_b_sd": "tafel_fit_b_sd__VsRHE",
                        "tafel_fit_ResVar": "tafel_fit_ResVar",
                        "fit_method": "tafel_fit_method",
                    },
                ).loc[
                                        :,
                                        [
                                            "changed_exp_parameters",
                                            "chosen_j_geo_col",
                                            "tafel_fit_left_limit__j_geo",
                                            "tafel_fit_right_limit__j_geo",
                                            "tafel_fit_m__V_dec",
                                            "tafel_fit_m_sd__V_dec",
                                            "tafel_fit_b__VvsRHE",
                                            "tafel_fit_b_sd__VsRHE",
                                            "tafel_fit_ResVar",
                                            "tafel_fit_method",
                                        ],
                                        ]
                exp_ec_polcurve_tosql.loc[:, "t_inserted_data__timestamp"] = str(
                    dt.datetime.now()
                )
                exp_ec_polcurve_tosql.loc[:, "number_datapoints_in_tail"] = intslider_tail_datapoints.value
                exp_ec_polcurve_tosql.loc[:, "skip_id_data_eis_greater"] = skip_id_data_eis_greater
                exp_ec_polcurve_tosql.loc[:, "ignore_overload_n_first_datapoints"] = ignore_overload_n_first_datapoints

                if exp_ec_polcurve_tosql.tafel_fit_method.iloc[0] != tafel_fit_method:
                    raise Exception('Report to admin: tafel_fit_method not as given in the function')

                id_exp_ec_polcurve = (
                    exp_ec_polcurve_tosql.reset_index().id_exp_ec_polcurve.iloc[0]
                )

                with engine.begin() as conn:
                    # begin() not connect()
                    # --> this way everythin is handled as transaction and will be rollbakced in case of error
                    db.call_procedure(
                        engine,
                        "Reset_Autoincrement",
                        ["exp_ec_polcurve", "id_exp_ec_polcurve"],
                    )
                    try:
                        if id_exp_ec_polcurve == "new":
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

                            # Determine AI id_exp_ec_polcurve given by db
                            id_exp_ec_polcurve = exp_ec_polcurve_tosql.reset_index().id_exp_ec_polcurve.iloc[
                                0
                            ]
                            # Update global exp_ec_polcurve
                            exp_ec_polcurve.rename(
                                index={
                                    exp_ec_polcurve.reset_index().id_exp_ec_polcurve.iloc[
                                        0
                                    ]: id_exp_ec_polcurve
                                },
                                inplace=True,
                            )  # Overwrite id_exp_ec_polcurve in global exp_ec_polcurve
                        else:
                            # update exp (not delete+reinsert --> won't work with auto increment)
                            sql_update = ", ".join(
                                [
                                    col
                                    + ''' = "'''
                                    + str(exp_ec_polcurve_tosql.loc[:, col].iloc[0])
                                    + '"'
                                    for col in exp_ec_polcurve_tosql.columns
                                ]
                            )
                            db.query_sql(
                                """UPDATE exp_ec_polcurve
                                            SET """
                                + sql_update
                                + """ 
                                            WHERE id_exp_ec_polcurve = %s
                                         ;""",
                                con=conn,
                                params=[int(id_exp_ec_polcurve)],
                                method="sqlalchemy",
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
                                [int(id_exp_ec_polcurve)],
                            )

                        # Save figure and update file path in db
                        file_name = str(
                            str(int(id_exp_ec_polcurve))
                            + "_"
                            + str(date)
                            + "_"
                            + str(name_user)
                            + "_"
                            + str(name_setup_sfc)
                            + "_"
                            + str(id_ML)
                        )
                        export_file_path = str(export_folder / file_name)
                        plt.savefig(export_file_path + ".svg")
                        # TODO: Use PlotDataStorage export routine
                        print(
                            "\x1b[32m",
                            "Successfully stored plot:",
                            file_name,
                            "\x1b[0m",
                        )
                        db.query_sql(
                            '''UPDATE exp_ec_polcurve
                                        SET file_path_processing_plot = "'''
                            + str(export_file_path)
                            + """"
                                        WHERE id_exp_ec_polcurve = %s
                                     ;""",
                            con=conn,
                            params=[int(id_exp_ec_polcurve)],
                            method="sqlalchemy",
                        )
                        exp_ec_polcurve_tosql.loc[:, "file_path_processing_plot"] = str(
                            export_file_path
                        )  # just for display resaons
                        # display(exp_ec_polcurve_tosql)

                        # recalculate data_ec_polcurve with chosen intslider_tail and updated id_ecp_ec_polcurve
                        (
                            data_ec_polcurve,
                            data_ec_polcurve_raw,
                        ) = calculate_data_ec_polcurve(
                            data_ec,
                            data_eis,
                            exp_ec_sgeis,
                            number_datapoints_in_tail_new=intslider_tail_datapoints.value,
                        )
                        # Insert into data_ec_polcurve
                        data_ec_polcurve.loc[
                            data_ec_polcurve.id_exp_sfc_geis.isin(
                                exp_ec_sgeis.loc[
                                    (
                                            exp_ec_sgeis.id_data_eis_chosen_Ru
                                            != exp_ec_sgeis.id_data_eis_chosen_Ru_init
                                    ),
                                    "id_exp_sfc_geis",
                                ].tolist()
                            ),
                            "R_u_derived_by",
                        ] = "manual"
                        data_ec_polcurve.rename(
                            columns={
                                "I__A_linear_fit_m": "I_drift__A_s",
                                # "E_WE_uncompensated__VvsRHE_linear_fit_m": "E_WE_uncompensated_drift__V_s",
                                "E_WE_uncompensated_step_normalized__VvsRHE_linear_fit_m": "E_WE_uncompensated_drift__V_s",
                            }
                        ).loc[
                        :,
                        [
                            "id_exp_sfc_geis",
                            "id_exp_sfc_ghold",
                            "id_current_step",
                            "scan_direction",
                            "id_data_eis_chosen_Ru",
                            "R_u_derived_by",
                            # 'R_u__ohm',
                            "overload_list",  # min
                            "gooddata",
                            # Timestamp't_synchronized__s', #min
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
                        radiobuttons,
                        intSlider_choose_R_u,
                        checkbox_gooddata,
                        upload_button.widget,
                        upload_output,
                    ],
                    layout=Layout(width="50%"),
                ),
                widgets.VBox(
                    [
                        intslider_tail_datapoints,
                        floattext_tafel_leftlimit,
                        floattext_tafel_rightlimit,
                    ]
                ),
            ]
        )

        display(widgets_box)

        # fig.align_ylabels(axs)
        plt.show()

        return exp_ec_polcurve, data_ec_polcurve, exp_ec_sgeis, data_ec


def get_individual_cycles(exp_ec, data_ec):
    """
    Extend experimental list by number of cycles within the experiment (cyclic voltammetry).
    Use this to individual navigate to the cycles for example to style or highlight them.
    :param exp_ec: EC experimental dataframe
    :param data_ec: EC data dataframe
    :return: exp_ec_cycles, data_ec_cycles
        having "cycle" appended to index of both dataframes
    """

    # get index names, default='id_exp_sfc' but could be more columns when using overlay feature
    index_names = exp_ec.index.names

    exp_ec_cycles = exp_ec.join(
        data_ec.reset_index().loc[:, index_names+['cycle', ]].drop_duplicates().set_index(index_names),
        on=index_names).set_index('cycle', append=True)
    data_ec_cycles = data_ec.reset_index().set_index(index_names+['cycle', 'id_data_ec'])

    return exp_ec_cycles, data_ec_cycles



