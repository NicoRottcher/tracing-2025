"""
Scripts for database-related functions
Created in 2023
@author: Nico Röttcher
"""

import os.path

import sqlalchemy as sql
import pandas as pd
import sys
import numpy as np
from ipywidgets import *
import warnings
import datetime
from IPython.display import SVG

from evaluation.utils import db_config, tools, user_input
from evaluation.processing import tools_ec
from evaluation.visualization import plot

MYSQL = db_config.MYSQL


def connect(*args, **kwargs):
    """
    method to connect to database either Mysql or Sqlite as defined in evaluation.utils.db_config
    """
    return db_config.connect(*args, **kwargs)


def query_sql(query, params=None, con=None, method="pandas", debug=False, **kwargs):
    """
    Standard command to run a SQL query. If run with MySQL: The SQL query will be verified to be compatible
    with sqlite syntax (thereby, some common syntax differences are considered in evaluation.utils.mysql_to_sqlite.py)
    The compatibility is required to be able to run the code with sqlite in mybinder when publishing the code.
    Therefore, the direct use of sqlalchemy.connection.execute() or pandas.read_sql() is highly discouraged.
    :param query: str
        requested SQL query. If method='pandas' the table name is accepted to query the select complete table data.
    :param params: list of Any (any supported types)
        list of the parameters marked in query with '%s'
    :param con: sql.Connection or None, optional, default None
        database connection object, if None a new will be initialized
    :param method: one of ['pandas', 'sqlalchemy']
        choose with which module to run the query: sqlalchemy.connection.execute() or pandas.read_sql()
    :param debug: bool, optional, Default False
        print additional debug info
    :param kwargs:
        kwargs of sqlalchemy.connection.execute() or pandas.read_sql()
    :return: return of sqlalchemy.connection.execute() or pandas.read_sql()
    """
    query = db_config.verify_sql(query, params=params, method=method, debug=debug)

    if con is None:
        engine = connect()
        with engine.begin() as con:
            temp = query_sql(query, params=params, con=con, method=method, debug=debug, **kwargs)
        engine.dispose()
        return temp
    else:
        return query_sql_execute_method(query, params=params, con=con, method=method, debug=debug, **kwargs)


def query_sql_execute_method(
    query, params=None, con=None, method="pandas", debug=False, **kwargs
):
    """
    function called in evaluation.utils.db.query_sql() and evaluation.utils.db_config.verify_sql()
    Not intended for different use. Rather use evaluation.utils.db.query_sql()
    :param query: str
        requested SQL query. If method='pandas' the table name is accepted to query the select complete table data.
    :param params: list of Any (any supported types)
        list of the parameters marked in query with '%s'
    :param con: sql.Connection or None, optional, default None
        database connection object, if None a new will be initialized
    :param method: one of ['pandas', 'sqlalchemy']
        choose with which module to run the query: sqlalchemy.connection.execute() or pandas.read_sql()
    :param debug: bool, optional, Default False
        print additional debug info
    :param kwargs:
        kwargs of sqlalchemy.connection.execute() or pandas.read_sql()
    :return: return of sqlalchemy.connection.execute() or pandas.read_sql()
    """
    print(query, params, con) if debug else ""
    if method == "pandas":
        return pd.read_sql(query, params=params, con=con, **kwargs)
    elif method == "sqlalchemy":
        if params is None:
            return con.execute(query)
        else:
            return con.execute(query, params)
    else:
        raise NotImplementedError("Method not implemented")


def insert_into(conn, tb_name, df=None):
    """
    Run an 'INSERT INTO' query for data from df into database table with Auto Increment index column. Returns the
    auto increment index in the column inserted_primary_key
    :param conn: db connection
    :param tb_name: name of the table as string
    :param df: pd.DataFrame or None, optional, Default None
        values to be inserted as dataframe
        if None, current auto increment value is returned
    :return:
        if df is None, current auto increment value of the table is returned
        else: df is returned with the auto increment index added in the column inserted_primary_key
    """
    if df is None:
        stmt = sql.Table(tb_name, sql.MetaData(), autoload_with=conn).insert().values()
        return conn.execute(stmt).inserted_primary_key
    else:
        for index, row in df.iterrows():
            stmt = (
                sql.Table(tb_name, sql.MetaData(), autoload_with=conn)
                .insert()
                .values(**row.to_dict())
            )
            df.loc[index, "inserted_primary_key"] = conn.execute(
                stmt
            ).inserted_primary_key
        return df


def call_procedure(engine, name, params=None):
    """
    A function to run stored procedure, this will work for mysql-connector-python but may vary with other DBAPIs!
    :param engine: sqlalchemy engine
    :param name: name of the stored procedure
    :param params: parameter or the stored procedure as list
    :return: list of all result sets as pd.DataFrame
    """
    return db_config.call_procedure(engine, name, params=params)


def db_constraints_update():
    """
    restrictions for updates
    :return: dict
        keys: table name
        value: None (updates to the whole table allowed)
                list of str (names of column to which an update is allowed)
    """
    return {
        "exp_icpms_sfc": None,
        "gases": None,
        "exp_icpms_integration": None,
        "exp_ec_integration": None,
        "ana_integrations": None,
        "exp_sfc": ["t_end__timestamp"],
        "exp_icpms_sfc_batch": ["id_exp_ec_dataset", "name_analysis"],
    }


def sql_update(df_update, table_name, engine=None, con=None, add_cond=None):
    """
    Update database by values given in DataFrame df_update. If error occurs, transaction is rolled back.
    :param df_update: pd.DataFrame
        DataFrame with rows and columns which should be updated in the database.
        Reduce columns to columns which actually should be updated.
    :param table_name: str
        Name of the table in database
    :param engine: sql.engine, optional
        Sqlalchemy engine to perform the update
    :param con: sql.connection, optional
        Sqlalchemy connection to perform the update, instead of engine
    :param add_cond: str
        additional condition to subselect rows in the table meant to be updated
    :return: None
    """
    return db_config.sql_update(
        df_update, table_name, engine=engine, con=con, add_cond=add_cond
    )


def get_exp(
    by,
    name_table=None,
    con=None,
    index_col=None,
    groupby_col=None,
    join_col=None,
    debug=False,
    dropna_index=False,
    **kwargs_read_sql,
):
    """
    Standard routine to get list of experiments either by setting by=SQLquery or by=DataFrame with experimental
    index column.
    From the SQL query the name_base_table is automatically derived, from this the index_col is automatically set.
    If executed within a TrackedExport folder the requested experiments are linked to the TrackedExport.
    :param by: str or pd.DataFrame
        str: SQLquery to request experiment from database
        pd.DataFrame: having a column of the index column  of the experiment.
    :param name_table: str or None, optional Default None,
        name of the table from which to request data, required  or th
        if by is SQLquery: required only if auto detection of the table name from SQLquery fails
        if by is pd.DataFrame: required
    :param con: sqlalchemy.connection
        database connection object
    :param index_col: str or list of str or None
        name of the index columns
        if None: will be automatically set as the primary key of the given table
    :param groupby_col: str or list of str or None, optional, Default None
        only relevant if by is pd.Dataframe
        if None: set as index_col
        else: columns by which the given DataFrame should be grouped to derive the experiments
        necessary to redirect request to evaluation.utils.db.get_data()
    :param join_col:  list or None, optional, Default None
        common columns of given experimental and requested data table, if None index column(s) of caller is/are taken
        required for experiment overlaying
    :param debug: bool, Default False
        print additional debug info
    :param dropna_index:
        whether to drop NaN indices
    :param kwargs_read_sql:
        keyword arguments of pandas.read_sql
    :return: experimental DataFrame
    """
    if type(by) == str and name_table is None:
        name_table = derive_name_table(by, debug)
    name_base_table = derive_name_base_table(name_table, debug)
    if index_col is None:
        index_col = (
            get_primarykeys(name_base_table) if name_base_table is not None else []
        )

    if type(by) == str:  # by is a sql statement
        sql_exp = by
        if con is None:
            con = connect()
        df_exp = query_sql(
            sql_exp,
            con=con,
            index_col=index_col if len(index_col) > 0 else None,
            method="pandas",
            **kwargs_read_sql,
        )
        df_exp = df_cols_dtype_to_timestamp(df_exp)  # transform timestamp columns
    elif (
        type(by) == pd.core.frame.DataFrame
    ):  # by is data DataFrame from which the experiments should be derived
        if groupby_col is None:
            groupby_col = index_col

        # short way via restructuring data table and hand it over to get_data
        df_exp = get_data(
            by.reset_index().groupby(groupby_col, dropna=dropna_index).min(),
            name_table,
            auto_add_id=False,
            join_cols=join_col,
            index_cols=index_col,
            **kwargs_read_sql,
        )
    else:
        raise Exception(
            '"by" must be either SQL statement or a DataFrame from which to derive data from'
        )

    if db_config.MYSQL:
        from evaluation.export import tracking

        tracking.add_tracked_exp(df_exp, name_base_table, debug=debug)

    return df_exp


def get_data_raw(name_table,
                 col_names,
                 col_values,
                 add_cond):
    """
    Core part of the get_data defined in evaluation.utils.db in which the database query is built and executed.
    Here, query is built in sqlite syntax.
    :param name_table: name of the table from which data should be received
    :param col_names: name of the index columns
    :param col_values: values of the index columns
    :param add_cond: str, optional Default None
        additional condition to subselect only part of the data
    :return: data as pd.DataFrame
    """
    return db_config.get_data_raw(name_table, col_names, col_values, add_cond)


def get_data(
    df_exp,
    name_table,
    join_cols=None,
    join_overlay_cols=None,
    index_cols=None,
    auto_add_id=True,
    add_cond=None,
    extend_time_window=0,
    t_start_shift__s=0,
    t_end_shift__s=0,
    add_data_without_corresponding_ec=True,
):
    """
    Convenient way to get data tables from database, without formulating sql queries.
    Matches id_exp_sfc to data_icpms if name_table == data_icpms_sfc_analysis
    :param df_exp: pandas.DataFrame
        DataFrame of experimental set
    :param name_table: str, name of the table to get the data from
    :param join_cols: optional, list,
        common columns of given experimental and requested data table, if None index column(s) of df_exp is/are taken
    :param join_overlay_cols: optional, list,
        columns additional to join_cols used to join df_exp on the data table in case in index_col columns are given
        which are not supplied by database (required for overlay example)
    :param index_cols: optional, list,
        column(s) of requested columns which should be the index in the returned dataframe,
        if None the index of the df_exp plus the data id column is taken (if auto_add_id == True)
    :param auto_add_id: bool default True,
        if True searches for another index column in the requested data dataframe and add this to index column
    :param add_cond: str,
        additional condition for requesting the data dataframe.
        For example, use this to select only a specific cycle in a CV experiment
    :param extend_time_window: depracated
    :param t_start_shift__s: float
        substitutes expand_time_window will match n seconds before or after the start timestamp
        of the icpms experiment to th ec experiment
    :param t_end_shift__s: float
        substitutes expand_time_window will match icpms n seconds before or after the end timestamp
        of the icpms experiment to th ec experiment
    :param add_data_without_corresponding_ec: optional, default True,
            only applies for data_table_name == 'data_icpms_sfc_analysis'
            and when the df_exp is match_exp_sfc_exp_icpms,
            True will select also data which has no corresponding ec experiment
            False ignores that data, lower performance
    :return: experimental data DataFrame
    """
    join_cols = tools.check_type(
        "join_cols",
        join_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    join_overlay_cols = tools.check_type(
        "join_overlay_cols",
        join_overlay_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    index_cols = tools.check_type(
        "index_cols",
        index_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )

    if join_overlay_cols is None:
        join_overlay_cols = []
    print('Read data from "' + name_table + '" ...')
    t_start = datetime.datetime.now()

    if (
        name_table in ["exp_ec_expanded", "data_icpms_sfc_analysis"]
        and df_exp.reset_index()
        .columns.isin(
            ["id_exp_sfc", "id_exp_icpms", "t_start__timestamp", "t_end__timestamp"]
        )
        .sum()
        < 4
        and df_exp.reset_index().columns.str.contains("id_data").any()
    ):
        text = "This version of data grabbing will be deprecated soon. " \
               "Please consider adjusting your data grabbing method. " \
               "Refer to shared/02_templates/03_Plotting/ICPMS_plot_templates.ipynb !"
        print("\x1b[31m", text, "\x1b[0m")
        warnings.warn(text)
        name_table += "_old"

    if name_table.lower() in ["data_icpms_sfc_analysis_no_istd_fitting"]:
        print(
            "\x1b[31m",
            "You requested ICPMS data with point-by-point count ratio, which is not recommended. "
            "Please use the ISTD fitting tool before using the data.",
            "\x1b[0m",
        )
    if extend_time_window > 0:
        warnings.warn(
            "extend_time_window parameter is deprecated and will be ignored. "
            "Use t_start_shift__s and t_end_shift__s instead"
        )
    if name_table == 'data_ec_analysis':
        print(
            "\x1b[33m",
            "Querying from VIEW data_ec_analysis is not recommended due to performance issues. "
            "Use the faster method db.get_data_ec to avoid this error message. ",
            "\nIf you query sfc and icpms experiments use directly: db.get_exp_sfc_icpms()",
            "\x1b[0m",
        )
        #return get_data_ec(df_exp,
        #                   join_cols=join_cols,
        #                  join_overlay_cols=join_overlay_cols,
        #                  index_cols=index_cols,
        #                  auto_add_id=auto_add_id,
        #                  add_cond=add_cond,
        #                  )
    if name_table == 'data_icpms_sfc_analysis':
        print(
            "\x1b[33m",
            "Querying from VIEW data_icpms_sfc_analysis is not recommended due to performance issues. "
            "The faster method db.get_data_icpms is available. Please use that instead:\n ",
            "db.get_data_icpms_sfc(exp_icpms, match_ec_icpms, ....). ",
            "\nIf you query sfc and icpms experiments use: db.get_exp_sfc_icpms()",
            "\x1b[0m",
        )
        # not used because match_ec_icpms is missing
        # return get_data_icpms(df_exp,
        #                      join_cols=join_cols,
        #                     join_overlay_cols=join_overlay_cols,
        #                     index_cols=index_cols,
        #                     auto_add_id=auto_add_id,
        #                     add_cond=add_cond,
        #                     t_start_shift__s=t_start_shift__s,
        #                     t_end_shift__s=t_end_shift__s,
        #                     add_data_without_corresponding_ec=add_data_without_corresponding_ec,
        #                     )

    # Derive the columns from which the respective data should be requested
    # if using_cols not specified use index cols of the dataframe
    # else use the specified once
    col_names = df_exp.index.names if join_cols is None else join_cols
    col_values = (
        df_exp.index.to_list()
        if join_cols is None
        else df_exp.reset_index().groupby(join_cols).first().index.to_list()
    )

    if len(col_values) == 0:
        print(
            "\x1b[31m",
            "You requested data for an empty experimental dataset! I return an empty dataframe",
            "\x1b[0m",
        )
        with connect().begin() as con:
            return pd.read_sql(f'SELECT * FROM {name_table} LIMIT 0',
                               con=con,
                               index_col=get_primarykeys(derive_name_base_table(name_table)),
                               )
        #return pd.DataFrame({}, index=col_names + ["id_data"])
        # sys.exit('No rows found to get data from')

    data = get_data_raw(
        name_table=name_table,
        col_names=col_names,
        col_values=col_values,
        add_cond=add_cond,
    )

    if len(data.index) == 0:
        print(
            "\x1b[31m",
            "There is no data found for the requested experiments. I return an empty dataframe",
            "\x1b[0m",
        )
        with connect().begin() as con:
            return pd.read_sql(f'SELECT * FROM {name_table} LIMIT 0',
                               con=con,
                               index_col=get_primarykeys(derive_name_base_table(name_table)),
                               )

        # return pd.DataFrame({}, index=col_names + ["id_data"])
        # sys.exit('No data found in database, for the requested query.')

    data = df_cols_dtype_to_timestamp(data)

    # special timestamp matching for data_icpms_sfc_analysis and when df_exp is match_exp_sfc_exp_icpms
    if df_exp.reset_index().columns.isin(
        ["id_exp_sfc", "id_exp_icpms"]
    ).sum() == 2 and name_table in [
        "data_icpms_sfc_analysis",
        "data_icpms_sfc_analysis_no_istd_fitting",
    ]:
        t_3 = datetime.datetime.now()
        data = data.set_index(["id_exp_icpms", "id_data_icpms"]).sort_index()
        start_shift = pd.Timedelta(seconds=t_start_shift__s)
        end_shift = pd.Timedelta(seconds=t_end_shift__s)
        for index, row in df_exp.reset_index().iterrows():
            # old (=slower) versions for time matching
            # v1 - not correct exp_icpms and exp_ec matching
            # data_icpms.loc[(row.id_exp_icpms,
            #                slice((data_icpms.t_delaycorrected__timestamp_sfc_pc
            #                - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()[1],
            #                    (data_icpms.t_delaycorrected__timestamp_sfc_pc
            #                    - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin()[1]
            #                 )), 'id_exp_sfc'] = row.id_exp_sfc

            # v2  - not correct exp_icpms and exp_ec matching, problems when two icpms measurements simultaneously
            # data_icpms2.loc[(data_icpms2.t_delaycorrected__timestamp_sfc_pc
            #                   - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin():\
            #                    (data_icpms2.t_delaycorrected__timestamp_sfc_pc
            #                    - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin()
            #                 , 'id_exp_sfc'] = row.id_exp_sfc

            # v3 select all icpms experiments belonging to the looped exp_ec from these compare timestamps
            # - faster and correct matching
            # data_icpms.loc[(row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()):\
            #       (row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].t_delaycorrected__timestamp_sfc_pc
            #           - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin())
            #       , 'id_exp_sfc'] = row.id_exp_sfc

            # v3.2 with grab lines with matching id_exp_icpms only once
            # and calculate timedelta for start and end_shift before for loop - even a bit faster
            a = data.loc[row.id_exp_icpms]
            data.loc[
                (
                    row.id_exp_icpms,
                    (
                        a.t_delaycorrected__timestamp_sfc_pc
                        - (pd.to_datetime(row.t_start__timestamp) + start_shift)
                    )
                    .abs()
                    .idxmin(),
                ): (
                    row.id_exp_icpms,
                    (
                        a.t_delaycorrected__timestamp_sfc_pc
                        - (pd.to_datetime(row.t_end__timestamp) + end_shift)
                    )
                    .abs()
                    .idxmin(),
                ),
                "id_exp_sfc",
            ] = row.id_exp_sfc

            # v3.3 placing correction of start and end time before for loop --> slower
            # df_match.loc[:,'t_start__timestamp'] = pd.to_datetime(df_match.t_start__timestamp)
            #                                       - pd.Timedelta(seconds=500)
            # df_match.loc[:,'t_end__timestamp'] = pd.to_datetime(df_match.t_start__timestamp)
            #                                       + pd.Timedelta(seconds=500)

            # v4.1 with grouping index, reduce redudndat time columns appear with multiple measured analyte elements,
            # but slower
            # data_icpms.loc[(row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].groupby(level=0)
            # .t_delaycorrected__timestamp_sfc_pc.first() - (pd.to_datetime(row.t_start__timestamp)
            # -pd.Timedelta(seconds=0))).abs().idxmin()):\
            #               (row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].groupby(level=0)
            #               .t_delaycorrected__timestamp_sfc_pc.first() - (pd.to_datetime(row.t_end__timestamp)
            #               +pd.Timedelta(seconds=0))).abs().idxmin())
            #               , 'id_exp_sfc'] = row.id_exp_sfc

            # v4.2 similar but with .index.duplicated instead
            # a=data_icpms.loc[row.id_exp_icpms]
            # data_icpms.loc[(row.id_exp_icpms,
            #               (a.loc[~a.index.duplicated(keep='first')].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()):\
            #               (row.id_exp_icpms,
            #               (a.loc[~a.index.duplicated(keep='first')].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin())
            #               , 'id_exp_sfc'] = row.id_exp_sfc

        # remove unmmatched data if requested by add_data_without_corresponding_ec
        if not add_data_without_corresponding_ec:
            data = data.loc[~data.id_exp_sfc.isna(), :]

        # add A_geo_cols
        A_geo_cols = df_exp.columns[df_exp.columns.isin(plot.geo_columns.A_geo)]
        # if join columns not overlapping --> check that all dataframes not series
        # data=data.reset_index().set_index(index_cols).sort_index() # index will be set later

        data_joined = data.join(
            df_exp.set_index("id_exp_sfc").sort_index().loc[:, A_geo_cols],
            on="id_exp_sfc",
        )  # .loc[:, A_geo_cols]
        # (data_joined.loc[:, A_geo_cols] / 100).div(data_joined.loc[:, ['dm_dt__ng_s', ]].values)#loc[:, A_geo_cols]

        for A_geo_col in A_geo_cols:
            data.loc[
                :, plot.get_geo_column(A_geo_col, "A_geo", "dm_dt_S", analysis_parameter=False)
            ] = data_joined.dm_dt__ng_s / (
                data_joined.loc[:, A_geo_col] / 100
            )  # loc[:, A_geo_cols]

        # might improve performance, but jupyter crashes when executing
        # data = data.join((data.join(match_ec_icpms.set_index('id_exp_sfc').sort_index().loc[:, A_geo_cols]
        # .rename(columns=plot.geo_columns.set_index('A_geo').to_dict()['dm_dt_S']), on='id_exp_sfc')
        # .loc[:, plot.get_geo_column(A_geo_cols, 'A_geo', 'dm_dt_S')]/ 100).rdiv(data.loc[:, ['dm_dt__ng_s', ]]
        # .values))

        data = data.reset_index()

        t_4 = datetime.datetime.now()
        print("Timestamp matching in ", t_4 - t_3)

    # for data tables an additional primary key column is added to index
    # for exp tables (when using get_exp()) this is not the case --> get_data is called with auto_add_id = False
    if index_cols is None:
        add_id_col = []
        if auto_add_id:
            id_cols = [
                colname
                for colname in data.columns
                if "id_" in colname and colname not in list(df_exp.index.names)
            ]
            if len(id_cols) > 0:
                add_id_col = [id_cols[0]]
        index_cols = list(df_exp.index.names) + add_id_col
    else:
        # eventually add requested index cols from df_Exp
        if any(
            [index_col not in data.reset_index().columns for index_col in index_cols]
        ):  # Any requested index cols not in data?
            index_col_from_self_obj = []
            for index_col in index_cols:
                if index_col in data.reset_index().columns:
                    continue
                elif index_col in df_exp.reset_index().columns:  # get it from self._obj
                    index_col_from_self_obj = index_col_from_self_obj + [index_col]
                else:
                    warnings.warn(
                        "Index column: "
                        + index_col
                        + " not in requested dataframe nor in given dataframe"
                    )
            data = data.join(
                df_exp.reset_index()
                .set_index(join_cols + join_overlay_cols)
                .loc[:, index_col_from_self_obj],
                on=join_cols + join_overlay_cols,
            )

    data_indexed = data.set_index(index_cols).sort_index()
    t_end = datetime.datetime.now()
    print("Done in ", t_end - t_start)
    return data_indexed


def get_data_ec(exp_ec,
                name_table=None,
                geo_cols=None,
                j_geo_cols=None,
                add_cond=None,
                add_data_eis=True,
                add_cond_eis=None,
                ):
    """
    Substitutes the database VIEW data_ec_analysis.
    :param exp_ec: pd.DataFrame
        EC experimental dataframe from VIEW exp_ec_expanded
    :param name_table: str
        deprecated, just for compatibility reasons
    :param geo_cols: str, list of str or None
        specifiy which geo_cols to use for current density calculation, or specify j_geo_cols
    :param j_geo_cols: str, list of str or None
        specifiy which j_geo_cols to calculate, or specify geo_cols
    :param add_cond: str,
        additional condition for requesting the data dataframe.
        For example, use this to select only a specific cycle in a CV experiment
    :param add_data_eis: bool,
        if True and eis experiment in exp_ec, will query for eis data from data_eis_analysis and add it to data_ec
    :param add_cond_eis: str,
        additional condition for requesting the data_eis dataframe.
        Only applies if add_data_eis=True and eis technique in exp_ec
    :return: data_ec as pd.DataFrame
    """

    geo_cols = tools.check_type(
        "geo_cols",
        geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    j_geo_cols = tools.check_type(
        "j_geo_cols",
        j_geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )

    if name_table is not None:
        print("\x1b[33m", "The second parameter 'name_table' is not required anymore, thus deprecated.", "\x1b[0m")

    # Handle exp_ec from overlay example
    index_cols = None
    if exp_ec.index.names != ['id_exp_sfc']:
        print('You changed the index of exp_ec, index of data_ec will be adjusted accordingly.')
        index_cols = exp_ec.index.names
        exp_ec = exp_ec.reset_index().set_index('id_exp_sfc')

    # get non-VIEW data
    data_ec = get_data(exp_ec,
                    'data_ec',  # querying from view is too slow
                    add_cond=add_cond
                    )
    if add_data_eis:
        if 'exp_ec_peis' in exp_ec.ec_name_technique.values or 'exp_ec_geis' in exp_ec.ec_name_technique.values:
            data_eis = get_data(exp_ec, 'data_eis', add_cond=add_cond_eis)\
                                .rename(columns={'id_data_eis': 'id_data_ec',
                                                 'E_dc__VvsRE': 'E_WE_raw__VvsRE',
                                                 'I_dc__A': 'I__A'})
            data_eis.loc[:, 'Delta_E_WE_uncomp__V'] = 0
            data_ec = pd.concat([data_ec, data_eis]).sort_index()

    t_1 = datetime.datetime.now()

    required_cols_exp = ['t_start__timestamp', 'ec_E_RE__VvsRHE']
    if any([col not in exp_ec.columns for col in required_cols_exp]):
        print("\x1b[31m", "Missing columns in exp_ec to calculate standard columns for data_ecs."
                          " Required: %s" % required_cols_exp,
              "\x1b[0m")
        return data_ec

    # Timestamp not given for all Versions of eCat(?)
    if data_ec.loc[:, 'Timestamp'].isna().any():
        data_ec.loc[data_ec.loc[:, 'Timestamp'].isna(), 'Timestamp'] = pd.to_timedelta(
            data_ec.loc[data_ec.loc[:, 'Timestamp'].isna(), 't__s'], unit='s') + exp_ec.t_start__timestamp

    # Potential columns
    data_ec.loc[:, 'E_WE_raw__VvsRHE'] = data_ec.E_WE_raw__VvsRE + exp_ec.ec_E_RE__VvsRHE
    data_ec.loc[:, 'E_WE_uncompensated__VvsRHE'] = data_ec.E_WE_raw__VvsRHE + data_ec.Delta_E_WE_uncomp__V
    exp_ec, data_ec = tools_ec.ohmic_drop_correction(exp_ec, data_ec)

    # Geometric current columns
    if geo_cols is not None:
        j_geo_cols = tools_ec.geo_columns.loc[tools_ec.geo_columns.A_geo.isin(geo_cols), 'j_geo'].values.tolist()
    elif j_geo_cols is not None:
        geo_cols = tools_ec.geo_columns.loc[tools_ec.geo_columns.j_geo.isin(j_geo_cols), 'A_geo'].values.tolist()
    else:
        geo_cols = tools_ec.geo_columns.A_geo.values.tolist()
        j_geo_cols = tools_ec.geo_columns.j_geo.values.tolist()

    for geo_col, j_geo_col in zip(geo_cols, j_geo_cols):
        if geo_col not in exp_ec.columns:
            if not tools_ec.geo_columns.loc[lambda row: row.A_geo == geo_col, 'analysis_parameter'].iloc[0]:
                print("\x1b[31m", f"{geo_col} not found in given exp_ec", "\x1b[0m")
            continue
        data_ec.loc[:, j_geo_col] = (data_ec.I__A * 1000) / (exp_ec.loc[:, geo_col] / 100)

    # compatibility with user application of get_data_ec (instead of get_exp_sfc_icpms
    if index_cols is not None:
        new_index_cols = [index_col for index_col in index_cols if index_col not in data_ec.reset_index().columns]

        data_ec = data_ec.join(exp_ec.loc[:, new_index_cols], on='id_exp_sfc')\
                   .reset_index()\
                   .set_index(index_cols + ['id_data_ec']) \
                   .sort_index()

    t_2 = datetime.datetime.now()
    print("Done in ", t_2 - t_1, " - data_ec_analysis python VIEW substitution")
    return data_ec


def get_data_icpms(exp_icpms,
                   match_ec_icpms=None,
                   geo_cols=None,
                   t_start_shift__s=0,
                   t_end_shift__s=0,
                   add_data_without_corresponding_ec=True,
                   add_cond=None,
                   apply_calibration_internalstandardfitting=True,
                   ):
    """
    Substitutes the database VIEW data_icpms_sfc_analysis.
    :param exp_icpms: pd.DataFrame
        ICPMS experimental dataframe from VIEW exp_icpms_sfc_expanded
    :param name_table: str
        deprecated, just for compatibility reasons
    :param match_ec_icpms: pd.DataFrame
        dataframe from VIEW match_exp_sfc_exp_icpms.
        Additionally needed to match data_icpms datapoints to SFC experiment.
        Use method match_exp_sfc_exp_icpms(exp_icpms, ...) for thispurpose
    :param geo_cols: str, list of str or None
        specifiy which geo_cols to use for current density calculation, or specify j_geo_cols
    :param t_start_shift__s: float
        substitutes expand_time_window will match n seconds before or after the start timestamp
        of the icpms experiment to th ec experiment
    :param t_end_shift__s: float
        substitutes expand_time_window will match icpms n seconds before or after the end timestamp
        of the icpms experiment to th ec experiment
    :param add_data_without_corresponding_ec: bool, default True
        only applies if match_exp_sfc_exp_icpms is given,
        True: will select also data which has no corresponding ec experiment
        False: ignores sorting out non SFC data
    :param add_cond: str,
        additional condition for requesting the data dataframe.
        For example, use this to select only a specific cycle in a CV experiment
    :param apply_calibration_internalstandardfitting bool, default True
        this will calculate count ratio, concentration and mass flow rate
        based on internalstandardfitting and calibration results
    :return: data_icpms as pd.DataFrame
    """
    geo_cols = tools.check_type(
        "geo_cols",
        geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )

    if type(match_ec_icpms) == str:
        print("\x1b[33m", "The frst two non-keyword arguments are exp_icpms and match_ec_icpms. "
                          "The parameter 'name_table' is not required anymore.", "\x1b[0m")
        match_ec_icpms = None
    match_ec_icpms = tools.check_type(
        "match_ec_icpms",
        match_ec_icpms,
        allowed_types=[pd.DataFrame],
        allowed_None=True,
    )

    data_icpms = get_data(exp_icpms,
                          'data_icpms',  # querying from view is too slow
                          add_cond=add_cond,
                          )


    t_1 = datetime.datetime.now()



    required_cols_exp = ['name_setup_sfc', 't_start__timestamp_sfc_pc', 't_delay__s',
                         'calibration_intercept__countratio', 'calibration_slope__countratio_mug_L',
                         'flow_rate_real__mul_min']
    if any([col not in exp_icpms.columns for col in required_cols_exp]):
        print("\x1b[31m", "Missing columns in exp_icpms to calculate standard columns for data_icpms."
                          " Ensure you handover exp_icpms as first parameter of this function"
                          " Required: %s" % required_cols_exp,
              "\x1b[0m")
        return data_icpms

    data_icpms = data_icpms.join(exp_icpms.name_setup_sfc)
    data_icpms.loc[:, 't__timestamp_sfc_pc'] = exp_icpms.t_start__timestamp_sfc_pc + pd.to_timedelta(data_icpms.t__s,
                                                                                                     unit='s')
    data_icpms.loc[:, 't_delaycorrected__timestamp_sfc_pc'] = exp_icpms.t_start__timestamp_sfc_pc \
                                                              + pd.to_timedelta(data_icpms.t__s, unit='s') \
                                                              - pd.to_timedelta(exp_icpms.t_delay__s, unit='s')

    if apply_calibration_internalstandardfitting:
        data_icpms_internalstandard_fitting = get_data(exp_icpms,
                                                       'data_icpms_internalstandard_fitting',
                                                       # querying from view is too slow
                                                       )
        data_icpms = data_icpms.join(data_icpms_internalstandard_fitting)
        data_icpms.loc[:, 'a_is__countratio'] = data_icpms.counts_analyte / data_icpms.counts_internalstandard_fitted
        data_icpms.loc[:, 'c_a__mug_L'] = (
                                                      data_icpms.a_is__countratio - exp_icpms.calibration_intercept__countratio) / exp_icpms.calibration_slope__countratio_mug_L
        data_icpms.loc[:, 'dm_dt__ng_s'] = data_icpms.c_a__mug_L * exp_icpms.flow_rate_real__mul_min / (1000 * 60)
    else:
        data_icpms.loc[:, 'a_is__countratio'] = np.nan
        data_icpms.loc[:, 'c_a__mug_L'] = np.nan
        data_icpms.loc[:, 'dm_dt__ng_s'] = np.nan


    # special timestamp matching if exp_icpms is match_exp_sfc_exp_icpms
    # if exp_icpms.reset_index().columns.isin(["id_exp_sfc", "id_exp_icpms"]).sum() == 2:
    required_cols_exp = ['id_exp_icpms', 'id_exp_sfc']
    if match_ec_icpms is None:
        print("\x1b[31m", "You must handover match_ec_icpms to be able to match sfc experiments",
              "\x1b[0m")
    elif any([col not in match_ec_icpms.columns for col in required_cols_exp]):
        print("\x1b[31m", "Missing columns in match_ec_icpms to be able to match sfc experiments."
                          " Required: %s" % required_cols_exp,
              "\x1b[0m")
    else:
        # t_3 = datetime.datetime.now()
        data_icpms_index_cols = data_icpms.index.names
        data_icpms = data_icpms.reset_index().set_index(["id_exp_icpms", "id_data_icpms"]).sort_index()
        start_shift = pd.Timedelta(seconds=t_start_shift__s)
        end_shift = pd.Timedelta(seconds=t_end_shift__s)
        for index, row in match_ec_icpms.reset_index().iterrows():
            # old (=slower) versions for time matching
            # v1 - not correct exp_icpms and exp_ec matching
            # data_icpms.loc[(row.id_exp_icpms,
            #                slice((data_icpms.t_delaycorrected__timestamp_sfc_pc
            #                - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()[1],
            #                    (data_icpms.t_delaycorrected__timestamp_sfc_pc
            #                    - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin()[1]
            #                 )), 'id_exp_sfc'] = row.id_exp_sfc

            # v2  - not correct exp_icpms and exp_ec matching, problems when two icpms measurements simultaneously
            # data_icpms2.loc[(data_icpms2.t_delaycorrected__timestamp_sfc_pc
            #                   - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin():\
            #                    (data_icpms2.t_delaycorrected__timestamp_sfc_pc
            #                    - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin()
            #                 , 'id_exp_sfc'] = row.id_exp_sfc

            # v3 select all icpms experiments belonging to the looped exp_ec from these compare timestamps
            # - faster and correct matching
            # data_icpms.loc[(row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()):\
            #       (row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].t_delaycorrected__timestamp_sfc_pc
            #           - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin())
            #       , 'id_exp_sfc'] = row.id_exp_sfc

            # v3.2 with grab lines with matching id_exp_icpms only once
            # and calculate timedelta for start and end_shift before for loop - even a bit faster
            a = data_icpms.loc[row.id_exp_icpms]
            data_icpms.loc[
            (
                row.id_exp_icpms,
                (
                        a.t_delaycorrected__timestamp_sfc_pc
                        - (pd.to_datetime(row.t_start__timestamp) + start_shift)
                )
                    .abs()
                    .idxmin(),
            ): (
                row.id_exp_icpms,
                (
                        a.t_delaycorrected__timestamp_sfc_pc
                        - (pd.to_datetime(row.t_end__timestamp) + end_shift)
                )
                    .abs()
                    .idxmin(),
            ),
            "id_exp_sfc",
            ] = row.id_exp_sfc

            # v3.3 placing correction of start and end time before for loop --> slower
            # df_match.loc[:,'t_start__timestamp'] = pd.to_datetime(df_match.t_start__timestamp)
            #                                       - pd.Timedelta(seconds=500)
            # df_match.loc[:,'t_end__timestamp'] = pd.to_datetime(df_match.t_start__timestamp)
            #                                       + pd.Timedelta(seconds=500)

            # v4.1 with grouping index, reduce redudndat time columns appear with multiple measured analyte elements,
            # but slower
            # data_icpms.loc[(row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].groupby(level=0)
            # .t_delaycorrected__timestamp_sfc_pc.first() - (pd.to_datetime(row.t_start__timestamp)
            # -pd.Timedelta(seconds=0))).abs().idxmin()):\
            #               (row.id_exp_icpms, (data_icpms.loc[row.id_exp_icpms].groupby(level=0)
            #               .t_delaycorrected__timestamp_sfc_pc.first() - (pd.to_datetime(row.t_end__timestamp)
            #               +pd.Timedelta(seconds=0))).abs().idxmin())
            #               , 'id_exp_sfc'] = row.id_exp_sfc

            # v4.2 similar but with .index.duplicated instead
            # a=data_icpms.loc[row.id_exp_icpms]
            # data_icpms.loc[(row.id_exp_icpms,
            #               (a.loc[~a.index.duplicated(keep='first')].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_start__timestamp)-pd.Timedelta(seconds=0))).abs().idxmin()):\
            #               (row.id_exp_icpms,
            #               (a.loc[~a.index.duplicated(keep='first')].t_delaycorrected__timestamp_sfc_pc
            #               - (pd.to_datetime(row.t_end__timestamp)+pd.Timedelta(seconds=0))).abs().idxmin())
            #               , 'id_exp_sfc'] = row.id_exp_sfc

        # remove unmmatched data if requested by add_data_without_corresponding_ec
        if not add_data_without_corresponding_ec:
            data_icpms = data_icpms.loc[~data_icpms.id_exp_sfc.isna(), :]

        # add A_geo_cols
        if geo_cols is None:
            geo_cols = [col for col in tools_ec.geo_columns.A_geo.values.tolist() if col in match_ec_icpms.columns]

        #elif geo_cols is not None:
        dmdt_geo_cols = tools_ec.geo_columns.loc[
            tools_ec.geo_columns.A_geo.isin(geo_cols), 'dm_dt_S'].values.tolist()
        #else:
        #    geo_cols = tools_ec.geo_columns.A_geo.values.tolist()  # all columns
        #    dmdt_geo_cols = tools_ec.geo_columns.dm_dt_S.values.tolist()  # all columns

        # if join columns not overlapping --> check that all dataframes not series
        # data=data.reset_index().set_index(index_cols).sort_index() # index will be set later
        data_joined = data_icpms.join(
            match_ec_icpms.set_index("id_exp_sfc").sort_index().loc[:, geo_cols],
            on="id_exp_sfc",
        )

        for geo_col, dmdt_geo_col in zip(geo_cols, dmdt_geo_cols):
            if geo_col not in match_ec_icpms.columns:
                print("\x1b[31m", f"{geo_col} not found in given match_ec_icpms", "\x1b[0m")
                continue
            data_icpms.loc[:, dmdt_geo_col] = data_joined.dm_dt__ng_s / (data_joined.loc[:, geo_col] / 100)

        # might improve performance, but jupyter crashes when executing
        # data = data.join((data.join(match_ec_icpms.set_index('id_exp_sfc').sort_index().loc[:, A_geo_cols]
        # .rename(columns=plot.geo_columns.set_index('A_geo').to_dict()['dm_dt_S']), on='id_exp_sfc')
        # .loc[:, plot.get_geo_column(A_geo_cols, 'A_geo', 'dm_dt_S')]/ 100).rdiv(data.loc[:, ['dm_dt__ng_s', ]]
        # .values))

        data_icpms = data_icpms.reset_index().set_index(data_icpms_index_cols).sort_index()

        # t_4 = datetime.datetime.now()
        # print("Timestamp matching in ", t_4 - t_3)
    t_2 = datetime.datetime.now()
    print("Done in ", t_2 - t_1, " - data_icpms_sfc_analysis python VIEW substitution")
    return data_icpms


def match_exp_sfc_exp_icpms(
    df_exp, overlay_cols=None, add_cond=None, A_geo_cols=None, add_cols=None
):
    """
    Get a DataFrame which matches sfc and icpms experiments. Matching on experiment level is required to
    optimize the matching on datapoint level later (match an sfc experiment to each icpms datapoint), is performed in+
    evaluation.utils.db.get_data()
    :param df_exp: pd.DataFrame
        either exp_icpms or exp_ec, depending on whether sfc-icpms experiments are selected by icpms or ec experiments
    :param overlay_cols: str or list of str or None
        name of index columns used to overlay multiple experiments
    :param add_cond: str
        additional condition to subselect specific experiments
    :param A_geo_cols:
        geometric columns (information on electrode size) which should be handed over to icpms experiments
        (to calculate geometric corrected icpms mass transfer rates)
    :param add_cols:
        additonal columns which shoul dbe handed over from icpms or ec to the other
    :return: pd.DataFrame matching sfc and icpms experiment
    """
    if overlay_cols is None:
        overlay_cols = []
    if len(df_exp.index) == 0:
        display(df_exp)
        raise ValueError('No experiments selected to match')

    A_geo_cols = plot.get_geo_column(
        A_geo_cols, "A_geo", "A_geo", analysis_parameter=False
    )  # check that geo col exist, remove columns with analysis_parameter=True
    if A_geo_cols is not None:
        A_geo_cols = A_geo_cols if isinstance(A_geo_cols, list) else [A_geo_cols]
    index_names = df_exp.reset_index().columns
    index_names = index_names[index_names.isin(["id_exp_sfc", "id_exp_icpms"])]
    if len(index_names) > 1:  # len(index_names) >1:
        print(index_names)
        warnings.warn(
            "id_exp_sfc and id_exp_icpms found in df_exp. Did you already match? Just returned df_exp dataframe"
        )
        return df_exp
    elif len(index_names) == 0:
        raise Exception(
            "id_exp_sfc and id_exp_icpms not found in columns of dfexp. "
            "Please use propper experiment dataframe as df_exp"
        )
    # print(index_names)
    index_name = index_names[0]
    index_values = df_exp.reset_index().loc[:, str(index_name)].unique()  # .to_list()

    sql_query = (
        """SELECT id_exp_sfc, t_start__timestamp,t_end__timestamp, id_exp_icpms"""
        + ((", " + ", ".join(A_geo_cols)) if A_geo_cols is not None else "")
        + ((", " + ", ".join(add_cols)) if add_cols is not None else "")
        + """  FROM match_exp_sfc_exp_icpms m   
           WHERE ("""
        + str(index_name).replace("'", "`")
        + """)   IN ("""
        + str(list(index_values))[1:-1]
        + ")"
        + (" AND " + str(add_cond) if add_cond is not None else "")
        + ";"
    )
    print(sql_query)

    #checked_t_end__timestamp = 0
    #while checked_t_end__timestamp <= 1:
    with connect().begin() as con:
        df = pd.read_sql(
            sql_query,
            con=con,
        )

    update_t_end__timestamp = False
    if index_name == 'id_exp_sfc':
        if df.t_end__timestamp.isna().any():
            update_t_end__timestamp = True
    elif index_name == 'id_exp_icpms':
        if not df_exp.reset_index().id_exp_icpms.isin(df.reset_index().id_exp_icpms).all():
            update_t_end__timestamp = True
    if update_t_end__timestamp:  # df.t_end__timestamp.isna().any() or len(df.index) == 0:
        # run this after the database query and check whether any t_end_timestamp is missing,
        # if yes run query again (just once)
        # engine = db.connect('hte_write')
        # db.call_procedure(engine, 'update_exp_sfc_t_end__timestamp')
        print('check and update exp_sfc_t_end__timestamp')
        tools_ec.update_exp_sfc_t_end__timestamp()
        # checked_t_end__timestamp += 1
        # ugly but while lopp doesn#t work
        with connect().begin() as con:
            df = pd.read_sql(
                sql_query,
                con=con,
            )

    if len(overlay_cols) > 0:
        df = df.join(
            df_exp.reset_index().set_index(index_name).loc[:, overlay_cols],
            on=index_name,
        )

    df = df_cols_dtype_to_timestamp(df)

    return df


def get_exp_ec_dataset(exp_ec, con=None):
    """
    Shorthand function to get a single dataset definer from a list of ec experiments
    :param exp_ec: pd.DataFrame
        list of ec experiments
    :param con: sqlalchemy.connection or None
        database connection
        if None a new will be initialized
    :return: exp_ec_dataset_definer
    """
    if con is None:
        con = connect()
    ids_exp_sfc = exp_ec.reset_index().id_exp_sfc.tolist()
    ids_exp_sfc_str = ", ".join(["%s"] * len(ids_exp_sfc))

    exp_ec_datasets_definer = query_sql(
        """SELECT *
           FROM exp_ec_datasets_definer
           WHERE  id_exp_ec_dataset IN (SELECT id_exp_ec_dataset FROM exp_ec_datasets_definer
                                        GROUP BY id_exp_ec_dataset
                                        HAVING COUNT(*)= %s)
                                        # Exclude datasets with more or less id_exp_sfc than selected                
               AND  id_exp_ec_dataset NOT IN (SELECT id_exp_ec_dataset FROM exp_ec_datasets_definer
                                              WHERE id_exp_sfc NOT IN (""" + ids_exp_sfc_str + """))
                                        # Exclude datasets with other id_exp_sfc than selected
        ; """,
        params=[len(ids_exp_sfc)] + ids_exp_sfc,
        method="pandas",
        con=con,
    )
    return exp_ec_datasets_definer


def get_exp_ec_datasets(exp_ec, con=None):
    """
    Shorthand function to get datasets definer from a list of ec experiments
    :param exp_ec: pd.DataFrame
        list of ec experiments
    :param con: sqlalchemy.connection or None
        database connection
        if None a new will be initialized
    :return: exp_ec_dataset_definer
    """
    if con is None:
        con = connect()
    ids_exp_sfc = exp_ec.reset_index().id_exp_sfc.tolist()
    ids_exp_sfc_str = ", ".join(["%s"] * len(ids_exp_sfc))

    exp_ec_datasets_definer = query_sql(
        """SELECT id_exp_sfc, id_exp_ec_dataset, name_exp_ec_dataset
           FROM exp_ec_datasets_definer
                LEFT JOIN exp_ec_datasets  USING (id_exp_ec_dataset)
           WHERE  id_exp_ec_dataset NOT IN (SELECT id_exp_ec_dataset FROM exp_ec_datasets_definer
                                              WHERE id_exp_sfc NOT IN (""" + ids_exp_sfc_str + """))
                                        # Exclude datasets with other id_exp_sfc than selected
        ; """,
        params=ids_exp_sfc,
        method="pandas",
        con=con,
    )
    return exp_ec_datasets_definer


def get_ana_icpms_sfc_fitting(exp_ec, exp_icpms, id_fit=0, show_result_svg=True):
    """
    Shorthand function to get sfc icpms peak fitting reuslts stored in ana_icpms_sfc_fitting
    from ec and icpms experiemnt list
    peak details can be retrieved by db.get_data(ana_icpms_sfc_fitting, name_table='ana_icpms_sfc_fitting_peaks')
    :param exp_ec: pd.DataFrame
        EC experiment list
    :param exp_icpms: pd.DataFrame
        EC experiment list
    :param id_fit: int, optional Default 0
        index of the fit, id_fit!=0 if multiple fits are performed on the same dataset
    :param show_result_svg: boll, optional, Default True
        whether to show the result plot of the fitting procedure
        as linked in ana_icpms_sfc_fitting.file_path_plot_sfc_icpms_peakfit
    :return: ana_icpms_sfc_fitting
    """
    exp_ec_dataset_definer = get_exp_ec_dataset(exp_ec)
    dataset_sfc_icpms = pd.DataFrame(
        index=tools.multiindex_from_product_indices(
            exp_ec_dataset_definer.set_index("id_exp_ec_dataset").index.unique(),
            exp_icpms.index,
        )
    )
    dataset_sfc_icpms.loc[:, "id_fit"] = id_fit
    ana_icpms_sfc_fitting = get_exp(
        dataset_sfc_icpms, name_table="ana_icpms_sfc_fitting"
    )
    if show_result_svg:
        for path in ana_icpms_sfc_fitting.file_path_plot_sfc_icpms_peakfit.tolist():
            if os.path.isfile(path):
                display(SVG(filename=path))

    return ana_icpms_sfc_fitting


def get_exp_sfc_icpms(
    sql_ec=None,
    sql_icpms=None,
    id_exp_sfc=None,
    id_exp_ec_dataset=None,
    id_exp_icpms=None,
    name_isotope_analyte=None,
    name_isotope_internalstandard=None,
    overlay_cols=None,
    cols_ec_to_icpms=None,
    multiple_exp_ec=True,
    multiple_exp_ec_datasets=True,
    multiple_exp_icpms=True,
    multiple_exp_icpms_isotopes=True,
    join_exp_ec_dataset_to_exp_ec='aggregate',
    add_gravimetric=False,
    get_data_geo_cols=None,
    get_data_j_geo_cols=None,
    get_data_ec_add_cond=None,
    get_data_icpms_add_cond=None,
    get_data_icpms_t_start_shift__s=0,
    get_data_icpms_t_end_shift__s=0,
    get_data_icpms_add_data_without_corresponding_ec=False,
    add_match_ec_icpms=False,
    add_data_stability_analysis=False,
    get_data_stability_analysis_add_cond=None,
    debug=False,
):
    """
    Shorthand function to retrieve sfc icpms datasets by SQL query for ec experiments or icpms experiments
    or by specifiying one of the experiment indices.
    This function will query for:
        - exp_ec (from database VEIW exp_ec_expanded)
        - data_ec (using get_data_ec, which queries database TABLE data_ec and calculates some basic columns
        - exp_icpms (from database VEIW exp_icpms_sfc_expanded)
        - data_ec (using get_data_icpms, which queries database TABLE data_icpms and calculates some basic columns
        - match_ec_icpms (from database VEIW match_exp_sfc_icpms,
                            used to match both experiments data_icpms get column id_exp_sfc)
        - optional: data_stability_analysis (from database VIEW data_stability_analysis
                                             containing results from integration
                                             performed with icpms_update_exp.sfc_icpms_integration_analysis)
    :param sql_ec: str or None, default None
        SQL query for ec experiments
    :param sql_icpms: str or None, default None
        SQL query for icpms experiments
    :param id_exp_sfc: int, or list of int or None
        indices of sfc experiments
    :param id_exp_ec_dataset: int, or list of int or None
        indices of ec experiment datasets
    :param id_exp_icpms: int, or list of int or None
        indices of icpms experiments
    :param name_isotope_analyte: str, or list of str or None
        indices of icpms analyte isotopes, not yet implemented
    :param name_isotope_internalstandard: str, or list of str or None
        indices of icpms internalstandard isotopes, not yet implemented
    :param overlay_cols: str, or list of str or None
        exp_ec columns on which the experiments should be overlayed when time-synced
    :param cols_ec_to_icpms: overlay_cols: str, list of str, None, default None
        additional columns you want to transfer from exp_ec to exp_icpms
        so that you can use it for styling, labeling in the plot
    :param multiple_exp_ec: bool, Default True
        restrict selection to maximum one ec experiment
    :param multiple_exp_ec_datasets: bool, Default True
        restrict selection to maximum one ec experiment dataset
    :param multiple_exp_icpms: bool, Default True
        restrict selection to maximum one icpms experiment
    :param multiple_exp_icpms_isotopes: bool, Default True
        restrict selection to maximum one icpms isotope pair
    :param join_exp_ec_dataset_to_exp_ec: str or bool, Default 'aggregate'
        how to join id_exp_ec_dataset into exp_ec or not
        'aggregate': if id_exp_sfc belong to multiple experiment they will be stored in list
        'extend': if id_exp_sfc belong to multiple experiment a new row will be created (be aware of non-unique indices)
        'single': id_exp_ec_dataset will only be joined if all id_exp_sfc belong to one exp_ec_dataset,
            all other datasets will be ignored,
            if such a dataset is not existing, id_exp_ec_dataset will be populated with None
        'None': exp_ec_dataset is not joined
        True: 'aggregate'
        False: 'None'
    :param add_gravimetric: bool, default False
        whether to add gravimetric information to exp_ec and data_ec
    :param get_data_geo_cols: str, list of str or None
        specifiy which geo_cols to use for current density calculation, or specify j_geo_cols
    :param get_data_j_geo_cols: str, list of str or None
        specifiy which j_geo_cols to calculate, or specify geo_cols
    :param get_data_ec_add_cond: str or None, default None
        add_cond for get_data(., 'data_ec_analysis')
    :param get_data_icpms_add_cond: str or None, default None
        add_cond for get_data(., 'data_icpms_analysis')
    :param get_data_icpms_t_start_shift__s: float, default 0
        substitutes expand_time_window will match n seconds before or after the start timestamp
        of the icpms experiment to th ec experiment
    :param get_data_icpms_t_end_shift__s: float, default 0
        substitutes expand_time_window will match icpms n seconds before or after the end timestamp
        of the icpms experiment to th ec experiment
    :param get_data_icpms_add_data_without_corresponding_ec: bool, Default False
        whether to add icpms data during which no ec experiment was performed
    :param add_match_ec_icpms: bool, default False
        whether to add match_ec_icpms to output
    :param add_data_stability_analysis: bool, default False
        whether to query for data_stability_analysis
    :param get_data_stability_analysis_add_cond:
    tr or None, default None
        add_cond for get_data(., 'data_stability_analysis')
    :param debug: bool, default False
        debug information, especicall for get_exp()
    :return: exp_ec, data_ec, exp_icpms, data_icpms, (data_stability_analysis)
        each as pd.DataFrame
    """
    id_exp_sfc = tools.check_type(
        "id_exp_sfc",
        id_exp_sfc,
        allowed_types=[int, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    id_exp_ec_dataset = tools.check_type(
        "id_exp_ec_dataset",
        id_exp_ec_dataset,
        allowed_types=[int, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    id_exp_icpms = tools.check_type(
        "id_exp_icpms",
        id_exp_icpms,
        allowed_types=[int, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    name_isotope_analyte = tools.check_type(
        "name_isotope_analyte",
        name_isotope_analyte,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    name_isotope_internalstandard = tools.check_type(
        "name_isotope_internalstandard",
        name_isotope_internalstandard,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    overlay_cols = tools.check_type(
        "overlay_cols",
        overlay_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    if cols_ec_to_icpms is None:
        cols_ec_to_icpms = []
    cols_ec_to_icpms = tools.check_type(
        "cols_ec_to_icpms",
        cols_ec_to_icpms,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=False,
    )

    join_exp_ec_dataset_to_exp_ec = tools.check_type(
        "join_exp_ec_dataset_to_exp_ec",
        join_exp_ec_dataset_to_exp_ec,
        allowed_types=[str, bool],
        allowed_None=False,
    )

    # geo columns
    get_data_geo_cols = tools.check_type(
        "get_data_geo_cols",
        get_data_geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    get_data_j_geo_cols = tools.check_type(
        "get_data_j_geo_cols",
        get_data_j_geo_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=True,
    )
    if get_data_geo_cols is not None:
        get_data_j_geo_cols = tools_ec.geo_columns.loc[tools_ec.geo_columns.A_geo.isin(get_data_geo_cols), 'j_geo'].values.tolist()
    elif get_data_j_geo_cols is not None:
        get_data_geo_cols = tools_ec.geo_columns.loc[tools_ec.geo_columns.j_geo.isin(get_data_j_geo_cols), 'A_geo'].values.tolist()
    else:
        get_data_geo_cols = tools_ec.geo_columns.A_geo.values.tolist()
        get_data_j_geo_cols = tools_ec.geo_columns.j_geo.values.tolist()

    if type(join_exp_ec_dataset_to_exp_ec) == bool:
        join_exp_ec_dataset_to_exp_ec = 'aggregate' if join_exp_ec_dataset_to_exp_ec else 'None'
    if join_exp_ec_dataset_to_exp_ec not in ['aggregate', 'extend', 'single', 'None']:
        raise Exception("join_exp_ec_dataset_to_exp_ec must be one of ['aggregate', 'extend', 'single', 'None']")

    input_params = [sql_ec, id_exp_ec_dataset, id_exp_sfc, sql_icpms, id_exp_icpms]
    if sum([val is not None for val in input_params]) > 1:
        print(
            "\x1b[31m",
            "You can only specify one input parameter to get data."
            "The first in this order will be taken, the rest ignored: "
            "[sql_ec, id_exp_ec_dataset, id_exp_sfc, sql_icpms, id_exp_icpms]",
            "\x1b[0m",
        )

    # Init with None
    exp_ec_datasets_definer = None

    if any([param is not None for param in [sql_ec, id_exp_ec_dataset, id_exp_sfc]]):

        if sql_ec is not None:
            exp_ec = get_exp(sql_ec,
                             index_col=["id_exp_sfc"],
                             debug=debug,)

        elif id_exp_ec_dataset is not None:
            exp_ec_datasets_definer = get_exp(
                pd.DataFrame(
                    id_exp_ec_dataset, columns=["id_exp_ec_dataset"]
                ).set_index("id_exp_ec_dataset"),
                name_table="exp_ec_datasets_definer",
                #join_col=["id_exp_ec_dataset"],
                index_col=["id_exp_ec_dataset"],
                             debug=debug,
            )
            exp_ec = get_exp(
                exp_ec_datasets_definer,
                name_table="exp_ec_expanded",
                join_col=["id_exp_sfc"],
                index_col=["id_exp_sfc"],
                             debug=debug,
            )
            join_exp_ec_dataset_to_exp_ec = 'extend'  # nothing to aggregate

        elif id_exp_sfc is not None:
            exp_ec = get_exp(
                pd.DataFrame(id_exp_sfc, columns=["id_exp_sfc"]).set_index(
                    "id_exp_sfc"
                ),
                name_table="exp_ec_expanded",
                             debug=debug,
            )
        else:
            raise ValueError("Not enough parameters to get sfc icpms data!")

        # exp_ec = exp_ec.reset_index().set_index(overlay_cols + ["id_exp_sfc"])

        # ICP-MS experiments
        match_ec_icpms = match_exp_sfc_exp_icpms(exp_ec,
                                                 #overlay_cols=overlay_cols,
                                                 A_geo_cols=get_data_geo_cols
                                                 )
        exp_icpms = get_exp(
            match_ec_icpms,
            "exp_icpms_sfc_expanded",
            groupby_col=["id_exp_icpms"],
            index_col=["id_exp_icpms", "name_isotope_analyte", "name_isotope_internalstandard"],
            join_col=["id_exp_icpms"],
                             debug=debug,
        )

    elif any([param is not None for param in [sql_icpms, id_exp_icpms, ]]):
        #raise Exception("Not developed yet!")
        if sql_icpms is not None:
            exp_icpms = get_exp(sql_icpms)

        elif id_exp_icpms is not None:
            exp_icpms = get_exp(
                pd.DataFrame(id_exp_icpms, columns=["id_exp_icpms"]).set_index(["id_exp_icpms",]),
                name_table="exp_icpms_sfc_expanded",
                groupby_col='id_exp_icpms',
                             debug=debug,
            )
        else:
            raise ValueError("Not enough parameters to get sfc icpms data!")
        match_ec_icpms = match_exp_sfc_exp_icpms(exp_icpms,
                                                 A_geo_cols=get_data_geo_cols)
        exp_ec = get_exp(match_ec_icpms,
                         name_table='exp_ec_expanded',
                         index_col=['id_exp_sfc'],
                             debug=debug,)

    else:
        raise Exception("Not enough parameters to get sfc icpms data!")

    # exp_ec_datasets
    if exp_ec_datasets_definer is None:
        exp_ec_datasets_definer = get_exp_ec_datasets(exp_ec)
    if join_exp_ec_dataset_to_exp_ec != 'None' and not exp_ec_datasets_definer.empty:
        if join_exp_ec_dataset_to_exp_ec == 'aggregate':
            exp_ec = exp_ec.join(plot.aggregate_list_unique(exp_ec_datasets_definer,
                                                            on='id_exp_sfc',
                                                            print_report=False,
                                                            aggregate_prev_index_cols=False,
                                                            ),
                                 on='id_exp_sfc',
                                 how='left')
        elif join_exp_ec_dataset_to_exp_ec == 'extend':
            exp_ec = exp_ec.join(exp_ec_datasets_definer.reset_index().set_index('id_exp_sfc'),
                                 on='id_exp_sfc',
                                 how='outer')
        elif join_exp_ec_dataset_to_exp_ec == 'single':
            # number of id_exp_sfc for each dataset
            counts_id_exp_sfc = exp_ec_datasets_definer.groupby('id_exp_ec_dataset')['id_exp_sfc'].count()
            # dataset which has all id_exp_sfc (should only be None or one)
            exp_ec_datasets_all_id_exp_sfc = counts_id_exp_sfc.loc[counts_id_exp_sfc == len(exp_ec.index)].index
            if len(exp_ec_datasets_all_id_exp_sfc) == 0:
                print('No exp_ec_dataset matching all id_exp_sfc, set id_exp_ec_dataset = None')
                exp_ec.loc[:, 'id_exp_ec_dataset'] = None
                exp_ec.loc[:, 'name_exp_ec_dataset'] = None
            else:
                if len(exp_ec_datasets_all_id_exp_sfc) > 1:
                    print(
                        'Multiple exp_ec_dataset matching all id_exp_sfc. This should not occur, please refer to admin.',
                        exp_ec_datasets_all_id_exp_sfc[0], ' is used.')
                # print('One exp_ec_dataset matching all id_exp_sfc', exp_ec_datasets_all_id_exp_sfc[0])
                exp_ec.loc[:, 'id_exp_ec_dataset'] = exp_ec_datasets_all_id_exp_sfc[0]
                exp_ec.loc[:, 'name_exp_ec_dataset'] = exp_ec_datasets_definer.loc[(exp_ec_datasets_definer.id_exp_ec_dataset
                                                                          == exp_ec_datasets_all_id_exp_sfc[0]
                                                                          ),
                                                                         'name_exp_ec_dataset']\
                                                                    .unique()[0]

    # get data ec
    data_ec = get_data_ec(
        exp_ec,
        geo_cols=get_data_geo_cols,
        #j_geo_cols=get_data_j_geo_cols, # geo_cols is sufficient see top
        #join_cols=["id_exp_sfc"],
        #index_cols=overlay_cols + ["id_exp_sfc", "id_data_ec"],
        add_cond=get_data_ec_add_cond,
    )

    # get data icpms
    if name_isotope_analyte is not None:
        exp_icpms = exp_icpms.loc[exp_icpms.index.get_level_values(level='name_isotope_analyte')
                                           .isin(name_isotope_analyte), :]
    if name_isotope_internalstandard is not None:
        exp_icpms = exp_icpms.loc[exp_icpms.index.get_level_values(level='name_isotope_internalstandard')
                                           .isin(name_isotope_internalstandard), :]

    # icpms data
    data_icpms = get_data_icpms(
        exp_icpms,
        match_ec_icpms,
        #join_cols=["id_exp_icpms"],
        #join_overlay_cols=["id_exp_sfc"],
        #index_cols=overlay_cols
        #+ [
        #    "id_exp_icpms",
        #    "name_isotope_analyte",
        #    "name_isotope_internalstandard",
        #    "id_data_icpms",
        #],
        add_cond=get_data_icpms_add_cond,
        t_start_shift__s=get_data_icpms_t_start_shift__s,
        t_end_shift__s=get_data_icpms_t_end_shift__s,
        add_data_without_corresponding_ec=get_data_icpms_add_data_without_corresponding_ec,
    )


    # check number of experiments
    # exp_ec and exp_ec_dataset
    if not multiple_exp_ec:
        if len(exp_ec.reset_index().id_exp_sfc.unique()) > 1:
            display(exp_ec)
            raise Exception("More than one ec experiment selected")
    if not multiple_exp_ec_datasets:
        #if (len(exp_ec_datasets_definer.reset_index().id_exp_ec_dataset.unique()) > 1 ):
        if 'id_exp_ec_dataset' not in exp_ec.columns:
            raise Exception("multiple_exp_ec_datasets cannot be True when no exp_ec_datasets joined.")
        if len(exp_ec.id_exp_ec_dataset.astype(str).unique()) > 1:
            display(exp_ec)
            raise Exception("More than one ec experiment dataset selected")

    # icpms
    if not multiple_exp_icpms:
        if len(exp_icpms.reset_index().id_exp_icpms.unique()) > 1:
            display(exp_icpms)
            raise Exception("More than one icpms experiment dataset selected")
    if not multiple_exp_icpms_isotopes:
        if len(exp_icpms.reset_index().name_isotope_analyte.unique()) > 1:
            display(exp_icpms)
            raise Exception("More than one name_isotope_analyte selected")
    if not multiple_exp_icpms_isotopes:
        if len(exp_icpms.reset_index().name_isotope_internalstandard.unique()) > 1:
            display(exp_icpms)
            raise Exception("More than one name_isotope_internalstandard selected")

    # gravimetric current
    if add_gravimetric:
        exp_ec, data_ec = tools_ec.gravimetric_current(exp_ec,
                                                       data_ec,
                                                       j_geo_cols=get_data_j_geo_cols)

    # overlay_cols
    if overlay_cols is not None:
        exp_ec, data_ec, exp_icpms, data_icpms = overlay_exp_sfc_icpms(exp_ec,
                                                                       data_ec,
                                                                       exp_icpms,
                                                                       data_icpms,
                                                                       overlay_cols=overlay_cols,
                                                                       cols_ec_to_icpms=cols_ec_to_icpms,
                                                                       )
    else:
        # Time synchronization
        data_ec, data_icpms = plot.synchronize_timestamps(
            data_ec=data_ec,
            data_icpms=data_icpms,
            timestamp_col_ec="Timestamp",
            timestamp_col_icpms="t_delaycorrected__timestamp_sfc_pc",
            #overlay_index_cols=overlay_cols,
        )

    if not (exp_ec.name_user == current_user()).any():
        print(
            "\x1b[33m",
            "This is data from ",
            ", ".join(exp_ec.name_user.unique().tolist()),
            ". You have Read but no Write rights.",
            "\x1b[0m",
        )

    output = exp_ec, data_ec, exp_icpms, data_icpms

    # Add match_ec_icpms
    if add_match_ec_icpms:
        output += match_ec_icpms,

    # Add integration data
    if add_data_stability_analysis:
        if join_exp_ec_dataset_to_exp_ec == 'None':
            warnings.warn('data_stability_analysis can only be queried with exp_ec_dataset information. '
                          'join_exp_ec_dataset_to_exp_ec must not be "None".')
        else:
            data_stability_analysis = get_data(exp_ec,
                    'data_stability_analysis',
                    join_cols=['id_exp_ec_dataset'],
                    index_cols=['id_exp_ec_dataset'],
                    add_cond=get_data_stability_analysis_add_cond,
                    )
            output += data_stability_analysis,

    return output


def overlay_exp_sfc_icpms(exp_ec,
                          data_ec,
                          exp_icpms,
                          data_icpms,
                          overlay_cols=None,
                          cols_ec_to_icpms=None,
                          ):
    """
    Change index for ec and icpms DataFrames, and adjust time synchronization so that different experiments
    are overlayed according to the given overlay_cols.
    :param exp_ec: pd.DataFrame as received from VIEW exp_ec_expanded
    :param data_ec: pd.DataFrame as received from function db.get_data_ec
    :param exp_icpms: pd.DataFrame as received from VIEW exp_icpms_sfc_expanded
    :param data_icpms: pd.DataFrame as received from function db.get_data_icpms
    :param overlay_cols: str, list of str, None, default None
        columns of exp_ec_expanded which should be in the index of your experiment.
        Choose columns with the parameters you changed between experiments you want to overlay.
        Example:
            - You want to compare different spots of same catalyst use id_spot
            - You want to compare different catalysts use id_sample (or any column in samples table which specifies cat
        None: nothing will change
    :param cols_ec_to_icpms: overlay_cols: str, list of str, None, default None
        additional columns you want to transfer from exp_ec to exp_icpms
        so that you can use it for styling, labeling in the plot
    :return: exp_ec_overlay, data_ec_overlay, exp_icpms_overlay, data_icpms_overlay
    """
    if overlay_cols is None:
        overlay_cols = []
    overlay_cols = tools.check_type(
        "overlay_cols",
        overlay_cols,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=False,
    )
    if cols_ec_to_icpms is None:
        cols_ec_to_icpms = []
    cols_ec_to_icpms = tools.check_type(
        "cols_ec_to_icpms",
        cols_ec_to_icpms,
        allowed_types=[str, list, np.array],
        str_int_to_list=True,
        allowed_None=False,
    )

    # check input:
    df_index_cols = {'exp_ec': ['id_exp_sfc'],
                     'data_ec': ['id_exp_sfc', 'id_data_ec'],
                     'exp_icpms': ['id_exp_icpms', 'name_isotope_analyte', 'name_isotope_internalstandard', ],
                     'data_icpms': ['id_exp_icpms', 'name_isotope_analyte', 'name_isotope_internalstandard',
                                    'id_data_icpms'],
                     }

    for df, (name, idx_cols) in zip([exp_ec, data_ec, exp_icpms, data_icpms], df_index_cols.items()):
        if df.index.names != idx_cols:
            print("\x1b[33m", f"Wrong input index for {name}. This shouold be {str(idx_cols)}", "\x1b[0m")

    # add index
    if any([col not in exp_ec.columns for col in overlay_cols]):
        print("\x1b[33m",
              f"All overlay_cols must be contained in exp_ec, these are not:. {str([col for col in overlay_cols if col not in exp_ec.columns])}",
              "\x1b[0m")
    if any([col not in exp_ec.columns for col in cols_ec_to_icpms]):
        print("\x1b[33m",
              f"All cols_ec_to_icpms must be contained in exp_ec, these are not:. {str([col for col in cols_ec_to_icpms if col not in exp_ec.columns])}",
              "\x1b[0m")

    exp_ec_overlay = exp_ec.reset_index()\
                           .set_index(overlay_cols + df_index_cols['exp_ec'])\
                           .sort_index()
    data_ec_overlay = data_ec.join(exp_ec.loc[:, overlay_cols],
                                   on=df_index_cols['exp_ec'],
                                   how='inner',  # remove data for not selected exp_ec - important for time syncing
                                   ) \
        .reset_index() \
        .set_index(overlay_cols + df_index_cols['data_ec'])\
        .sort_index()

    match_sfc_icpms = exp_ec.join(match_exp_sfc_exp_icpms(exp_ec,
                                                 #overlay_cols=overlay_cols,
                                                 A_geo_cols='spots_spot_size__mm2'  # just to reduce computive power
                                                 )\
                                    .loc[:, ['id_exp_sfc', 'id_exp_icpms']].set_index('id_exp_sfc'))\
                            .groupby(overlay_cols+['id_exp_icpms']).first()\
                            .reset_index().loc[:, overlay_cols + cols_ec_to_icpms + ['id_exp_icpms']]\
                            .set_index('id_exp_icpms')

    exp_icpms_overlay = exp_icpms.reset_index() \
        .join(match_sfc_icpms,
              on='id_exp_icpms',
              #exp_ec_overlay.groupby(overlay_cols).first().reset_index().loc[:, overlay_cols + cols_ec_to_icpms],
              #how='cross'
              ) \
        .set_index(overlay_cols + df_index_cols['exp_icpms'])\
        .sort_index()
    data_icpms_overlay = data_icpms.join(exp_ec.loc[:, overlay_cols],
                                         on='id_exp_sfc',
                                         how='inner',
                                         # remove data for not selected exp_ec - important for time syncing
                                         ) \
        .reset_index() \
        .set_index(overlay_cols + df_index_cols['data_icpms'])\
        .sort_index()

    ## Time sync
    data_ec_overlay, data_icpms_overlay = plot.synchronize_timestamps(data_ec=data_ec_overlay,
                                                                      data_icpms=data_icpms_overlay,
                                                                      timestamp_col_ec='Timestamp',
                                                                      timestamp_col_icpms='t_delaycorrected__timestamp_sfc_pc',
                                                                      overlay_index_cols=overlay_cols)

    return exp_ec_overlay, data_ec_overlay, exp_icpms_overlay, data_icpms_overlay


def df_cols_dtype_to_timestamp(df):
    # transform VARCHAR(45) timestamp columns to datetime64[ns]
    # (necessary as LabView is unable to insert into Datetime columns)
    for timestamp_col in [
        col
        for col in df.columns
        if "timestamp" in col.lower() and df[col].dtypes == "O"
    ]:
        df.loc[:, timestamp_col] = df.loc[:, timestamp_col].astype("datetime64[ns]")
    return df


def derive_name_table(sql_exp, debug=False):
    """
    auto-derive the name of the table from which data is requested
    :param sql_exp: str
        sql query
    :param debug: bool
        print additional debug info
    :return: name of the table
    """
    name_table = (
        sql_exp.split("FROM")[1]
        .split("WHERE")[0]
        .strip(" \n\t;")
        .replace("hte_data.", "")
    )
    if debug:
        print("Derived name_table:", name_table)
    return name_table


def derive_name_base_table(name_table, debug=False):
    """
    derive the name of the base table from documentation_tables,
    this table name will be stored when experiments are linked to a publication in get_df_tracked_exports_exp
    :param name_table: str
        name of a table or view
    :param debug: bool
        print additional debug info
    :return: name of the base table
    """
    primary_keys = get_primarykeys()
    display(name_table) if debug else ""
    if name_table not in primary_keys.index:
        with connect().begin() as con:
            df_view_information = pd.read_sql(
                """SELECT name_table, name_base_table
                             FROM documentation_tables
                             WHERE table_type = 'VIEW'
                            """,
                con=con,
                index_col="name_table",
            )
        if name_table not in df_view_information.index:
            raise Exception(
                name_table
                + " is neither name of a table or view. "
                  "Correct your sql statement or specify name_table as function parameter."
            )
        elif df_view_information.loc[name_table, "name_base_table"] is None:
            warnings.warn(
                "Base table of view " + name_table + " is not defined. "
                "Please refer to admin to add to hte_data_documentation.view_information"
            )
        name_base_table = df_view_information.loc[name_table, "name_base_table"]

        if debug:
            print("Derived name_base_table:", name_base_table)
    else:
        name_base_table = name_table

    return name_base_table


def get_primarykeys(name_table=None,
                    table_schema=None,
                    engine=None,
                    ):
    """
    Get primary keys of all tables (name_table is None) or of specific table (name_table = str) in the database.
    :param name_table: str or None, optional, Default None
        get primary keys of all (None) or specific table
    :param table_schema: str, default=None specified by db_config
        name of the database schema of the table
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: primary_keys_grouped
    """
    return db_config.get_primarykeys(name_table=name_table,
                                     table_schema=table_schema,
                                     engine=engine)


def get_foreignkey_links(
    table_schema=None,
    referenced_table_schema=None,
    engine=None,
):
    """
    Get Foreign keys in sqlite database.
    :param table_schema: str, default=None specified by db_config
        name of the database schema of the table
    :param referenced_table_schema: str, default=None specified by db_config
        name of the database schema of the referenced table
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: foreign_key table as pd.DataFrame
    """
    return db_config.get_foreignkey_links(
        table_schema=table_schema,
        referenced_table_schema=referenced_table_schema,
        engine=engine,
    )


def get_views(table_schema=None,
              debug=False,
              engine=None,
):
    """
    Get a list of all views in database
    :param table_schema: table_schema: str, default=None specified by db_config'
        name of the database schema of the table
    :param debug: bool
        print additional debug info
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: list of all views in database
    """
    return db_config.get_views(table_schema=table_schema,
                               debug=debug,
                               engine=engine)


def get_create_view(name_view,
                    debug=False,
                    engine=None,
                    ):
    """
    get Create View statement from database
    :param name_view: name of the view
    :param debug: print extra info if True
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: create view statement
    """
    return db_config.get_create_view(name_view=name_view,
                                     debug=debug,
                                     engine=engine)


def get_views_sorted(table_schema=None,
                     debug=False,
                     engine=None,
                     remove_views=None,
                     ):
    """
    Read and sort views in the database
    :param table_schema: table_schema: str, default=None specified by db_config
        name of the database schema of the table
    :param debug: bool
        print additional debug info
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :param remove_views: list of str, default None --> ['documentation_columns', 'documentation_tables']
        list of views to be removed from list
    :return: list of all views in database
    """
    if remove_views is None:
        remove_views = ['documentation_columns', 'documentation_tables']
    view_tables_list = get_views(table_schema=table_schema,
                                 debug=debug,
                                 engine=engine,
                                 )

    view_references = {}
    for name_view in view_tables_list:
        # print(view)
        create_view_statement = get_create_view(name_view=name_view,
                                                debug=False,
                                                engine=engine)

        view_references[name_view] = np.array(view_tables_list)[
            [
                referenced_view in create_view_statement
                and referenced_view != name_view
                for referenced_view in view_tables_list
            ]
        ]
    # view_references

    view_tables_sorted = [
        key for key, val in view_references.items() if len(val) == 0
    ]  # views without any references

    while not all([view in view_tables_sorted for view in view_tables_list]):
        if debug:
            print(
                "Still missing views: ",
                np.array(view_tables_list)[
                    [view not in view_tables_sorted for view in view_tables_list]
                ],
            )
        for key, val in view_references.items():
            if key in view_tables_sorted:
                continue

            if all([referenced_view in view_tables_sorted for referenced_view in val]):
                view_tables_sorted.append(key)

    for view in remove_views:
        view_tables_sorted.remove(view)
        #view_tables_sorted.remove("documentation_columns")
        #view_tables_sorted.remove("documentation_tables")
    # print('All views sorted')
    # print(view_tables_sorted)
    return view_tables_sorted


def current_user():
    """
    get the current user name
    :return: current user name
    """
    return db_config.current_user()


def user_is_owner(index_col, index_value):
    """
    Check whether user is owner of a database entry specified index column name and value. Used to verify whether
    data processing is allowed.
    :param index_col: str
    :param index_value: int
    :return: bool
    """
    return db_config.user_is_owner(index_col, index_value)


def create_exp_ec_dataset(list_id_exp_ec_dataset,
                          name_exp_ec_dataset=None,
                          ):
    """
    create a exp_ec_dataset from a list of id_exp_sfc and insert into corresponding tables
    exp_ec_dataset and exp_ec_dataset_definer
    :param list_id_exp_ec_dataset: list of int
        list of id_exp_sfc
    :param name_exp_ec_dataset: str or None, default None
        give a name of the new dataset, if a new one is created
    :return: id_exp_ec_dataset of new or already created dataset
    """
    engine = connect("hte_processor")
    with engine.begin() as con:
        exp_ec_datasets = query_sql('''
                             SELECT id_exp_ec_dataset, ids_exp_sfc
                             FROM (
                                SELECT id_exp_ec_dataset, GROUP_CONCAT(id_exp_sfc) AS ids_exp_sfc
                                FROM (
                                        SELECT * 
                                        FROM exp_ec_datasets_definer  # removed hte_data.
                                        ORDER BY id_exp_ec_dataset, id_exp_sfc ASC 
                                        ) a
                                GROUP BY id_exp_ec_dataset
                                ) b
                                WHERE ids_exp_sfc = %s;
                            ''',
                                    params=[','.join([str(idx) for idx in list_id_exp_ec_dataset])],
                                    method='pandas',
                                    index_col='id_exp_ec_dataset',
                                    )
        if len(exp_ec_datasets) > 1:
            raise Exception('set of exp_ec_ experiments has multiple id_exp_ec_dataset. '
                            'This is not allowed. Please inform admin.')
        elif len(exp_ec_datasets) == 1:
            id_exp_ec_dataset = exp_ec_datasets.index.values[0]
            print(
                "\x1b[32m",
                "exp_ec_dataset already exists with id_exp_ec_dataset = ", id_exp_ec_dataset,
                "\x1b[0m",
            )
            return id_exp_ec_dataset
        else: #len(exp_ec_datasets)== 0
            # not initialized
            """ # no need to ask user - force hom to do so. User can still abrupt if he doesn't like
            if not user_input.user_input(
                    text="For uploading fit results, at first a new exp_ec_dataset "
                         "from selected ec experiments will be created. Continue? \n",
                    dtype="bool",
                    optional=False,
            ):
                print(
                    "\x1b[31m",
                    "Insertion canceled by user. "
                    "You need to create an exp_ec_dataset from selected ec experiments.",
                    "\x1b[0m",
                )
                raise Exception('Insertion canceled by user. 
                You need to create an exp_ec_dataset from selected ec experiments')
            """
            if name_exp_ec_dataset is None:
                name_exp_ec_dataset = user_input.user_input(
                        text="For uploading fit results, "
                             "at first a new exp_ec_dataset from selected ec experiments will be created. "
                             "Give a name of the dataset:",
                        dtype="str",
                        optional=False,
                    )
            #raise Exception('In')
            call_procedure(
                    engine,
                    "Reset_Autoincrement",
                    ["exp_ec_datasets", "id_exp_ec_dataset"],
                )

            # create new dataset
            id_exp_ec_dataset_created = int(
                insert_into(
                    con,
                    tb_name="exp_ec_datasets",
                    df=pd.DataFrame.from_dict(
                        {"name_exp_ec_dataset": [name_exp_ec_dataset]}
                    ),
                )["inserted_primary_key"][0]
            )

            # define wich id_exp_sfc belong to exp_ec_dataset
            pd.DataFrame(list_id_exp_ec_dataset, columns=['id_exp_sfc']).assign(
                    id_exp_ec_dataset=id_exp_ec_dataset_created
                ).set_index("id_exp_ec_dataset").to_sql(
                    "exp_ec_datasets_definer", con=con, if_exists="append"
                )
            print(
                    "\x1b[32m",
                    "Successfully inserted new exp_ec_dataset with id: ",
                    id_exp_ec_dataset_created,
                    "\x1b[0m",
                )
            return id_exp_ec_dataset_created


            # insert new id_exp_ec_dataset value into fitting dataframes - do this in your function
            #for df in [
            #    df_ana_icpms_sfc_fitting,
            #    df_ana_icpms_sfc_fitting_peaks,
            #]:
            #    index_name = df.index.names
            #    df.reset_index(inplace=True)
            #    df.id_exp_ec_dataset = id_exp_ec_dataset_created
            #    df.set_index(index_name, inplace=True)
