# If you like to use connection to MySQL database as default, replace the db_config.py file with this one.

"""
Scripts for MySQL database connection
Created in 2023
@author: Nico Röttcher
"""

from pathlib import Path
import sqlalchemy as sql
import pandas as pd
from mysql.connector import Error
import numpy as np
from ipywidgets import *
import warnings
import os


# from evaluation.utils import db
import evaluation.utils.db as db
# import evaluation.export.publication as publication_export
# import evaluation.export.tracking as tracking
import evaluation.export.db_config_binder as db_config_binder

MYSQL = True


DIR_JUPYTER = Path('/home/hte_admin/sciebo/jupyter')
DIR_JUPYTER_DEV = Path('/home/hte_admin/sciebo/jupyter_dev')

if os.getcwd().startswith(str(DIR_JUPYTER)+'/'):
    WHICH_JUPYTER = 'prod'
    DIR_JUPYTER_HOME = DIR_JUPYTER
elif os.getcwd().startswith(str(DIR_JUPYTER_DEV)+'/'):
    WHICH_JUPYTER = 'dev'
    DIR_JUPYTER_HOME = DIR_JUPYTER_DEV
else:
    WHICH_JUPYTER = None
    DIR_JUPYTER_HOME = None


def DIR_REPORTS():
    """
    Get directory to store reports,
    if called from a publication folder the processing reports folder of the publication will be used
    else the standard location on jupyter hub server
    :return: directory to store reports
    """
    from evaluation.export.publication import PublicationExport
    pub = PublicationExport()
    if pub.created and pub.export_type == 'PublicationExport':
        return pub.path_to_jupyter_folder / db_config_binder.REL_DIR_REPORTS
    else:
        return Path(r"/home/hte_admin/sciebo/jupyter/shared/03_processing_reports")


def connect(user="hte_read", echo=False, **kwargs_get_db_config):
    """
    method to connect to MySQL database reading database credential from file located up in the system
    :param user: str, optional default 'hte_read'
        name of the MySQL user to connect with to MySQL database
    :param echo: bool, optional default 'False
        whether to use echo parameter in sqlalchemy engine.
        This will print out detailed information on any interaction with the database
    :param **kwargs_get_db_config
        keyword arguments from get_db_config: database, host
    :return: sqlalchemy.engine
    """
    config = get_db_config(user=user, **kwargs_get_db_config)

    """
    # overwrite database  if given or working in developer jupyterhub
    if database is not None:
        config["database"] = database

    # In this version:
    # only the config["database"] is changed, this requires privilege for all hte_xx users also to dev databases,
    # This requires calling hte_data_documentation.user_hte_grant_privileges
    # within hte_data_documentation.create_db_copy_session_user 

    # if the current working directory is inside the jupyter_dev folder, the most current dev database will be used
    elif os.getcwd().split("/")[4] == "jupyter_dev":
        credential_file_dir = '/'.join(os.getcwd().split('/')[:6]) + '/db_session'
        if os.path.isdir(credential_file_dir):
            credential_files = os.listdir(credential_file_dir)
            modified_time = [os.path.getctime(os.path.join(credential_file_dir, filename)) for filename in
                             credential_files]
            if credential_files:
                latest_file_index = modified_time.index(max(modified_time))  # Index of the most recent file
                latest_credential_file = os.path.join(credential_file_dir, credential_files[latest_file_index])
                print(f"Taking the most recent credential file: {credential_files[latest_file_index]}")
                print("Fetching database name...")

                credentials = pd.read_csv(latest_credential_file, sep='\t')
                config["database"] = credentials['Databases'][0].split(',')[0]
            else:
                raise Exception("No credential file found in the directory.")
        else:
            if user != 'hte_read':  # avoid making any write changes to the production db
                raise Exception("No changes allowed to production database.")

            config["database"] = "hte_data"

    # overwrite host if given
    config["host"] = host if host is not None else config["host"]
    """

    # return sqlalchemy.engine.Base.Engine
    return sql.create_engine(
        "mysql+mysqlconnector://%s:%s@%s/%s"
        % (config["user"], config["password"], config["host"], config["database"]),
        echo=echo,
    )


def get_db_config(user="hte_read",
                  database=None,
                  host=None):
    """
    Get credentials form database configuration file
    When working in Developer Jupyterhub:
        database specified in "dev_$user$/db_session/current_db.txt" will be used, adjust the db using dev_settings
    :param user: str, optional default 'hte_read'
        name of the MySQL user to connect with to MySQL database
    :param database: str or None, default None
        name of the database, forces connection to a different database while keeping other credentials the same
        the user might not have sufficient privileges to connect to that database
    :param host: str or None, default None
        host IP-Adress or localhost, forces connection to a different host while keeping other credentials the same
        if None, host from config file will be used
    :return: pd.Series with database connection info
    """
    # style of config file:
    # header line
    # credentials tabulator-separated
    """
    user    password    host    database  
    your_username   your_password    localhost_or_IPaddress   your_database_name
    """

    if os.name == "nt":  # for windows development computer
        config_file_path = Path(r"db_config.ini")  
    else:  # for linux server
        config_file_path = Path("db_config.ini")  

    # handling dev_jupyterhub - overwrite config_file_path if connect to database copy
    if os.getcwd().startswith(str(DIR_JUPYTER_DEV)):
        chosen_database = 'hte_data'  # default if current db not found

        # read from db session current db file
        from evaluation.export.mysqldev import REL_DIR_DB_SESSION, \
            FILE_DB_SESSION_CURRENT_DB, \
            FILE_ENDING_DB_SESSION_CREDS
        DIR_DB_SESSION_FOLDER = DIR_JUPYTER_DEV / ('dev_' + current_user()) / REL_DIR_DB_SESSION
        DIR_DB_SESSION_CURRENT_DB = DIR_DB_SESSION_FOLDER / FILE_DB_SESSION_CURRENT_DB
        if DIR_DB_SESSION_CURRENT_DB.is_file():
            with open(DIR_DB_SESSION_CURRENT_DB, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    raise ConnectionRefusedError(
                        'Database choose file should only contain one line: ' + str(DIR_DB_SESSION_CURRENT_DB))
                chosen_database = lines[0].strip('\n')

        if chosen_database == 'hte_data' \
                or database == 'hte_data_documentation':
            # if hte_data selected take original configuration file
            # also requests for checking tracked_export table always to hte_data_documentation
            config_file_path = config_file_path  # no changes

            # prevent changes in production database from developer jupyterhub
            # only exception hte_exporter
            if user not in ['hte_read', 'hte_exporter']:
                user = 'hte_read'
        else:
            DIR_DB_SESSION_CREDS = DIR_DB_SESSION_FOLDER / (chosen_database + FILE_ENDING_DB_SESSION_CREDS)
            if DIR_DB_SESSION_CREDS.is_file():
                print('Connect to your database copy '
                      + chosen_database +
                      ' with your database user. Be aware, you have all privileges.')
                config_file_path = DIR_DB_SESSION_CREDS
                user = chosen_database + '_user'
            else:
                raise Exception("No credential file found for "+chosen_database+" in the directory.")

        """ In this version, always connect with high-privileged user: (Not tested)
        # if the current working directory is inside the jupyter_dev folder,
        # only hte_read will be used or the most current dev database will be used
        if os.getcwd().split("/")[4] == "jupyter_dev":
            # set default user to hte_read to avoid any changes to production database
            user = 'hte_read'

            # check if dev database exist and use it
            credential_file_dir = '/'.join(os.getcwd().split('/')[:6]) + '/db_session'
            if os.path.isdir(credential_file_dir):
                credential_files = os.listdir(credential_file_dir)
                modified_time = [os.path.getctime(os.path.join(credential_file_dir, filename)) for filename in
                                 credential_files]
                if credential_files:
                    latest_file_index = modified_time.index(max(modified_time))  # Index of the most recent file
                    latest_credential_file = os.path.join(credential_file_dir, credential_files[latest_file_index])
                    print(f"Taking the most recent credential file: {credential_files[latest_file_index]}")
                    print("Fetching database name...")
                    credentials = pd.read_csv(latest_credential_file, sep='\t')
                    user = credentials['user'][0].split(',')[0]

                    config_file_path = latest_credential_file
            """

    db_config = pd.read_csv(config_file_path, sep="\t", header=0)

    # read credentials of user from config file
    if user not in db_config.user.tolist():
        warnings.warn('Login details for requested database user: '
                      + user + ' not stored in database config file. '
                      'Database connection not established. Please report to admin')
    config = db_config.loc[db_config.loc[:, "user"] == user, :].transpose().iloc[:, 0]

    # overwrite config parameter if given
    config["database"] = database if database is not None else config["database"]
    config["host"] = host if host is not None else config["host"]

    return config


def verify_sql(query, params=None, method="pandas", debug=False):
    """
    Verify suitability of given SQL query with sqlite syntax. Some common syntax differences in syntax are translated in
    evaluation.utils.mysql_to_sqlite. If the query still fails to run an error is thrown. In this case, execution of the
    SQL query in SQLite database won't be possible, which should be avoided when intended to upload the code with a
    publication.
    :param query: str
        query in mysql syntax
    :param params: list of Any (any supported types) or None
        list of the parameters marked in query with '%s'
    :param method: one of ['pandas', 'sqlalchemy']
        choose with which module to run the query: sqlalchemy.connection.execute() or pandas.read_sql()
    :param debug: bool, optional, Default False
        print additional debug info
    :return: str
        query in mysql syntax
    """
    from evaluation.export.publication import PublicationExport
    pub = PublicationExport()
    if pub.created:
        # Check compatibility with sqlite by using the empty sqlite database
        sql_query_sqlite = db_config_binder.verify_sql(query, debug=debug)
        try:
            with pub.connect_sqlite(path_to_sqlite=pub.path_to_test_sqlite).begin() as con_sqlite:
                savepoint = con_sqlite.begin_nested()
                db.query_sql_execute_method(
                    sql_query_sqlite, params=params, con=con_sqlite, method=method
                )
                savepoint.rollback()

        except Exception as error:
            print(
                "\x1b[31m"
                + "Translating query to SQlite fails (only problematic when exporting as publication). "
                "Please report to admin." + "\x1b[0m"
            )
            print(sql_query_sqlite, "\n", error)
    return query


def user_is_owner(index_col, index_value):
    """
    Check whether user is owner of a database entry specified index column name and value. Used to verify whether
    data processing is allowed.
    :param index_col: str
        name of the index column
    :param index_value: int
        value of the index
    :return: bool
    """
    name_table = {"id_exp_sfc": "exp_sfc", "id_exp_icpms": "exp_icpms"}[index_col]
    con = connect()
    is_owner = (
        con.execute(
            """ SELECT name_user 
                                FROM """
            + name_table
            + """ 
                                WHERE """
            + index_col
            + """ = %s""",
            [index_value],
        ).fetchall()[0][0]
        == current_user()
    )
    # print('is owner? ', is_owner)
    con.dispose()
    return is_owner


def current_user():
    """
    get the current user name
    :return: str, current user name
    """

    # return 'HTE_team'

    # production jupyterhub
    if os.getcwd().startswith(str(DIR_JUPYTER)+'/'):
        username = os.getcwd().replace(str(DIR_JUPYTER)+"/", "").split("/")[0]

    # for dev_jupyter
    elif os.getcwd().startswith(str(DIR_JUPYTER_DEV)+'/'):
        username = os.getcwd().replace(str(DIR_JUPYTER_DEV)+"/dev_", "").split("/")[0]
    else:
        raise ConnectionRefusedError(
            "Wrong current working directory. Please inform Admin."
        )

    if username == "shared":
        raise ConnectionRefusedError(
            "Do not run this script in shared but in your personal folder."
        )

    if "JUPYTERHUB_USER" in [name for name, value in os.environ.items()]:
        if username != os.environ["JUPYTERHUB_USER"]:
            raise ConnectionRefusedError("Do not change your username.")

    return username


def call_procedure(engine, name, params=None):
    """
    A function to run stored procedure, this will work for mysql-connector-python but may vary with other DBAPIs!
    :param engine: sqlalchemy.engine
        database connection
    :param name: str
        name of the stored procedure
    :param params: list of Any (any supported types)
        list of the parameters marked in query with '%s'
    :return: list of all result sets as pd.DataFrame
    """

    # If this get stuck make sure, tables are unlocked and you have the privilege to execute the stored procedure
    if params is None:
        params = []
    if engine is None:
        raise Exception(
            "Call_procedure must be called with an engine from sqlalchemy and cannot run with out"
        )

    try:
        connection = engine.raw_connection()
        cursor = connection.cursor()
        cursor.callproc(name, params)
        results = []
        for (
            result
        ) in cursor.stored_results():  # multiple resultssets should be possible
            results = results + [
                pd.DataFrame(
                    result.fetchall(), columns=[i[0] for i in result.description]
                )
            ]
        return results

    except Error as e:
        raise Exception(e)
    finally:
        cursor.close()
        connection.commit()
        connection.close()
        #


def sql_update(df_update,
               table_name,
               engine=None,
               con=None,
               add_cond=None,
               table_schema=None):
    """
    Update sqlite database by values given in DataFrame df_update. If error occurs, transaction is rolled back.
    :param df_update: pd.DataFrame
        DataFrame with rows and columns which should be updated in the database.
        Not required to give all columns of the database table just the one meant to be updated.
    :param table_name: str
        Name of the table in sqlite database
    :param engine: sql.engine, optional
        Sqlalchemy engine to perform the update
    :param con: sql.connection, optional
        Sqlalchemy connection to perform the update, instead of engine
    :param add_cond: str
        additional condition to subselect rows in the table meant to be updated
    :return: None
    """
    con_init = con
    if con is None:  # cursor is None and
        if engine is None:
            engine = connect("hte_write")
        connection = engine.raw_connection()
        con = connection.cursor()
    if table_schema is None:
        if engine is None:
            table_schema = con.execute('SELECT DATABASE();').fetchall()[0][0]
        else:
            table_schema = engine.url.database
    # name_user would need to be checked to ensure user_safe update
    # index_col = [df_update.index.name] if df_update.index.name is not None else df_update.index.names
    # print(db.current_user(), index_col)

    # tablename: [updatable colums]
    db_constraints = db.db_constraints_update()
    if (
        table_name not in db_constraints.keys()
    ):  # ['exp_icpms_sfc', 'exp_icpms_integration', 'exp_ec_integration', 'ana_integrations', 'exp_sfc']:
        raise Exception("Udating " + table_name + " not implemented yet")

    for index, row in df_update.iterrows():
        # print(len(row.index))
        sql_query = "UPDATE  "+table_schema+".`" + table_name + "` SET "
        vals = []
        for iteration, (col, val) in enumerate(row.to_dict().items()):
            if db_constraints[table_name] is not None:
                if col not in db_constraints[table_name]:
                    raise ConnectionRefusedError(
                        "Update column " + col + " is not allowed."
                    )
            sql_query += (
                "`"
                + col
                + "` = %s"
                + (" " if iteration == len(row.index) - 1 else ", ")
            )
            vals += [val]
        if type(index) == tuple:
            sql_query += (
                " WHERE (("
                + ", ".join(df_update.index.names)
                + ") = ("
                + ("%s, " * len(index))[:-2]
                + "))"
            )
            vals += list(index)
        else:
            sql_query += " WHERE (" + df_update.index.name + " = %s)"
            vals += [index]
        sql_query += ";" if add_cond is None else "AND " + add_cond + ";"

        # print(sql, vals)
        print(
            " ".join(
                [query + str(vals) for query, vals in zip(sql_query.split("%s"), vals + [""])]
            )
        )
        con.execute(sql_query, vals)
    if con_init is None:  # cursor is None
        connection.commit()


def get_data_raw(name_table, col_names, col_values, add_cond):
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
    col_names_str = "(" + str(list(col_names))[1:-1].replace("'", "`") + ")"
    col_values_str = "(" + str(list(col_values))[1:-1] + ")"

    sql_query = (
        "SELECT * FROM "
        + name_table
        + " WHERE "
        + col_names_str  # str(tuple(self._obj.index.names)).replace('\'', '`')
        + " IN "
        + col_values_str
        + (" AND " + str(add_cond) if add_cond is not None else "")
        + ";"
    )
    print(sql_query)
    # data = pd.DataFrame(conn.execute(sql.text(sql_query)).fetchall())
    with connect().begin() as con:
        data = pd.read_sql(sql_query, con=con)  # similar performance
    return data


def get_primarykeys(name_table=None,
                    table_schema=None,
                    engine=None,
                    ):
    """
    Get primary keys of all tables (name_table is None) or of specific table (name_table = str) in MysQL database.
    :param name_table: str or None, optional, Default None
        get primary keys of all (None) or specific table
    :param table_schema: str, default=database specified in engine, usually hte_data
        name of the database schema of the table
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: primary_keys_grouped
    """
    # print('Get primary keys of database tables.')
    if engine is None:
        engine = connect()
    if table_schema is None:
        table_schema = engine.url.database
    with engine.begin() as con:
        # PRIMARY KEYS
        primary_keys = pd.read_sql(
            """SELECT * 
               FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
               WHERE TABLE_SCHEMA = %s
                    AND CONSTRAINT_NAME = 'PRIMARY'
            """
            + (" AND TABLE_NAME = %s" if name_table is not None else ""),
            params=[table_schema] + ([name_table] if name_table is not None else []),
            con=con,
        )

        if name_table is not None:
            if name_table not in primary_keys.TABLE_NAME.tolist():
                warnings.warn("No primary key found for table " + str(name_table))
                return None
            return primary_keys.COLUMN_NAME.tolist()
        else:
            primary_keys_grouped = (
                primary_keys.loc[
                    :,
                    [
                        "CONSTRAINT_NAME",
                        "TABLE_NAME",
                        "COLUMN_NAME",
                    ],
                ]
                .groupby(
                    [
                        "CONSTRAINT_NAME",
                        "TABLE_NAME",
                    ]
                )
                .apply(
                    lambda group: group.apply(
                        lambda col: pd.Series({col.index[0]: list(col.tolist())})
                    )
                )
                .loc[
                    :,
                    [
                        "COLUMN_NAME",
                    ],
                ]
                .reset_index()
                .set_index("TABLE_NAME")
                .loc[:, "COLUMN_NAME"]
            )
            # display(primary_keys_grouped)
            return primary_keys_grouped


def get_foreignkey_links(
        table_schema=None,
        referenced_table_schema=None,
        engine=None
):
    """
    Get Foreign keys in MySQL database.
    :param table_schema: str, default=database specified in engine, usually hte_data
        name of the database schema of the table
    :param referenced_table_schema: str, default=database specified in engine, usually hte_data
        name of the database schema of the referenced table
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: foreign_key table as pd.DataFrame
    """
    # print('Get links between database tables.')
    if engine is None:
        engine = connect()
    if table_schema is None:
        table_schema = engine.url.database
    if referenced_table_schema is None:
        referenced_table_schema = engine.url.database
    with engine.begin() as con:
        # FOREIGN KEYS
        foreign_keys = pd.read_sql(
            """SELECT * 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = '%s'
                    AND REFERENCED_TABLE_SCHEMA = '%s'
            """
            % (table_schema, referenced_table_schema),
            con=con,
        )
        foreign_keys_grouped = (
            foreign_keys.loc[
                :,
                [
                    "CONSTRAINT_NAME",
                    "TABLE_NAME",
                    "REFERENCED_TABLE_NAME",
                    "COLUMN_NAME",
                    "REFERENCED_COLUMN_NAME",
                ],
            ]
            .groupby(["CONSTRAINT_NAME", "TABLE_NAME", "REFERENCED_TABLE_NAME"])
            .apply(
                lambda group: group.apply(
                    lambda col: pd.Series({col.index[0]: list(col.tolist())})
                )
            )
            .loc[:, ["COLUMN_NAME", "REFERENCED_COLUMN_NAME"]]
        )
        foreign_keys_grouped = foreign_keys_grouped.reset_index()
        recursive_keys_grouped = foreign_keys_grouped.loc[
            foreign_keys_grouped.TABLE_NAME
            == foreign_keys_grouped.REFERENCED_TABLE_NAME
        ]
        # display(foreign_keys_grouped)
        # display(recursive_keys_grouped)
        return foreign_keys_grouped, recursive_keys_grouped


def get_views(table_schema=None,
              debug=False,
              engine=None,
):
    """
    Get a list of all views in MySQL database
    :param table_schema: table_schema: str, default=database specified in engine, usually hte_data
        name of the database schema of the table
    :param debug: bool
        print additional debug info
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: list of all views in MySQL database
    """
    # print('Get database views.')
    if engine is None:
        engine = connect()
    if table_schema is None:
        table_schema = engine.url.database
    with engine.begin() as con:
        # VIEWS
        view_tables = pd.read_sql(
            """ SELECT TABLE_NAME, TABLE_TYPE
                                    FROM information_schema.tables
                                    WHERE TABLE_SCHEMA = %s
                                     AND TABLE_TYPE IN (%s)
                                    ;""",  # , 'VIEW'
            params=[table_schema, "VIEW"],
            con=con,
        ).TABLE_NAME
        view_tables_list = view_tables.tolist()
    print(view_tables_list) if debug else ''
    return view_tables_list


def get_create_view(
        name_view,
        debug=False,
        engine=None,
):
    """
    get Create View statement from MySQL database
    :param name_view: name of the view
    :param debug: bool
        print additional debug info
    :param engine: sqlalchemy.engine.base.Engine
        engine of database connection, to specify the database to which connect to
    :return: create view statement
    """
    sql_query = "SHOW CREATE VIEW %s;" % name_view
    print(sql_query) if debug else ""

    if engine is None:
        engine = connect()#(user="hte_processor")
    with engine.begin() as con:
        return pd.read_sql(sql_query, con=con).loc[:, "Create View"].loc[0]
