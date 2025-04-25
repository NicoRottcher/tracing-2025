"""
Scripts for preparing a tracked export of data and files for publication or development
Created in 2025
@author: Nico Röttcher
"""

import os
import warnings
from pathlib import Path
import shutil  # copy files

import pandas as pd
import numpy as np
import sqlalchemy as sql  # Handling python - sql communication

# import evaluation
# from evaluation.utils import mysql_to_sqlite  # , db
from evaluation.utils import db, user_input, tools


# from evaluation.export import db_config_binder
# from evaluation.utils import user_input  # import user_input

disable_tracking = False


class Export:
    """
    class storing all information of an ExportObject
    Exports in principle from all database possible.
    """
    def __init__(self,
                 debug=False,
                 init_mysql_hte_exporter=True,
                 from_database='hte_data',
                 from_database_documentation='hte_data_documentation',
                 ):
        """
        Initializes Export object
        :param debug: print extra info for all TrackedExport functions
        """
        self.debug = debug
        self.from_database = from_database
        self.from_database_documentation = from_database_documentation

        if init_mysql_hte_exporter:
            self.engine_from_database = self.connect_engine_from_database()
            self.engine_from_database_documentation = db.connect(user="hte_exporter",
                                                                 database=self.from_database_documentation)

        # database structure - will be filled later
        self.foreign_keys_grouped = None
        self.recursive_keys_grouped = None
        self.primary_keys_grouped = None

        # dictionary of dataframes holding copy of all entries in database tables to copy to sqlite
        self.DATABASE_DF = {}
        self.DATABASE_documentation_DF = {}

        # name_user
        self.name_user = db.current_user()

    def connect_engine_from_database(self):
        """
        Connect to production database from where the data will be exported
        :return: sqlalchemy.engine.base.Engine
        """
        return db.connect(user="hte_exporter",
                          database=self.from_database)

    def _transfer_data_mysql2df(
            self,
            df_transfer_experiments=None,
            link_children_table=None,
            tables_transfer_all_values=None,
    ):
        """
        :param tables_transfer_all_values: list of name of tables from which all data should be appended
            - but no links to these tables will be considered
        :param link_children_table: None or dict.
            Restriction to only certain tables which is required to avoid circular dependencies
            and to avoid unexpected addition of data.
            (For example should all experiments on all spots of a sample be added
                or only all experiments on the spots selected?)

            None: default settings are used (see self._transfer_table_mysql2df)
            dict: keys: from which to link corresponding children tables
                  values: None or dict of style {'whitelist': ['name_table', ...], 'blacklist' ['name_table', ...]}
                    None: all children tables will be added
                    dict: only tables in the whitelist or blacklist will be added.
        :return: None
        """
        if tables_transfer_all_values is None:
            tables_transfer_all_values = []

        # get database structure
        (
            self.foreign_keys_grouped,
            self.recursive_keys_grouped,
        ) = db.get_foreignkey_links()
        self.primary_keys_grouped = db.get_primarykeys()
        # self.view_tables_sorted = db.get_views_sorted() # not required?

        # reinitialize DATABASE_DF
        self.DATABASE_DF = {}

        transfer_experiments = False
        if df_transfer_experiments is not None:
            if len(df_transfer_experiments.index) != 0:
                transfer_experiments = True

        if not transfer_experiments:
            print("\x1b[33m",
                  "No experiments selected to transfer from production db (MYSQL) to DATABASE_DF (pd.DataFrame)",
                  "\x1b[0m")
        else:
            print("\x1b[34m",
                  "Transfer experiment data from production db (MYSQL) to DATABASE_DF (pd.DataFrame)",
                  "\x1b[0m")

            # transfer mysql --> self.DATABASE_DF
            for name_table in df_transfer_experiments.name_table.unique().tolist():
                unstacked = (
                    df_transfer_experiments.loc[
                        df_transfer_experiments.name_table == name_table
                        ]
                        .set_index(["name_table", "count_exp", "name_index_col"])
                        .unstack()
                )
                unstacked.columns = unstacked.columns.get_level_values(
                    level="name_index_col"
                )
                print(f"Transfer experiments from {self.from_database}.{name_table} to DATABASE_DF")
                #print('debug', unstacked.values)
                self._transfer_table_mysql2df(
                    table_name=name_table,
                    index_name=unstacked.columns.tolist(),
                    index_values=unstacked.values,
                    link_children_table=link_children_table,
                )

        # add complete data of specified tables
        print("\x1b[34m",
              "Transfer complete data for selected tables from production db (MYSQL) to DATABASE_DF (pd.DataFrame)",
              "\x1b[0m")
        index_cols = {'documentation_tables': ['name_table'],
                      'documentation_columns': ['name_table', 'name_column'],
                      }
        for name_table in tables_transfer_all_values:
            self._transfer_table_specified_mysql2df(
                name_table=name_table,
                index_col=index_cols[name_table] if name_table in index_cols.keys() else None
            )

        # add publication data - 20241212 not considered anymore as moved to hte_data_documentation.tracked_exports
        # for name_table in ['publications', 'publication_exps']:
        #    self._transfer_table_specified_mysql2df(name_table=name_table,
        #                                             add_cond='id_tracked_export = %s',
        #                                             add_cond_params=[self.id_tracked_export])

    def _transfer_table_mysql2df(
            self,
            table_name,
            index_name,
            index_values,
            caller_table_name="",
            recursive_level="",
            link_children_table=None,
    ):
        """
        Core function to transfer all linked data of linked experiments of a publication from the corresponding
         MySQL table (table_name) to Publication.DATABASE_DF
        From the given table (name_table) all rows defined by (index_name, index_values) are added.
        If the given table has a foreign key to another (parent) table all required data is added recursively
        If another (children) table has a foreign key on the given table all required data is added if the given table
            is in link_children_table.

        :param table_name: str
            name of the given table
        :param index_name: list of str
            list of name of the index columns
        :param index_values: nested list or nested np.array()
            values for the index columns which should be selected from the database table
            either [1,2,3] for single index or [['a', 1, 2], ['b', 2,2]] for multiindex
        :param caller_table_name: str, optional, Default ''
            name of the caller table (parent or child of the given table), required to avoid circular dependencies
        :param recursive_level: str, optional, Default ''
            level of recursiveness, to reconstruct how the experiments
        :param link_children_table: None or dict.
            Restriction to only certain tables which is required to avoid circular dependencies
            and to avoid unexpected addition of data.
            (For example should all experiments on all spots of a sample be added
                or only all experiments on the spots selected?)

            None: default settings are used (see self._transfer_table_mysql2df)
            dict: keys: from which to link corresponding children tables
                  values: None or dict of style {'whitelist': ['name_table', ...], 'blacklist' ['name_table', ...]}
                    None: all children tables will be added
                    dict: only tables in the whitelist or blacklist will be added.
        :return: None
        """
        if link_children_table is None:
            link_children_table = {
                "exp_ec": None,  # data_ec, exp_ec_cv,peis,... , gamry tables, ...
                "exp_sfc": None,  # flow_cell_assemblies
                "exp_ec_polcurve": None,  # data_ec_polcurve
                "flow_cell_assemblies": None,
                "spots": None,
                # for spots_composition (and possibly other experiments pointing to that spot)
                "samples": {'whitelist': ['samples_composition']},
                # for samples_composition (and possibly other experiments pointing to that sample)
                # problem spots is children to samples, meaning all spots on sample will be added
                # further: all all sfc experiments of selected spots will be added
                # --> define link_children_table as dict with key table name, val: allowed children
                "exp_icpms": None,  # for exp_icpms_analyte_internalstandard, ...
                "exp_icpms_analyte_internalstandard": None,
                # for data_icpms, exp_icpms_calibration_params, exp_icpms_integration
                "exp_icpms_calibration_set": None,  # for calibration experiments
                "data_icpms": None,  # for data_icpms_internalstandard_fitting
                "ana_icpms_sfc_fitting": None,  # for ana_icpms_sfc_fitting_peaks
            }
        print(recursive_level, table_name) if self.debug else ""

        def query_sql_data(table_name, index_name, index_values):
            index_name_str = ", ".join(index_name)
            index_values_str, index_values_params = values2sqlstring(
                index_values
            )  # str(list(index_values))[1:-1]#', '.join(index_values)#
            # print(index_values)
            # print(index_values_str, index_values_params)# == '', len(index_values_str) > 0)

            print(index_values_str, index_values_params) if self.debug else ""

            with self.engine_from_database.begin() as con_mysql:
                table_data = pd.read_sql(
                    """SELECT * 
                                            FROM %s 
                                            WHERE (%s) IN """
                    % (table_name, index_name_str)
                    + "("
                    + index_values_str
                    + ")"
                    + ";"
                    if True
                    else """;""",
                    params=index_values_params,
                    con=con_mysql,
                    index_col=index_name,
                )
            return table_data

        len_index_values = len(index_values)
        n_split = int(1e6)

        if len_index_values > n_split:
            print('Separate long query in multiple subqueries:')
            # arr = np.array([[1,1,1,1]])
            table_data_list = []
            for i in range(int(tools.round_up(len_index_values / n_split))):
                print('    Query rows: ', i*n_split, '-', (i+1)*n_split)
                index_values_part = index_values[i * n_split: (i + 1) * n_split]
                table_data_list += [query_sql_data(table_name,
                                                   index_name,
                                                   index_values=index_values_part)]
                # index_values_str, index_values_params = values2sqlstring(index_values_part)
                # print(len(index_values_params))
                # arr = np.append(arr, index_values_part, axis=0)
                # (arr[1:] == index_values).all()
            table_data = pd.concat(table_data_list)
        else:
            table_data = query_sql_data(table_name, index_name, index_values)


        # global df_sqlite_tables
        if len(table_data.index) > 0:
            table_data_primary_key = table_data.reset_index().set_index(
                self.primary_keys_grouped.loc[table_name]
            )
            if table_name not in self.DATABASE_DF.keys():
                new_data_added = True
                self.DATABASE_DF[table_name] = table_data_primary_key
            else:
                new_data_added = not table_data_primary_key.index.isin(
                    self.DATABASE_DF[table_name].index
                ).all()
                if new_data_added:
                    display(self.DATABASE_DF[table_name]) if self.debug else ""
                    display(table_data) if self.debug else ""
                    self.DATABASE_DF[table_name] = (
                        pd.concat(
                            [self.DATABASE_DF[table_name], table_data_primary_key]
                        )
                            .reset_index()
                            .drop_duplicates()
                            .set_index(self.primary_keys_grouped.loc[table_name])
                    )
                else:
                    print("Nothing new for %s" % table_name) if self.debug else None
        else:
            new_data_added = False

        if (
                new_data_added
        ):  # len(index_values_params) > 0 and not all([val == 'None' for val in index_values_params]):
            # if recursive:
            if (
                    table_name in self.foreign_keys_grouped.TABLE_NAME.tolist()
            ):  # index.get_level_values(level='TABLE_NAME')
                for index, row in self.foreign_keys_grouped.loc[
                                  (
                                          (self.foreign_keys_grouped.TABLE_NAME == table_name)
                                          & (
                                                  self.foreign_keys_grouped.REFERENCED_TABLE_NAME
                                                  != caller_table_name
                                          )
                                  ),
                                  :,
                                  ].iterrows():
                    # if row.COLUMN_NAME == index_name:
                    #    print('index', index_name, table_name, row.REFERENCED_COLUMN_NAME, row.REFERENCED_TABLE_NAME)
                    #    continue
                    # print(index_name, row.TABLE_NAME, row.COLUMN_NAME, ' child of ',
                    #       row.REFERENCED_TABLE_NAME, row.REFERENCED_COLUMN_NAME)

                    # display(table_data)
                    referenced_index_value = (
                        table_data.reset_index()
                            .loc[:, row.COLUMN_NAME]
                            .drop_duplicates()
                            .values
                    )
                    # referenced_index_value_str =
                    # print(referenced_index_value)
                    self._transfer_table_mysql2df(
                        table_name=row.REFERENCED_TABLE_NAME,
                        index_name=row.REFERENCED_COLUMN_NAME,
                        index_values=referenced_index_value,
                        caller_table_name=table_name,
                        recursive_level="---" + recursive_level,
                        link_children_table=link_children_table,
                    )

            for index, row in self.foreign_keys_grouped.loc[
                              (
                                      (self.foreign_keys_grouped.REFERENCED_TABLE_NAME == table_name)
                                      # & (self.foreign_keys_grouped.TABLE_NAME != caller_table_name)
                                      # must not be set: would avoid
                                      # exp_sfc <<< spots --- exp_sfc (addition of experiments previously performed at that spot)
                                      # exp_icpms <<< exp_icpms_calibration_set --- exp_icpms (addition of calibration exps)
                                      & (
                                              self.foreign_keys_grouped.REFERENCED_TABLE_NAME.isin(
                                                  link_children_table.keys()
                                              )
                                      )
                              ),
                              :,
                              ].iterrows():
                if row.REFERENCED_TABLE_NAME in link_children_table.keys():  # should always be fulfilled
                    if link_children_table[row.REFERENCED_TABLE_NAME] is not None:
                        if not isinstance(link_children_table[row.REFERENCED_TABLE_NAME], dict):
                            print(link_children_table[row.REFERENCED_TABLE_NAME])
                            raise ValueError('link_children_table must be None or dict')
                        if 'whitelist' in link_children_table[row.REFERENCED_TABLE_NAME].keys():
                            if row.TABLE_NAME not in link_children_table[row.REFERENCED_TABLE_NAME]['whitelist']:
                                print(f'link_children_table prevented: {row.REFERENCED_TABLE_NAME}<<<{row.TABLE_NAME}.'
                                      f'{row.TABLE_NAME} not in whitelist')
                                continue
                        if 'blacklist' in link_children_table[row.REFERENCED_TABLE_NAME].keys():
                            if row.TABLE_NAME in link_children_table[row.REFERENCED_TABLE_NAME]['blacklist']:
                                print(f'link_children_table prevented: {row.REFERENCED_TABLE_NAME}<<<{row.TABLE_NAME}.'
                                      f'{row.TABLE_NAME} in blacklist')
                                continue

                # if row.COLUMN_NAME == index_name:
                #    print('index', index_name, table_name, row.REFERENCED_COLUMN_NAME, row.REFERENCED_TABLE_NAME)
                #    continue
                # print(index_name,  row.REFERENCED_TABLE_NAME, row.REFERENCED_COLUMN_NAME,
                #       ' parent of ', row.TABLE_NAME, row.COLUMN_NAME)
                parent_index_value = (
                    table_data.reset_index()
                        .loc[:, row.COLUMN_NAME]
                        .drop_duplicates()
                        .values
                )
                if row.TABLE_NAME == caller_table_name and self.debug:
                    # TODO: Check whether also not selected spots from the same sample are selected
                    print(
                        caller_table_name,
                        "references back and forth",
                        row.TABLE_NAME,
                        "; This might lead to an endless loop",
                    )
                    print(
                        dict(
                            table_name=row.TABLE_NAME,
                            index_name=row.COLUMN_NAME,
                            index_values=parent_index_value,  # referenced_index_value,
                            caller_table_name=table_name,
                            recursive_level="<<<" + recursive_level,
                            link_children_table=link_children_table,
                        )
                    ) if self.debug else ""
                self._transfer_table_mysql2df(
                    table_name=row.TABLE_NAME,
                    index_name=row.COLUMN_NAME,
                    index_values=parent_index_value,  # referenced_index_value,
                    caller_table_name=table_name,
                    recursive_level="<<<" + recursive_level,
                    link_children_table=link_children_table,
                )
            # else:
            #    print('No recursive search for ', table_name)

    def _transfer_table_specified_mysql2df(self,
                                           name_table,
                                           index_col=None,
                                           add_cond=None,
                                           add_cond_params=None):
        """
        Transfers data from MySQL to self.DATABASE_DF for the given table, for entries fulfilling add_cond.
        Partially added tables are not supported.
        :param name_table: str
            table names from which entries fulfilling add_cond should be transferred to DATABASE_DF
        :param index_col: str or list of str or None
            index columns in the table. Does not have to provided if table has a primary key.
        :param add_cond: str
            additional SQL condition
        :param add_cond_params: str
            parameter for the additional SQL condition
        :return: None
        """
        if index_col is None:
            index_col = []
        if add_cond_params is None:
            add_cond_params = []
        print(name_table, end='')
        with self.engine_from_database.begin() as con_mysql:
            print(name_table) if self.debug else ""
            if add_cond is not None:
                table_data = db.query_sql(
                    """SELECT * 
                       FROM """ + name_table + """ 
                       WHERE """ + add_cond + """; """,
                    params=add_cond_params,
                    con=con_mysql,
                )
            else:
                table_data = db.query_sql(
                    """SELECT * 
                       FROM %s ; """
                    % name_table,
                    con=con_mysql,
                )
            if index_col:
                table_data = table_data.set_index(index_col)
            elif name_table in self.primary_keys_grouped.index:
                table_data = table_data.set_index(
                    self.primary_keys_grouped.loc[name_table]
                )
            else:
                warnings.warn('No index column provided. '
                              'This might lead to an error during insertion into sqlite database')

            if name_table not in self.DATABASE_DF.keys():
                self.DATABASE_DF[name_table] = table_data
            else:
                raise Exception(
                    "Table already partially added. Adding of all values for these tables not "
                    + "supported "
                )
            print('--> ', len(table_data.index), 'entries transferred')

    def _transfer_documentation_mysql2df(
            self,
            except_table_name=None
    ):
        """
        Transfer documentation data completely (no selection of rows) to self.DATABASE_documentation_DF (pd.DataFrame)
        :return: None
        """
        if except_table_name is None:
            except_table_name = ['tracked_exports', 'tracked_exports_exps']
        print("\x1b[34m",
              "Transfer documentation data from doc db (MYSQL) to DATABASE_documentation_DF (pd.DataFrame)",
              "\x1b[0m")
        # reinitialize DATABASE_documentation_DF
        self.DATABASE_documentation_DF = {}
        primary_keys_documentation_grouped = db.get_primarykeys(table_schema="hte_data_documentation")

        for name_table, index_cols in primary_keys_documentation_grouped.iteritems():
            if name_table not in except_table_name:
                print(f'Transfer documentation data of {self.from_database_documentation}.{name_table}')
                table_data = pd.read_sql(name_table,
                                         con=self.engine_from_database_documentation,
                                         index_col=index_cols)

                if name_table not in self.DATABASE_documentation_DF.keys():
                    self.DATABASE_documentation_DF[name_table] = table_data
                else:
                    raise Exception(
                        "Table already (partially) added. Adding of all values for these tables not supported."
                    )


class TrackedExport(Export):
    """
    subclass of Export
    stores all information of the TrackedExport
    Exports only from main production database possible.
        Database connections will always be made to hte_data_documentation.tracked_exports
    """

    def __init__(self, debug=False, init_mysql_hte_exporter=True):
        """
        Initializes TrackedExport object, checks whether any TrackedExport is already created within
        the current working directory
        :param debug: print extra info for all TrackedExport functions
        """
        if disable_tracking:
            self.created = False
            return None


        super().__init__(debug=debug,
                         init_mysql_hte_exporter=init_mysql_hte_exporter,
                         from_database='hte_data',
                         from_database_documentation='hte_data_documentation',
                         )

        self.cwd = Path(os.getcwd())  # current working directory

        # check if TrackedExport is initialized in folder
        # only read rights are required
        temp_engine = db.connect(user="hte_read",
                                 database=self.from_database_documentation
                                 )
        with temp_engine.begin() as con:
            df_tracked_exports = pd.read_sql("tracked_exports", con=con)
        temp_engine.dispose()
        tracked_exports_paths = [
            Path(path) for path in df_tracked_exports.path_to_jupyter_folder.tolist()
        ]
        # print(publication_paths)

        # Match publication by folder
        # cwd is wd or any subfolder of a publication
        df_tracked_exports_requested = df_tracked_exports.loc[
            [
                tracked_exports_path == self.cwd or tracked_exports_path in self.cwd.parents
                for tracked_exports_path in tracked_exports_paths
            ]
        ]
        # print(df_tracked_exports_requested)

        if len(df_tracked_exports_requested.index) == 0:
            self.created = False
            self.export_type = 'TrackedExport'
            # print('No publication initiated in this or parent folder. Create publication first.')
        elif len(df_tracked_exports_requested.index) == 1:
            if self.name_user != df_tracked_exports_requested.iloc[0].name_user:
                raise ConnectionRefusedError('You have not initialized that TrackedExport: '
                                             + str(self.name_user) + ' != '
                                             + str(df_tracked_exports_requested.iloc[0].name_user))

            self.created = True
            self.id_tracked_export = df_tracked_exports_requested.iloc[0].id_tracked_export
            self.export_type = df_tracked_exports_requested.iloc[0].export_type
            # self.title = df_tracked_exports_requested.iloc[0].title
            # self.name_journal = df_tracked_exports_requested.iloc[0].name_journal

            self.path_to_jupyter_folder = Path(
                df_tracked_exports_requested.iloc[0].path_to_jupyter_folder
            )

            print(
                "TrackedExport ", self.id_tracked_export, " initiated"
            ) if self.debug else None
        else:
            raise Exception(
                "Incorrect database state. Multiple publications initiated for this and parent folders:"
                + ", ".join(df_tracked_exports_requested.id_tracked_export.tolist())
            )

    def create(self, id_tracked_exports=None):
        """
        Create a TrackedExport object for the given folder. n
        :param id_tracked_exports: str or None, optional, Default None
            identifier for the publication
        :return: self
        """
        if self.created:
            print("Already created!")
            if self.id_tracked_export != id_tracked_exports:
                warnings.warn(
                    "Updating id_tracked_export not yet developed. "
                    "Still id_tracked_export = " + str(self.id_tracked_export)
                )

        else:
            self.id_tracked_export = id_tracked_exports
            """(
                user_input.user_input(
                    text="Input your id_tracked_export title:", dtype="str", optional=True
                )
            )
            """
            self.path_to_jupyter_folder = self.cwd


            with self.engine_from_database_documentation.begin() as con_mysql:
                con_mysql.execute(
                    """INSERT INTO `tracked_exports` 
                                    (`id_tracked_export`, `path_to_jupyter_folder`, `name_user`, `export_type`) 
                                    VALUES (%s, %s, %s, %s);""",
                    [
                        self.id_tracked_export,
                        str(self.path_to_jupyter_folder),
                        self.name_user,  # this have to be outsourced
                        # to another table like publication_authors to allow for collaboration
                        self.export_type,
                    ],
                )
                print("Inserted entry %s into database." % self.id_tracked_export)

        # reinitialize object to define all variables
        self.__init__(debug=self.debug)

        return self

    def get_df_tracked_exports_exp(self):
        """
        Get DataFrame of all experiments linked to the publication
        :return: DataFrame of all experiments linked to the publication
        """
        with db.connect(user="hte_read", database=self.from_database_documentation).begin() as con_mysql:
            df_tracked_exports_exp = pd.read_sql(
                "SELECT name_table, count_exp, name_index_col, value_index_col "
                "FROM tracked_exports_exps "
                "WHERE id_tracked_export=%s;",
                params=[self.id_tracked_export],
                con=con_mysql,
            )
        return df_tracked_exports_exp

    def remove_linked_experiment(self):
        """
        Removes all experiment links, after user confirmation
        :return: None
        """
        if self.name_user != db.current_user():
            print(
                "Deleting is only allowed for owner of TrackedExport: %s" % self.name_user
            )
            return False

        with self.engine_from_database_documentation.begin() as con_mysql:
            df_exp = pd.read_sql(
                "SELECT * FROM tracked_exports_exps WHERE id_tracked_export = %s",
                params=[self.id_tracked_export],
                con=con_mysql,
            )

            print(
                "Remove %s experiments linked to %s"
                % (
                    len(
                        df_exp.loc[:, ["name_table", "count_exp"]]
                            .drop_duplicates()
                            .index
                    ),
                    self.id_tracked_export,
                )
            )
            display(df_exp)

            if not user_input.user_input(
                    text="Do you want to delete the links? "
                         "(You will need to rerun all data grabbing routines "
                         "to establish the experimental link again) \n",
                    dtype="bool",
            ):
                print("Cancelled")
                return False

            con_mysql.execute(
                "DELETE FROM tracked_exports_exps WHERE id_tracked_export = %s",
                [self.id_tracked_export],
            )
            self.DATABASE_DF = {}
            print("\x1b[32m", "Successfully deleted experiment links!", "\x1b[0m")

    def display_linked_experiments(self):
        """
        Displays ids of all experiments linked to the publication
        :return: None
        """
        df_publication_exp = self.get_df_tracked_exports_exp()
        for name_table in df_publication_exp.name_table.unique().tolist():
            unstacked = (
                df_publication_exp.loc[df_publication_exp.name_table == name_table]
                    .set_index(["name_table", "count_exp", "name_index_col"])
                    .unstack()
            )
            unstacked.columns = unstacked.columns.get_level_values(
                level="name_index_col"
            )
            print(name_table)
            print(unstacked.values)


def add_tracked_exp(df_exp, name_base_table, debug=False):
    """
    Adds experiment to a TrackedExport in the database table get_df_tracked_exports_exp.
    Is performed when evaluation.utils.db.get_exp() is called within a previously initialized TrackedExport folder
    :param df_exp: pd.DataFrame
        DataFrame of the experiments to be added
    :param name_base_table: str
        name of the base table to which the experiment is referenced to
    :return:
    """
    tracked_export = TrackedExport(init_mysql_hte_exporter=False)
    if not tracked_export.created:
        #  print("No TrackedExport created to link experiments")
        return False

    print(f"Link selected experiments to {tracked_export.export_type}:  {tracked_export.id_tracked_export}")
    primary_keys = db.get_primarykeys()

    # TODO: add user restriction, only add experiment when own data

    with db.connect(user="hte_exporter", database='hte_data_documentation').begin() as con_write:
        counter_existed = 0
        counter_inserted = 0
        for index, row in df_exp.reset_index().iterrows():
            # check whether experiment already added
            sql_check_existing = (
                    """SELECT count_exp
                                    FROM tracked_exports_exps 
                                    WHERE id_tracked_export=%s 
                                        AND name_table =%s
                                    """
                    + " AND ("
                    + " OR ".join(
                ["(name_index_col = %s AND value_index_col = %s )\n"]
                * len(primary_keys.loc[name_base_table])
            )
                    + """) 
                    GROUP BY count_exp
                    HAVING COUNT(*) = %s;"""
            )
            print(sql) if debug else ""
            params = (
                    [tracked_export.id_tracked_export, name_base_table]
                    + sum(
                [
                    [name_index_col, str(row.loc[name_index_col])]
                    for name_index_col in primary_keys.loc[name_base_table]
                ],
                [],
            )
                    + [len(primary_keys.loc[name_base_table])]
            )

            print(params) if debug else ""

            count_exp_exist = pd.read_sql(
                sql_check_existing, params=params, con=con_write
            )  # .count_exp.max()
            if len(count_exp_exist) > 0:
                # print('Already added', params)
                counter_existed += 1
                continue

            # check for highest count_exp and add correspondingly
            counter_in_db = pd.read_sql(
                """SELECT MAX(count_exp)+1 AS counter 
                    FROM tracked_exports_exps 
                    WHERE id_tracked_export=%s 
                        AND name_table =%s""",
                params=[tracked_export.id_tracked_export, name_base_table],
                con=con_write,
            ).loc[0, "counter"]
            counter_in_db = 1 if counter_in_db is None else int(counter_in_db)

            # add experiment to get_df_tracked_exports_exp
            for name_index_col in primary_keys.loc[name_base_table]:
                # print('Insert experiment: ', name_base_table, counter_in_db, name_index_col, row.loc[index_col])
                con_write.execute(
                    """INSERT INTO tracked_exports_exps (`id_tracked_export`, 
                                                     `name_table`, 
                                                     `count_exp`, 
                                                     `name_index_col`, 
                                                     `value_index_col`)
                       VALUES (%s, %s, %s, %s, %s)
                    """,
                    [
                        tracked_export.id_tracked_export,
                        name_base_table,
                        counter_in_db,
                        name_index_col,
                        str(row.loc[name_index_col]),
                    ],
                )
            counter_inserted += 1
    print(
        "\x1b[32m",
        "For table",
        name_base_table,
        ": inserted new experiments =",
        counter_inserted,
        ", skipped existing experiments =",
        counter_existed,
        "\x1b[0m",
    )


def values2sqlstring(values):
    """
    used in _transfer_table_mysql2df, transform list of values to sql '%s' string and corresponding params list
    :param values: list of index values, either [1,2,3] for single index or [['a', 1, 2], ['b', 2,2]] for multiindex
    :return: sqlstring, params
    """
    # singlevalues_types = [str, int, float, ]
    multivalues_types = [list, np.ndarray]

    if len(values) == 0:
        sqlstring = ""
        params = []
    elif type(values) not in multivalues_types:
        sqlstring = "%s"
        params = [values]
    else:  # if type(values) in multivalues_types:
        # print(values)
        if (
                type(values[0]) in multivalues_types and len(values[0]) == 1
        ):  # singleindex in brakets reduce dimension
            values = [val[0] for val in values]
        # print(values,type(values[0]) not in multivalues_types)
        if type(values[0]) not in multivalues_types:  # singleindex
            # values.remove(None)
            sqlstring = ", ".join(["%s"] * len(values))
            params = [str(val) for val in values]
        else:  # multiindex
            # sqlstring = ", ".join(
            #    ["(" + ", ".join(["%s"] * len(row)) + ")" for row in values]
            # )
            sqlstring = (("(" + ", ".join(["%s"] * len(values[0])) + "), ") * len(values))[:-2]  # more efficient

            # sqlstring=''
            # params = []
            # [[params.append(str(val)) for val in row] for row in values]
            params = values.astype(str).flatten()  # more efficient

    # print(sqlstring, params)
    return sqlstring, params
