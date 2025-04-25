"""
Scripts for a export of data and files for publication
Created in 2025
@author: Nico Röttcher
"""


import os
import warnings
from pathlib import Path
import shutil  # copy files
import subprocess

import pandas as pd
import numpy as np
import sqlalchemy as sql  # Handling python - sql communication

import evaluation
from evaluation.utils import mysql_to_sqlite  # , db
import evaluation.utils.db as db
from evaluation.utils import user_input, db_config, tools  # import user_input
from evaluation.export import tracking, db_config_binder


REL_DIR_UPLOAD = Path("upload/")  # in publication folder
REL_DIR_LIBS = Path("libs/")
REL_DIR_EVALUATION = Path("libs/evaluation")
DIR_EVALUATION_MODULE = Path(evaluation.__file__).parent
# DIR_EMPTY_SQLITE_NO_VIEWS = DIR_EVALUATION_MODULE / Path("export/database/sqlite_no_views.db")
# DIR_EMPTY_SQLITE = DIR_EVALUATION_MODULE / Path("export/database/sqlite.db")
DIR_CREATE_EMPTY_SQLITE_SH = DIR_EVALUATION_MODULE / Path("export/create_sqlite_dump.sh")
FILE_EMPTY_SQLITE_TEST = "test_queries.db"


class PublicationExport(tracking.TrackedExport):
    """
    Subclass of tracking.TrackedExport()
    """
    def __init__(self,
                 debug=False,
                 init_mysql_hte_exporter=True,
                 init_sqlite_file=False,):
        super().__init__(debug=debug,
                         init_mysql_hte_exporter=init_mysql_hte_exporter,
                         )

        if not self.created:
            self.export_type = 'PublicationExport'
        elif self.export_type != 'PublicationExport':
            # avoid problems between different subclasses
            # warnings.warn('There is a TrackedExport Object created '
            #              'as ' + self.export_type +
            #              'but not as PublicationExport object. '
            #             'Overwrite self.export_type="PublicationExport".')
            self.created = False
        else:
            # self.export_type = 'PublicationExport'
            self.path_to_publication_upload = (
                    self.path_to_jupyter_folder / REL_DIR_UPLOAD
            )

            # Create a test sqlite file to test sql queries during adding of experiments
            self.path_to_test_sqlite = (
                    self.path_to_jupyter_folder / FILE_EMPTY_SQLITE_TEST
            )
            if not os.path.isfile(self.path_to_test_sqlite) or init_sqlite_file:
                self.create_empty_sqlite(self.path_to_test_sqlite)

            self.path_to_publication_sqlite = (
                    self.path_to_publication_upload / db_config_binder.REL_DIR_SQLITE
            )
            self.engine_sqlite = None  # self.connect_sqlite()
            print(
                "Publication ", self.id_tracked_export, " initiated"
            ) if self.debug else None

    def connect_sqlite(
            self,
            path_to_sqlite=None,
    ):
        """

        :param path_to_sqlite: str, pathlib.Path, or None
            None, searches for self.path_to_publication_sqlite and use it
        :return:
        """
        """
        Connect to Sqlite database
        :return: sqlite sqlalchemy engine
        """
        if path_to_sqlite is None:
            if os.path.isfile(self.path_to_publication_sqlite):
                path_to_sqlite = self.path_to_publication_sqlite
            else:
                warnings.warn('No path_to_sqlite given to connect to sqlite '
                              'and no sqlite db exported to which auto connect.')
                return None
        elif not os.path.isfile(path_to_sqlite):
            # print('Unable to find sqlite file.')
            return None
        return db_config_binder.connect(path_to_sqlite=path_to_sqlite)

    def export_to_upload(
        self,
        debug=False,
        tables_transfer_all_values=None,
        link_children_table=None,
        use_sqlite_file=False,
        zip_db=True,
    ):
        """
        Initialize database export
        :param debug: print extra info if True
        :param tables_transfer_all_values: list of name of tables from which all data should be appended
            - but no links to these tables will be considered
            default includes "dummy_data_icpms_sfc_batch_analysis", "documentation_columns", "documentation_tables"
        :param link_children_table: list of str or None
            list of name of tables from which to link corresponding children tables
            (a children table references (foreign key) on to a parent table index)
        :param use_sqlite_file: False or str
            Path to the sqlite file which should be used.
            Use this parameter if you have to create it manually via terminal.
        """
        # if exclude_views is None:
        #    exclude_views = ['data_icpms_sfc_analysis_old']
        if tables_transfer_all_values is None:
            tables_transfer_all_values = ["dummy_data_icpms_sfc_batch_analysis",
                                          "documentation_columns",
                                          "documentation_tables"]

        self.debug = debug

        if self.path_to_jupyter_folder.name == "upload":
            raise Exception("Do not run export_to_upload script in upload folder")

        # initialize upload folder
        self.create_upload_folder()

        # copy files to upload folder
        self.copy_upload_files()
        if isinstance(use_sqlite_file, str):
            self.create_empty_sqlite(use_sqlite_file)
            shutil.move(use_sqlite_file, self.path_to_publication_sqlite)
            print("\x1b[33m", f"Please remember to remove the database file which was copied "
                              f"from here {use_sqlite_file} to the upload folder.", "\x1b[0m")
        else:
            self.create_empty_sqlite(self.path_to_publication_sqlite)

        if not os.path.isfile(self.path_to_publication_sqlite):
            raise Exception('Copied database folder, but still could not find database file: %s'
                            % self.path_to_publication_sqlite)


        # check mysql connection
        try:
            self.engine_from_database.connect()
        except sql.exc.OperationalError as e:
            print("reconnect to mysql...")
            self.engine_from_database = self.connect_engine_from_database()

        # Transfer from production database to self.DATABASE_DF as pandas.DataFrame
        self._transfer_data_mysql2df(df_transfer_experiments=self.get_df_tracked_exports_exp(),
                                     link_children_table=link_children_table,
                                     tables_transfer_all_values=tables_transfer_all_values)

        # Fill sqlite database
        self.engine_sqlite = self.connect_sqlite(self.path_to_publication_sqlite)

        self._transfer_data_df2sqlite()
        # print("\x1b[34m", 'Transfer views from mysql to sqlite', "\x1b[0m")
        # self.transfer_views_mysql2sqlite(exclude_views)

        # zip sqlite database
        if zip_db:
            tools.zipfiles(zip_dst=str(self.path_to_publication_sqlite).replace('.db','.zip'),
                           src_files=[self.path_to_publication_sqlite]
                           )
            print("\x1b[32m", "A zip of your database was added but the sqlite.db is kept. "
                              "Decide which of the files (.db or .zip) to add to you repository."
                              "Zip size is probably smaller and will thus reduce loading time for binder.", "\x1b[0m")

        print("\x1b[32m", "Successfully exported publication for upload!", "\x1b[0m")

        # dispose database connections
        self.engine_from_database.dispose()  # doesn't have to be disposed
        self.engine_sqlite.dispose()

    def create_upload_folder(self):
        """
        Creates (or deletes and creates) the upload folder for the publication
        :return: None
        """
        if os.path.isdir(self.path_to_publication_upload):
            if not user_input.user_input(
                text="Upload folder already exists. Delete and continue data export?\n",
                dtype="bool",
            ):
                print("\x1b[31m", "Cancelled, no export has been performed!", "\x1b[0m")
                raise Exception('Program was cancelled by user.')

            shutil.rmtree(self.path_to_publication_upload)
            print("Deleted old upload folder")
        os.mkdir(self.path_to_publication_upload)
        print("Created upload folder ", str(self.path_to_publication_upload))

    def copy_upload_files(self,):
        """
        Copies to the upload folder:
            - all modules in evaluation, and also
                renaming db_config to db_config_mysql
                and copy db_config_binder as new db_config
            - all files in evaluation/export/repo_files
            - all files in evaluation/export/database
            - all files in publication folder
        Creates upload/reports/
        :return: None
        """
        print(
            "\x1b[34m",
            "Copy all required files to upload folder ... (This may take a while)",
            "\x1b[0m",
        )
        # all modules in folder
        shutil.copytree(
            src=DIR_EVALUATION_MODULE,
            dst=self.path_to_publication_upload / REL_DIR_EVALUATION,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "../export/db_config_binder.py", "db_config.py"
            ),
        )

        # __init__.py for scripts folder
        shutil.copy2(
            src=Path(evaluation.__file__),
            dst=self.path_to_publication_upload / REL_DIR_LIBS / "__init__.py",
        )

        # db_config_binder.py --> db_config.py
        shutil.copy(
            src=DIR_EVALUATION_MODULE / Path("export/db_config_binder.py"),
            dst=self.path_to_publication_upload
            / REL_DIR_EVALUATION
            / Path("utils/db_config.py"),
        )

        # db_config.py --> db_config_mysql.py
        copy_and_replace(
            src=DIR_EVALUATION_MODULE / Path("utils/db_config.py"),
            dst=self.path_to_publication_upload / REL_DIR_EVALUATION / Path("utils/db_config_mysql.py"),
            function=remove_part_between,
            str_find_remove='# %%% remove',
            str_prepend_file='# If you like to use connection to MySQL database as default, '
                             'replace the db_config.py file with this one.\n'
        )

        # repo-files including environment.yml
        shutil.copytree(
            src=DIR_EVALUATION_MODULE / Path("export/repo_files"),
            dst=self.path_to_publication_upload,
            dirs_exist_ok=True,
        )

        # database/sqlite.db
        shutil.copytree(
            src=DIR_EVALUATION_MODULE / Path("export/database"),
            dst=self.path_to_publication_sqlite.parent,  # same: self.path_to_publication_upload / Path("database"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("sqlite_no_views.db", "sqlite.db")  # sqlite.db will be created later
        )

        # all files in publication folder
        shutil.copytree(
            src=self.path_to_jupyter_folder,
            dst=self.path_to_publication_upload,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("upload", FILE_EMPTY_SQLITE_TEST),  # don't copy test file to upload
        )
        if not os.path.isdir(
            self.path_to_publication_upload / db_config_binder.REL_DIR_REPORTS
        ):
            # print("\x1b[33m", 'reports folder created', "\x1b[0m")
            os.mkdir(self.path_to_publication_upload / db_config_binder.REL_DIR_REPORTS)

    def create_empty_sqlite(self, path_to_sqlite, exclude_views=None, debug=False):
        """
        An empty sqlite file with the current db schema (without views) is created on the fly
         using mysql2sqlite available in conda env tools
        Afterwards run transfer_views_mysql2sqlite to transfer also view definitions.
        :return: None
        """

        create_file = True
        if os.path.isfile(path_to_sqlite):
            if user_input.user_input(
                text="Sqlite file already exists. Did you create it manually "
                     "and just want to add database VIEWS? If False existing file will be overwritten.\n",
                dtype="bool",
            ):
                create_file = False
                print("\x1b[33m", "Procede with VIEW creation", "\x1b[0m")
            else:
                os.remove(path_to_sqlite)
                create_file = True

        if create_file:
            print("\x1b[34m",
                  "Create empty sqlite file: ", path_to_sqlite,
                  "\x1b[0m")

            views_sorted = db.get_views_sorted(engine=self.engine_from_database)

            # Check that the base table for each view is defined in hte_data_documentation.view_information
            [db.derive_name_base_table(view) for view in views_sorted]

            config = db_config.get_db_config(user='hte_read')
            try:
                subprocess_result = subprocess.run(['sh',
                                DIR_CREATE_EMPTY_SQLITE_SH,
                                path_to_sqlite,#self.path_to_publication_sqlite, #path_to_sqlite_empty_tables,#'test.db',
                                config["database"],
                                config["user"],
                                config["password"],
                                config["host"],
                                " ".join(views_sorted),
                                ]
                               )

                if subprocess_result.returncode != 0:
                    print(subprocess_result)
                    raise RuntimeError('sh execution failed with returncode' + str(subprocess_result.returncode))

            except RuntimeError as e:
                print("\x1b[33m", 'Failed subprocess: \n', 'sh', DIR_CREATE_EMPTY_SQLITE_SH,
                      path_to_sqlite,  # self.path_to_publication_sqlite, #path_to_sqlite_empty_tables,#'test.db',
                      config["database"],
                      config["user"],
                      config["password"],
                       config["host"],
                      '"' + ' '.join(views_sorted) + '"',
                      '\n Run it within a jupyter terminal window and rerun the function', "\x1b[0m")
                #print("\x1b[33m", "You still need to create views afterwards. Run: ",
                #      f"self._create_sqlite_views(self, "
                #      f"path_to_sqlite={str(path_to_sqlite)}, "
                #      f"exclude_views={str(exclude_views)}, debug={str(debug)})",
                #      "\x1b[0m")
                raise RuntimeError('Follow error workaround and rerun function.')

        if self.debug:
            print("\x1b[33m", "Created empty sqlite file without views: ", path_to_sqlite, "\x1b[0m")
        self._create_sqlite_views(path_to_sqlite, exclude_views=exclude_views, debug=debug)
        print("\x1b[32m", "Successfully created empty sqlite file.", "\x1b[0m")

    def _create_sqlite_views(self, path_to_sqlite, exclude_views=None, debug=False):
        # Transfer Views
        if exclude_views is None:
            exclude_views = ["data_icpms_sfc_analysis_old"]

        for view in db.get_views_sorted(engine=self.engine_from_database):
            if view in exclude_views:
                print("Excluded view: ", view)
                continue
            print("CONVERT VIEW", view)

            with self.engine_from_database.begin() as con_mysql:
                create_view_statement = (
                    pd.read_sql(
                        """SHOW CREATE VIEW %s;""" % view, con=con_mysql  # , 'VIEW'
                    )
                    .loc[:, "Create View"]
                    .loc[0]
                )

            create_view_statement = mysql_to_sqlite.view_header(create_view_statement)
            create_view_statement = mysql_to_sqlite.time_intervals(
                create_view_statement, debug
            )
            create_view_statement = mysql_to_sqlite.timestampdiff(
                create_view_statement, debug
            )
            create_view_statement = mysql_to_sqlite.functions(create_view_statement)
            create_view_statement = mysql_to_sqlite.redundant_brackets(
                create_view_statement, debug
            )

            # print(create_view_statement)
            self.engine_sqlite = self.connect_sqlite(path_to_sqlite)
            with self.engine_sqlite.begin() as con_sqlite:
                # con_sqlite.execute("""DROP TABLE IF EXISTS %s""" % view) # not necessary if exclusion works fine
                con_sqlite.execute("""DROP VIEW IF EXISTS %s""" % view)
                con_sqlite.execute(create_view_statement)
                display(
                    pd.read_sql(""" SELECT * FROM %s;""" % view, con=con_sqlite)
                ) if debug else ""
            self.engine_sqlite.dispose()


    def _transfer_data_df2sqlite(self):
        """
        Transfers all data from Publication.DATABASE_DF to SQLITE database.
        :return:
        """
        # print()
        print("\x1b[34m", "Transfer experiments from DATABASE_DF to sqlite", "\x1b[0m")
        with self.engine_sqlite.begin() as con_sqlite:
            # loop through tables alphabetically
            for table_name in sorted(list(self.DATABASE_DF.keys())):
                table_data = self.DATABASE_DF[table_name]
                print(
                    "Add",
                    len(table_data.index),
                    "entries to table",
                    table_name,
                )
                # display(table)
                table_data.to_sql(table_name, con=con_sqlite, if_exists="append")
            if self.debug:
                display(
                    pd.read_sql(
                        """ SELECT *
                                    FROM sqlite_master
                                   WHERE type='table'
                                   ;""",  # , 'VIEW'
                        con=con_sqlite,
                    )
                )


def copy_and_replace(src, dst, function, str_prepend_file='', **kwargs):
    """
    Copy a file to another location while replacing lines according to separate function
    :param src: str or pathlib.Path
        path to the file which should be copied
    :param dst:  str or pathlib.Path
        path to where the file should be copied
    :param function: callable
        is called for each line and should return a formatted string as the line in th enew file
    :param str_prepend_file: str
        string which will be prepended to the copied file
    :param kwargs: keywaord arguments of fucntion
    :return:
    """
    with open(dst, 'w') as new_file:
        new_file.write(str_prepend_file+'\n')
        with open(src) as old_file:
            for line in old_file:
                new_file.write(function(line, **kwargs))


def remove_part_between(line, str_find_remove='# %%% remove'):
    """
    removes part of the given string as defined in a comment at the end of the line giving
    between start_string and end_string.
    :param line: str
        input line
    :param str_find_remove: str, default '# %%% remove'
        string which indicates a removal
    :return:
    """
    if str_find_remove in line:
        str_find_between = 'between'
        str_find_between_and = 'and'
        how_to_remove = line.split(str_find_remove)[1]
        if str_find_between in how_to_remove:
            str_start, str_end = [item.strip(' \n') for item in
                                  how_to_remove.split(str_find_between)[1].split(str_find_between_and)]
            # line_up_to_start, line_from_start_on
            line_removed_comment = line.split(str_find_remove)[0]
            idx_start = line_removed_comment.find(str_start) + len(str_start)
            newline = line_removed_comment[:idx_start]
            newline += line_removed_comment[idx_start:][line_removed_comment[idx_start:].find(str_end):]
            newline += '\n'
            print('Removed part of line, result: ', newline)
            return newline
        else:
            warnings.warn(
                'Found comment to remove line when uploading: ' + str_find_remove
                + '; But it is not developed how to remove.')
    else:
        return line


