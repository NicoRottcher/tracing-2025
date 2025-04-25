import os
import random
import string
import warnings
from pathlib import Path
import shutil  # copy files
import subprocess

import pandas as pd
import numpy as np
import sqlalchemy as sql  # Handling python - sql communication

import evaluation
import evaluation.utils.db as db
from evaluation.utils import user_input, db_config  # import user_input
from evaluation.export import tracking

REL_DIR_DB_SESSION = 'db_session'
FILE_DB_SESSION_CURRENT_DB = 'current_db.txt'
FILE_ENDING_DB_SESSION_CREDS = '_credentials.txt'
DIR_EVALUATION_MODULE = Path(evaluation.__file__).parent
DIRFILE_CREATE_EMPTY_MYSQL_SH = DIR_EVALUATION_MODULE / Path("export/create_empty_mysql.sh")
DIRFILE_DUMP_EXPORTED_MYSQL_SH = DIR_EVALUATION_MODULE / Path("export/dump_exported_mysql.sh")


class MysqlExport(tracking.TrackedExport):
    """
    subclass of TrackedExport for exporting to another mysql database for
    - students who would like to copy their data into an own database
    - development purposes, to test developed modules on another database
    """
    def __init__(self, debug=False, ):
        """
        Initialize MysqlExport
        :param debug: print extra info for all TrackedExport functions
        """
        super().__init__(debug=debug,)
        if not self.created:
            self.export_type = 'MysqlExport'
        elif self.export_type != 'MysqlExport':
            # avoid problems between different subclasses
            #     warnings.warn('There is a TrackedExport Object created '
            #                   'as ' + self.export_type +
            #                   'but not as PublicationExport object. '
            #                   'Overwrite self.export_type="PublicationExport".')
            self.created = False
        else:
            # self.export_type = 'MysqlExport'
            self.db_session_name = self.id_tracked_export  # input("Enter the session name (this will be used for the schema names): ")
            self.db_session_name_doc = f"{self.db_session_name}_documentation"
            self.db_host = "localhost"  # Database host
            self.db_port = "3306"  # Database host

            # Check if the directory exists, if not, create it
            prefix_personal_folder = 'dev_' if db_config.WHICH_JUPYTER == 'dev' else ''
            DIR_DB_SESSION_FOLDER = (db_config.DIR_JUPYTER_HOME
                                     / (prefix_personal_folder + self.name_user)
                                     / REL_DIR_DB_SESSION)
            if not DIR_DB_SESSION_FOLDER.is_dir():
                os.mkdir(DIR_DB_SESSION_FOLDER)
                print(f"Created directory: {DIR_DB_SESSION_FOLDER}")

            self.DIRFILE_DB_SESSION_CREDS = DIR_DB_SESSION_FOLDER / (self.db_session_name + FILE_ENDING_DB_SESSION_CREDS)

    def export_to_mysql(self,
                        if_exists="append",
                        tables_transfer_all_values=None):
        """

        :param if_exists: str, one of: ['append', 'truncate'], default 'append'
            append: the selected data will be added to the created database. If you want to add data
                which is already in the database (rerun the command) then this will fail.
            truncate: this will delete all data in the created database and add selected data afterwards
                Deletion has to be confirmed
        :param tables_transfer_all_values: list or None
            list of name of tables from which all data should be appended
            - but no links to these tables will be considered
        :return: None
        """
        # Setup empty database
        if not self.DIRFILE_DB_SESSION_CREDS.is_file():
            # easy assumption that db is created when credential file is created but sufficient for now
            print('you can run self.setup_mysql_session() also separately')
            self.setup_mysql_session()

        # Get linked_experiments
        linked_experiments = self.get_df_tracked_exports_exp()

        # Transfer Mysql prod to self.DATABASE_DF
        self._transfer_data_mysql2df(df_transfer_experiments=linked_experiments,
                                     tables_transfer_all_values=tables_transfer_all_values)

        # Transfer Mysql doc to self.DATABASE_documentation_DF
        self._transfer_documentation_mysql2df()

        # Transfer from DF to Mysql_session for both experimental data and documentation schema
        self._transfer_data_df2mysqlsession(df=self.DATABASE_DF,
                                            db_schema=self.db_session_name,
                                            if_exists=if_exists)
        self._transfer_data_df2mysqlsession(df=self.DATABASE_documentation_DF,
                                            db_schema=self.db_session_name_doc,
                                            if_exists=if_exists)

        print("\x1b[32m", f"Successfully exported to MYSQL database: {self.db_session_name}", "\x1b[0m")

    def dump_exported_mysql(self):
        """
        Create a dump of your database with the selected data.
        This is used if you want to setup a database with your selected data on a different computer/server
        :return:
        """
        db_prod_config = db_config.get_db_config(user='hte_exporter')
        subprocess.run(['sh',
                        DIRFILE_DUMP_EXPORTED_MYSQL_SH,
                        self.db_session_name,
                        self.db_session_name_doc,
                        db_prod_config["user"],
                        db_prod_config["password"],
                        self.db_host,
                        self.db_port,
                        ]
                       )

    def setup_mysql_session(self):
        """
        Setup a new Mysql database in which you want export your selected data.
        :return: None
        """

        # Check if file exists
        if self.DIRFILE_DB_SESSION_CREDS.is_file():
            raise FileExistsError(f'Credential file already exists: {self.DIRFILE_DB_SESSION_CREDS}')


        # Combine session name and username for database credentials
        db_session_user = f"{self.db_session_name}_user"
        db_session_password = generate_password()


        # Save credentials to DIRFILE_DB_SESSION_CREDS file
        with open(self.DIRFILE_DB_SESSION_CREDS, 'w') as f:
            f.write(f"user\tpassword\thost\tdatabase\n")
            f.write(f"{db_session_user}\t{db_session_password}\t{self.db_host}\t{self.db_session_name}\n")
        print(f"Credentials saved to {self.DIRFILE_DB_SESSION_CREDS}.")

        db_prod_config = db_config.get_db_config(user='hte_exporter')
        subprocess.run(['sh',
                        DIRFILE_CREATE_EMPTY_MYSQL_SH,
                        self.db_session_name,
                        self.db_session_name_doc,
                        db_session_user,
                        db_session_password,
                        db_prod_config["user"],
                        db_prod_config["password"],
                        self.db_host,
                        self.db_port,
                        self.from_database,
                        self.from_database_documentation
                        ]
                       )

    def _transfer_data_df2mysqlsession(self, df, db_schema, if_exists="append"):
        """
        :param if_exists: str one of 'append', 'truncate'
            append: will append data, fails if data already exists
            truncate: will delete all data beforehand
        Transfers all data from Publication.DATABASE_DF to Mysql session database.
        :return:
        """
        print("\x1b[34m", f"Transfer experiments from DATABASE_DF to Mysql: {db_schema}", "\x1b[0m")
        with self.engine_to_mysql().begin() as con_destination:
            # TODO: Currently FOREIGN_KEY_CHECKS must be disabled, how to circumvent?
            #  - loop through table names not alphabetically but according to foreign_key dependencies
            #  - with REFERENCED_TABLES inserted first
            con_destination.execute("SET FOREIGN_KEY_CHECKS = 0;")
            if if_exists == 'truncate':
                if user_input.user_input(
                        text=f"Do you want to delete all data from your copied database {db_schema} "
                             "to replace with data to be exported?"
                             "If not, I will go on with if_exists='append' trying to add data to existing data"
                             "This might fail, if you try to export the same data again\n",
                        dtype="bool",
                ):
                    protected_schemas = ['hte_data', 'hte_data_documentation']
                    if db_schema in protected_schemas:
                        raise Exception(f'You are not allowed to delete from one of {protected_schemas}')
                    for table_name in sorted(list(df.keys())):
                        print(f"Truncating: {db_schema}.{table_name}...")
                        con_destination.execute(f"TRUNCATE {db_schema}.{table_name};")

            # loop through tables alphabetically
            for table_name in sorted(list(df.keys())):
                table_data = df[table_name]
                print(f"{db_schema}.{table_name}: {len(table_data.index)} entries")
                # display(table)
                table_data.to_sql(table_name,
                                  con=con_destination,
                                  schema=db_schema,
                                  if_exists='append',  # 'replace' not possible as table shape does not stay consistent
                                  )
            con_destination.execute("SET FOREIGN_KEY_CHECKS = 1;")

            # when failed here, the above is anyway executed - transaction is not rolled back anymore
            for table_name in sorted(list(df.keys())):
                con_destination.execute(f"CALL ANALYZE_INVALID_FOREIGN_KEYS('{db_schema}', '{table_name}');")
                # will query ANALYZE_INVALID_FOREIGN_KEYS from default database defined in engine,
                # thus should work for prod and doc
                df_check = pd.read_sql('SELECT *  FROM INVALID_FOREIGN_KEYS', con=con_destination)
                if len(df_check.index) > 0:
                    raise Exception(f'FOREIGN KEY CONSTRAINTS are not fulfilled for {db_schema}.{table_name}')
            print(f"{db_schema}: All FOREIGN KEY CONSTRAINTS are fulfilled")

    def engine_to_mysql(self):
        if not self.DIRFILE_DB_SESSION_CREDS.is_file():
            raise FileNotFoundError(f'Credential file not found.')

        db_session_config = pd.read_csv(self.DIRFILE_DB_SESSION_CREDS, sep="\t", header=0).transpose().iloc[:, 0]
        return sql.create_engine(
                                "mysql+mysqlconnector://%s:%s@%s/%s"
                                % (db_session_config["user"],
                                   db_session_config["password"],
                                   db_session_config["host"],
                                   db_session_config["database"]),
                                echo=False,
                            )


def generate_password(length=12):
    """
    Function to generate a random password
    :param length: int
        length of the password
    :return:
    """

    # Define a set of characters to include in the password, excluding dangerous characters
    allowed_characters = string.ascii_letters\
                         + string.digits \
                         + string.punctuation.replace("'", "")\
                                             .replace('"','') \
                                             .replace('\\', '')  \
                                             .replace('@', '') #\
                                             # .replace('$', '') \
                                             # .replace('&', '') \
                                             # .replace('#', '') \
    password = ''.join(random.choice(allowed_characters) for i in range(length))
    return password





#