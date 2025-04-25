#!/bin/bash

# whoami

# Check if an environment name is provided
if [ "$#" -lt 6 ]; then #[ -z "$1" ]; then
    echo "Usage: $0 <mysql-username> <mysql-dbname> <mysql-username> <mysql-password> <mysql-host> <tables-excluded>"
    exit 1
fi


SQLITE_FILE=$1
MYSQL_DB=$2
MYSQL_USER=$3
MYSQL_PW=$4
MYSQL_HOST=$5
TBLS_EXCLUDED=$6

# echo $TBLS_EXCLUDED

CONDA_BASE=$(conda info --base)
. "$CONDA_BASE/etc/profile.d/conda.sh"

# conda env list
conda activate tools
mysql2sqlite -f "$SQLITE_FILE" -d "$MYSQL_DB" -u "$MYSQL_USER" --mysql-password "$MYSQL_PW" -W -e $TBLS_EXCLUDED -h "$MYSQL_HOST"
# $TBLS_EXCLUDED must not be quoted! multiple tables will be written here