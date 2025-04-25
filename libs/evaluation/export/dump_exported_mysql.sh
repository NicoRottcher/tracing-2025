#!/bin/bash

if [ "$#" -lt 6 ]; then #[ -z "$1" ]; then
    echo "Usage: $0 <db_session_name>
                    <db_session_name_doc>
                    <db_exporter_user>
                    <db_exporter_pw>
                    <db_host>
                    <db_port> "
    exit 1
fi


db_session_name=$1
db_session_name_doc=$2
db_exporter_user=$3
db_exporter_pw=$4
db_host=$5
db_port=$6

error_log="error_log_export_mysql.txt"



# 2. Dump the current database with data, routines, triggers, and events
sql_dump_file="exported__${db_session_name}.sql"
echo "Dumping exported database to $sql_dump_file. This can take some time..."

mysqldump --databases "${db_session_name}" "${db_session_name_doc}" --routines --skip-triggers --skip-events --no-tablespaces -h "$db_host" -P "$db_port" -u "$db_exporter_user" -p"$db_exporter_pw" > "$sql_dump_file" 2>"${error_log}" || {
    echo "Error during dump of empty database schema. Check ${error_log} for details."
    cat "${error_log}"
    exit 1
}

echo "Done dump database ${db_session_name} and ${db_session_name_doc}.
You can now download the file and import it using MYSQL Workbench"

