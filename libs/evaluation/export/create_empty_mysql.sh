#!/bin/bash

# """
#Script for exporting an mysql of the empty schema
#Created in 2025
#@author: Nico Röttcher
#"""

if [ "$#" -lt 10 ]; then #[ -z "$1" ]; then
    echo "Usage: $0 <db_session_name>
                    <db_session_name_doc>
                    <db_session_user>
                    <db_session_password>
                    <db_exporter_user>
                    <db_exporter_pw>
                    <db_host>
                    <db_port>
                    <db_export_name_prod>
                    <db_export_name_doc> "
    exit 1
fi


db_session_name=$1
db_session_name_doc=$2
db_session_user=$3
db_session_password=$4
db_exporter_user=$5
db_exporter_pw=$6
db_host=$7
db_port=$8
db_export_name_prod=$9
db_export_name_doc=${10}

error_log="error_log.txt"

# 1. Create new schemas for both hte_data and hte_data_documentation, create user and grant privileges
echo "Creating databases ${db_session_name} and ${db_session_name_doc}"
echo "Setup MySQL user ${db_session_user} and granting all privileges on both schemas..."

mysql -h "$db_host" -P "$db_port" -u "$db_exporter_user" -p"$db_exporter_pw" -e "CALL ${db_export_name_doc}.create_db_copy_session_user('${db_session_user}', '${db_session_password}', '${db_session_name}', '${db_session_name_doc}');" 2>"${error_log}" || {
    echo "Error creating database schema, user or granting privileges. Check ${error_log} for details."
    cat "${error_log}"
    exit 1
}
echo "Done create schema, user.

"


# 2. Dump the current database with data, routines, triggers, and events
sql_dump_file="dump__${db_export_name_prod}__to__${db_session_name}.sql"
echo "Dumping production database schema (no data) to $sql_dump_file. This can take some time..."

mysqldump --databases "${db_export_name_prod}" "${db_export_name_doc}" --routines --skip-triggers --no-data --skip-events --no-tablespaces -h "$db_host" -P "$db_port" -u "$db_exporter_user" -p"$db_exporter_pw" > "$sql_dump_file" 2>"${error_log}" || {
    echo "Error during dump of empty database schema. Check ${error_log} for details."
    cat "${error_log}"
    exit 1
}

echo "Done dump empty schema.

"


# 3. Perform the substitutions using sed to create a modified dump for the documentation schema
sql_dump_file_substituted="${sql_dump_file}__substituted.sql"
echo "Performing substitutions on $sql_dump_file ..."
sed -e "s/${db_export_name_prod}\`/${db_session_name}\`/g" \
    -e "s/${db_export_name_prod}\./${db_session_name}\./g" \
    -e "s/${db_export_name_prod}'/${db_session_name}'/g" \
    -e "s/${db_export_name_doc}/${db_session_name_doc}/g" \
    "$sql_dump_file" >"$sql_dump_file_substituted" 2>"${error_log}" || {
      echo "Error substituting in dump of ${db_export_name_prod}. Check ${error_log} for details."
      cat "${error_log}"
      exit 1
}

# Remove DEFINER clauses from the dump
sed 's/\sDEFINER=`[^`]*`@`[^`]*`//g' -i "$sql_dump_file_substituted" 2>"${error_log}" || {
      echo "Error substituting definer in dump of ${db_export_name_prod}. Check ${error_log} for details."
      cat "${error_log}"
      exit 1
}
echo "Substitutions complete. Modified file saved as $sql_dump_file_substituted

"


# 4. Load the dump into the new database
echo "Loading dumped schema into new database ${db_session_name}. This can take some time..."

# mysql -h $db_host -P $db_port -u $db_user -p"$password" $db_schema_dev < "$sql_dump_file_substituted" 2>load_errors.log
mysql -h "$db_host" -P "$db_port" -u "$db_session_user" -p"$db_session_password" "$db_session_name" < "$sql_dump_file_substituted" 2>"${error_log}" || {
    echo "Error: Failed to load the dump into new database. Check ${error_log} for details."
    cat "${error_log}"
    exit 1
}

echo "Dump loaded successfully into ${db_session_name}.

"

# 5. Calling setup_session.py script to extract and populate the database --> do this in python again
#echo "Extracting and populating database..."
#python3 db_export.py --db-host "$db_host" --db-port "$db_port" \
#    --db-user "$db_user" --db-password "$password" --db-name "$db_schema_dev" \
#    --documentation-db "$db_schema_doc"
#
#if [ $? -ne 0 ]; then
#    echo "Error: Failed to extract and populate the database."
#    exit 1
#fi
#echo "Database extracted and populated successfully."

