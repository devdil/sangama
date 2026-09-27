#!/bin/sh
set -eu
SANGAMA_DB_PASSWORD=$(cat /run/secrets/app_password)
export SANGAMA_DB_PASSWORD
psql -v ON_ERROR_STOP=1 --username postgres --dbname postgres <<'SQL'
\getenv app_password SANGAMA_DB_PASSWORD
CREATE ROLE sangama LOGIN PASSWORD :'app_password';
CREATE DATABASE sangama OWNER sangama;
SQL
unset SANGAMA_DB_PASSWORD
