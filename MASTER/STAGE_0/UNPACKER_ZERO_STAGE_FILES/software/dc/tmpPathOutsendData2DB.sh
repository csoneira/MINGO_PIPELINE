PGPASSWORD=pass psql -U user -p port -h 10.10.10.10  -d "systemName" -f - <<EOF

CREATE TEMPORARY TABLE tmp_table AS
