# need to dump existing Postgres database and import in this docker image/container

# syntax=docker/dockerfile:1

# dump existing data build stage
FROM postgres:17 

COPY ./init/01_schema.sql /docker-entrypoint-initdb.d/01_schema.sql

# When you have a full data dump, add it here:
# COPY ./init/02_seed_data.sql /docker-entrypoint-initdb.d/02_seed_data.sql