# need to dump existing Postgres database and import in this docker image/container

# syntax=docker/dockerfile:1

# dump existing data build stage
FROM postgres:17

COPY mealeon_dump.sql /docker-entrypoint-initdb.d/

RUN ["sed", "-i", "s/exec \"$@\"/echo \"skipping...\"/", "/usr/local/bin/docker-entrypoint.sh"]

RUN ["/usr/local/bin/docker-entrypoint.sh", "postgres"]

# final build stage
RUN postgres:17

COPY --from=dumper /data $PGDATA