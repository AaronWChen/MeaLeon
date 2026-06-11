# syntax=docker/dockerfile:1
#
# TWO MODES — switch between them as the frontend matures:
#
# MODE 1 (current): serve the existing static HTML directly from nginx.
#   No Node.js build needed. Gets the container stack running today.
#   The static/index.html already has the full working UI.
#
# MODE 2 (future): multi-stage React build.
#   Uncomment the build stage and swap the COPY source once the React
#   app in frontend/src/ is ready (package.json populated, components written).
#   See the commented block below.

# ── Mode 1: static HTML (active) ─────────────────────────────────────────

FROM nginx:stable-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Serve the existing static site from the repo root's static/ directory.
# In compose.yml, bind-mount it: volumes: - ../static:/usr/share/nginx/html:ro
# That way changes to static/ are reflected without rebuilding the image.

# ── Mode 2: React build (uncomment when ready) ───────────────────────────
#
# FROM node:22-alpine AS build
# WORKDIR /app
# COPY package.json package-lock.json ./
# RUN npm ci
# COPY ./src ./src
# COPY ./public ./public
# RUN npm run build
#
# FROM nginx:stable-alpine
# COPY --from=build /app/build /usr/share/nginx/html
# COPY nginx.conf /etc/nginx/conf.d/default.conf
