# syntax=docker/dockerfile:1

FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies first for layer caching
# npm ci requires package-lock.json — run `npm install` locally first
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY index.html ./
COPY vite.config.js ./
COPY public/ ./public/
COPY src/ ./src/

RUN npm run build

# Stage 2: serve with nginx
FROM nginx:stable-alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
