# syntax=docker/dockerfile:1

FROM node:26-alpine AS build

WORKDIR /app

# Install dependencies first for layer caching
# npm ci requires package-lock.json — run `npm install` locally first
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY index.html ./
COPY vite.config.js ./
# COPY public/ ./public/
COPY src/ ./src/

RUN npm run build

# Stage 2: serve with nginx
FROM nginx:stable-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]