# --- build stage: compile the SPA to static files ---
FROM node:22-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
# Backend base URL is baked in at build time (Vite inlines VITE_* into the
# bundle). Override per environment with --build-arg VITE_API_BASE_URL=...
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# --- serve stage: static files behind nginx ---
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
