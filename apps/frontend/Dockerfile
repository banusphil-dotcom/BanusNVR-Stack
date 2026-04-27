FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
ARG CACHEBUST=2
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
RUN chmod -R 755 /usr/share/nginx/html
# nginx.conf is mounted from host via docker-compose volumes
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
