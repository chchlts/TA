FROM node:14-alpine as builder

WORKDIR /app

COPY package.json .
COPY package-lock.json .
RUN npm install
COPY . .

# RUN mv .env.development .env
ENV NODE_OPTIONS="--max-old-space-size=2048"
RUN npm run build

FROM nginx:alpine
WORKDIR /usr/share/nginx/html

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist .
CMD ["nginx", "-g", "daemon off;"]
