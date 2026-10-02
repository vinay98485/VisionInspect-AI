# Stage 1: Build React App
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .

# Pass API URL at build time
ARG VITE_API_BASE
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy the custom Nginx config for React Router fallback
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built assets
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
