FROM node:20-bookworm-slim

WORKDIR /app

COPY app/package*.json ./
RUN npm install --include=dev --no-audit --no-fund \
    && node -e "console.log(require.resolve('expo/package.json'))"

COPY app ./

EXPOSE 19000 19001 19002 8081

CMD ["npx", "expo", "start", "--lan", "--clear"]
