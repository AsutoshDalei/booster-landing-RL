# Development Environment for SpaceX Landing Simulator
FROM node:18-alpine

# Install a simple static server with live reload capability
RUN npm install -g live-server

WORKDIR /app

# The default live-server port is 8080
EXPOSE 8080

# Run live-server on the current directory
CMD ["live-server", "--port=8080", "--host=0.0.0.0", "--no-browser", "."]
