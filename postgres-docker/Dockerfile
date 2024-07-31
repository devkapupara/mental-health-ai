# Use the official PostgreSQL image from the Docker Hub
FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=mental
ENV POSTGRES_PASSWORD=bot
ENV POSTGRES_DB=memory

# Add the initialization script to the Docker entrypoint
COPY init.sql /docker-entrypoint-initdb.d/

# Expose the default PostgreSQL port
EXPOSE 5432