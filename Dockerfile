FROM python:3.12-slim

WORKDIR /app

# Install deps first (cache layer)
COPY pyproject.toml README.md LICENSE ./
COPY caveman/ caveman/
RUN pip install --no-cache-dir -e ".[all]"

# Create caveman home with required directories
RUN mkdir -p /root/.caveman/workspace \
             /root/.caveman/memory \
             /root/.caveman/skills \
             /root/.caveman/plugins \
             /root/.caveman/cron \
             /root/.caveman/trajectories \
             /root/.caveman/sessions

# Verify installation
RUN caveman version && python -m caveman version

ENTRYPOINT ["caveman"]
CMD ["run", "-i"]
