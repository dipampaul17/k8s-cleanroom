FROM python:3.10-slim

# Install curl for downloading kube-linter, and clean up after
RUN apt-get update && \
    apt-get install -y curl && \
    curl -L -o /usr/local/bin/kube-linter https://github.com/stackrox/kube-linter/releases/download/v0.7.2/kube-linter-linux && \
    chmod +x /usr/local/bin/kube-linter && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir openai pyyaml gitpython

# Copy kube-linter config
COPY .kube-linter.yaml /.kube-linter.yaml

# Copy your grader script
COPY grader.py /grader.py

# Set entrypoint
ENTRYPOINT ["python", "/grader.py"] 