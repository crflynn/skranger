# Use the python slim image
FROM python:3.8.3-slim

# Add build deps for python packages
# curl to install vendored poetry
# g++ to build sksurv
RUN apt-get update && \
    apt-get install curl g++ -y && \
    apt-get clean

# Set the working directory to app
WORKDIR /app

# Set the poetry version explicitly
ENV POETRY_VERSION=1.1.4
# Unbuffer the logger so we always get logs
ENV PYTHONUNBUFFERED=1
# Update the path for poetry python
ENV PATH=/root/.poetry/bin:/root/.local/bin:$PATH

# Install vendored poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Add dep configs
ADD ./pyproject.toml .
ADD ./poetry.lock .

# Install packages
# Disable virtualenv creation so we just use the already-installed python
RUN poetry config virtualenvs.create false && \
    poetry run pip install pip==20.0.2 && \
    poetry install && \
    rm -r ~/.cache/pip

# Add everything
ADD . .

# Build skranger
RUN poetry run python buildpre.py
RUN poetry install

# Set the entrypoint to poetry
ENTRYPOINT ["poetry"]
