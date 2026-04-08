FROM continuumio/miniconda3:latest

WORKDIR /project

# Copy conda environment file and create environment
COPY backend/environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "mlproj", "/bin/bash", "-c"]

# Install poetry and project dependencies
COPY backend/pyproject.toml backend/poetry.lock backend/poetry.toml ./backend/
RUN cd backend && poetry install --no-root --no-interaction

# Copy project files
COPY backend/ ./backend/
COPY download_data.py ./
COPY Data/ ./Data/

# Make entrypoint executable
RUN chmod +x backend/entrypoint.sh

EXPOSE 8000

WORKDIR /project/backend

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlproj", "/bin/bash", "entrypoint.sh"]
