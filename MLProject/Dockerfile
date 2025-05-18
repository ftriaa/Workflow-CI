FROM continuumio/miniconda3

WORKDIR /app

# Copy folder MLProject dan semua isinya
COPY MLProject/ ./MLProject

# Update conda dan buat environment
RUN conda update -n base -c defaults conda -y && \
    conda env create -f MLProject/conda.yaml

SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

CMD ["python", "MLProject/modelling.py"]
