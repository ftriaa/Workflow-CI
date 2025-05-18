FROM continuumio/miniconda3

WORKDIR /app

# Copy semua isi folder MLProject (konteks build) ke /app
COPY . .

RUN conda update -n base -c defaults conda -y && \
    conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

CMD ["python", "modelling.py"]
