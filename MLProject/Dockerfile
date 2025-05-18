FROM continuumio/miniconda3

WORKDIR /app

# Copy seluruh isi MLProject ke dalam container
COPY MLProject/ ./MLProject

# Update conda dan install environment
RUN conda update -n base -c defaults conda -y && \
    conda env create -f MLProject/conda.yaml

# Gunakan shell dari conda env untuk run command berikutnya
SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

# Jalankan script training saat container dijalankan
CMD ["python", "MLProject/modelling.py"]
