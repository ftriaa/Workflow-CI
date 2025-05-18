FROM continuumio/miniconda3

WORKDIR /app

# Copy seluruh isi project ke dalam container
COPY . /app

# Update conda dan buat environment sekaligus
RUN conda update -n base -c defaults conda -y && \
    conda env create -f conda.yaml

# Gunakan shell dari conda env msml-env (ganti kalau nama env berbeda)
SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

# Jalankan script training (ganti kalau file .py ada di folder lain)
CMD ["python", "modelling.py"]
