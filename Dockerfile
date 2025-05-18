FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/ ./MLProject

RUN conda update -n base -c defaults conda && \
    conda env create -f MLProject/conda.yaml

SHELL ["conda", "run", "-n", "msml-env", "/bin/bash", "-c"]

CMD ["mlflow", "run", "MLProject", "--no-conda"]
