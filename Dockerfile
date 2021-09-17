# Install ocean_model_skill_assessor
FROM condaforge/mambaforge:4.10.3-5 as conda

COPY conda-linux-64.lock .
RUN mamba create --copy -p /env --file conda-linux-64.lock && \
    conda clean -afy

COPY . /omsa
RUN conda run -p /env python -m pip install --no-deps /omsa

# Temporary hack for getting demo up and running
COPY notebooks /demos

RUN mkdir -p /root/.ocean_data_gateway/variables
COPY demo-helper-files/* /root/.ocean_data_gateway/variables

COPY entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]