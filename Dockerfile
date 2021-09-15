# Install ocean_model_skill_assessor
FROM condaforge/mambaforge:4.10.3-5 as conda

COPY conda-linux-64.lock .
RUN mamba create --copy -p /env --file conda-linux-64.lock && \
    conda clean -afy

COPY . /omsa
RUN conda run -p /env python -m pip install --no-deps /omsa

COPY notebooks /demos

COPY entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh"]