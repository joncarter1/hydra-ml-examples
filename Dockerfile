FROM python:3.10-slim-buster
COPY . .
ARG EXAMPLE="0_basic"
RUN pip install -r ${EXAMPLE}/env/requirements.txt
WORKDIR ${EXAMPLE}
ENTRYPOINT [ "python", "script.py" ]