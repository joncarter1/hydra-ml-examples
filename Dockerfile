FROM python:3.10-slim-buster
COPY examples /examples
ARG EXAMPLE="0_basic"
RUN pip install -r /examples/${EXAMPLE}/env/requirements.txt
WORKDIR /examples/${EXAMPLE}
ENTRYPOINT [ "python", "script.py" ]