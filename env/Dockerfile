FROM python:3.10-slim-buster
COPY . /hydra-examples
RUN pip install -r /hydra-examples/env/requirements.txt
WORKDIR /hydra-examples/examples
ENTRYPOINT [ "python", "0_basic/script.py" ]
CMD ["--multirun", "model=mlp,randomforest,svm", "dataset=moons,circles,blobs"]