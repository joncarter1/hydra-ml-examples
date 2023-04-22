FROM python:3.10-slim-buster
COPY . /hydra-examples
RUN pip install -r /hydra-examples/requirements.txt
ENTRYPOINT [ "python", "hydra-examples/basic-example/script.py" ]
CMD ["--multirun", "model=mlp,randomforest,svm", "dataset=moons,circles,blobs"]