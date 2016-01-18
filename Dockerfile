FROM felixlaumon/deeplearning

ADD . /kaggle-right-whale
WORKDIR /kaggle-right-whale

RUN pip install -r requirements.txt
