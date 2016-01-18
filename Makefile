data/sample_submission.csv.zip:
	wget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/noaa-right-whale-recognition/download/sample_submission.csv.zip -O data/sample_submission.csv.zip --continue

data/imgs.zip:
	wget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/noaa-right-whale-recognition/download/imgs.zip -O data/imgs.zip --continue

data/train.csv.zip:
	localhostwget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/noaa-right-whale-recognition/download/train.csv.zip -O data/train.csv.zip --continue

data: data/sample_submission.csv.zip data/imgs.zip data/train.csv.zip

.PHONY: data
