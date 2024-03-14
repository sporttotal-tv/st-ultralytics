SHELL := $(shell which bash)
USER := st_user

################
# Inference tasks #
################

run_detection:
	python scripts/run.py detect_bboxes_from_video \
							--video_path '/mnt/ai-storage/jira/imtec356/ma25fad386_5100_5160/pano.mp4' \
							--court_mask_path '/mnt/ai-storage/jira/imtec356/ma25fad386_5100_5160/pano_mask.png' \
                         	--model_path '/mnt/ai-storage/jira/imtec356/models/20231231_yolov8x-albumentations.pt' \
                         	--model_type "yolov8" \
                         	--start_time 0 \
                         	--end_time 10 \
							--verbosity 1 \
                         	--sahi_inference False \
                         	--debug True

################
# Docker tasks #
################

IMAGE_NAME?=$(USER)-stultralytics
IMAGE_TAG?=latest
CONTAINER_NAME?=$(USER)-stultralytics-cont

docker-build:
	@echo "Creating DOCKER IMAGE WITH TAG:" $(IMAGE_NAME):$(IMAGE_TAG)
	docker build -f Dockerfile \
				 --tag $(IMAGE_NAME):$(IMAGE_TAG) .

NOTEBOOK_PORT?= 8893
TENSORBOARD_PORT?= 8894
docker-run:
	@echo "CREATING DOCKER CONTAINER WITH NAME:" $(CONTAINER_NAME)
	@echo "USING DOCKER IMAGE WITH TAG:" $(IMAGE_NAME):$(IMAGE_TAG)
	docker run -i -t --rm \
		--gpus all \
		--env-file $(cnf) \
		-v $(shell pwd):/multi-player-tracker/\
		-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) \
		-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT) \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

docker-stop:
	-docker stop $(CONTAINER_NAME)

docker-restart: docker-stop
	docker start $(CONTAINER_NAME) || make docker-run

docker-rm: docker-stop
	docker rm $(CONTAINER_NAME)

docker-rerun: docker-rm docker-run

docker-rmi:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

################
# Notebook tasks #
################

jupyter-install:
	pip install notebook==6.* jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user	

jupyter-start:
	rm -rf nohup.out
	nohup jupyter notebook --port $(NOTEBOOK_PORT) --ip=* --no-browser --allow-root &
	sleep 5
	jupyter notebook list

jupyter-stop:
	jupyter notebook stop $(NOTEBOOK_PORT)