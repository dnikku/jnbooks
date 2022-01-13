.PHONY: build


build:
	docker build . -t dnikku/jnbooks

run:
	#docker run --rm dnikku/jnbooks
	-docker stop my-jnbooks && docker rm my-jnbooks
	docker run -d -p 8888:8888 --name my-jnbooks dnikku/jnbooks
