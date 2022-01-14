.PHONY: build


build:
	docker build . -t dnikku/jnbooks

run:
	#docker run --rm dnikku/jnbooks
	-docker stop my-jnbooks && docker rm my-jnbooks
	docker run -d -p 8888:8888 -v $$(pwd)/nbooks:/home/nbooks --name my-jnbooks dnikku/jnbooks
	# see: https://serverfault.com/questions/984578/change-permissions-for-named-volumes-in-docker (option 3)
	#docker exec -u 0:0 my-jnbooks chown -R $$UID /home/nbooks
