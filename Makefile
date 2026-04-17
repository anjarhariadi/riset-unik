ENGINE ?= podman
REGISTRY ?= ghcr.io/anjarhariadi
TAG ?= latest

# image naming
FRONTEND_IMAGE = $(REGISTRY)/risetunik-web:$(TAG)
BACKEND_IMAGE  = $(REGISTRY)/risetunik-server:$(TAG)

clean-image:
	$(ENGINE) rmi $(BACKEND_IMAGE) $(FRONTEND_IMAGE) --force

# Build image frontend
build-frontend:
	$(ENGINE) rmi $(FRONTEND_IMAGE) --force 
	$(ENGINE) build -t $(FRONTEND_IMAGE) ./web

push-frontend:
	$(ENGINE) push $(FRONTEND_IMAGE)

build-and-push-frontend: build-frontend push-frontend

# Build image backend
build-backend:
	$(ENGINE) rmi $(BACKEND_IMAGE) --force
	$(ENGINE) build -t $(BACKEND_IMAGE) ./server

push-backend:
	$(ENGINE) push $(BACKEND_IMAGE)

build-and-push-backend: build-backend push-backend

# Build semua image sekaligus
build-all: build-frontend build-backend

push-all: push-frontend push-backend

# Build and push semua image sekaligus
build-and-push-all: build-frontend build-backend push-frontend push-backend

.PHONY: build-frontend build-backend build-all \
	   push-frontend push-backend push-all \
	   build-and-push-frontend build-and-push-backend build-and-push-all \
	   clean-image