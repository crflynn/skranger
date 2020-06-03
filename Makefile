.PHONY: build
build:
	poetry run python build.py clean
	poetry run python build.py build_ext --inplace --force

.PHONY: copy
copy:
	poetry run python buildpre.py

.PHONY: dist
dist: copy
	poetry build

.PHONY: docker
docker:
	docker build -t skranger .

.PHONY: linux
linux: copy docker
	docker run --rm -v $(shell pwd)/dist:/app/dist:rw skranger build

.PHONY: docs
docs:
	cd docs && \
	poetry run sphinx-build -M html . _build -a && \
	cd .. && \
	open docs/_build/html/index.html

.PHONY: fmt
fmt:
	poetry run isort -y
	poetry run black .

.PHONY: sdist
sdist: copy
	poetry build -f sdist

.PHONY: setup
setup:
	git submodule init
	git submodule update
	asdf install
	poetry install --no-root
	poetry run python buildpre.py
	poetry install

.PHONY: test
test:
	poetry run pytest --cov=skranger --cov-report=html tests/
	open htmlcov/index.html
