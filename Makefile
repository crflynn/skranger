.PHONY: build
build:
	rm -rf build/
	rm -f skranger/ensemble/ranger.cpp
	rm -f skranger/ensemble/ranger.cpython*
	rm -f skranger/ensemble/ranger.html
	poetry run python build.py clean
	poetry run python build.py build_ext --inplace --force

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

