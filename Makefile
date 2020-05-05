.PHONY: fmt
fmt:
	poetry run isort -y
	poetry run black .

.PHONY: build
build:
	poetry run python build.py build_ext --inplace --force