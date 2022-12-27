# Manual for the project

## Installing the dependencies and starting the program


Install dependencies with the command:

```
poetry install
```

Start the program with the command:

```
poetry run invoke start
```

## Run tests


Run unittests with the command:

```
poetry run invoke test
```

Generate the coverage report with the command:

```
poetry run invoke coveragereport
```

Test the effect of different parameters with the command:

```
poetry run invoke timecomplexity
```

## Format code

Generate pylint analysis with the command:
```
poetry run invoke pylint
```
Fix pylint errors with:
```
poetry run invoke format
```

