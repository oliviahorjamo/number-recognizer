from invoke import task

@task
def start(ctx):
    ctx.run("python3 src/index_mnist.py", pty=True)

@task
def pylint(ctx):
    ctx.run("pylint src", pty=True)

@task
def coveragereport(ctx):
    ctx.run("coverage run --branch -m pytest src", pty=True)
    ctx.run("coverage report -m", pty=True)
    ctx.run("coverage html", pty=True)

@task
def test(ctx):
    ctx.run("pytest src", pty=True)

@task
def format(ctx):
    ctx.run("autopep8 --in-place --recursive src --max-line-length 100")


@task
def timecomplexity(ctx):
    ctx.run("python3 src/time_complexity_tests.py", pty=True)