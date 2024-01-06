# ecfr: Eliot's CodeForces Runner
This command line utility is meant to make it easy to download and
run programs for use in CodeForces contests. It will automatically
download contest problems, including sample test cases, and will
compile and run code against these test cases. A fork of
[CodeforcesRunner](https://github.com/sayuan/CodeforcesRunner).

Also includes starter files. Meant to be a "batteries included"
way of doing Codeforces contests.

## Installation
The package is available on [PyPI](https://pypi.org/project/ecfr/). It can
be installed with the following command:
```sh
pip install ecfr
```

## Commands
This command line utility has four commands:
```sh
dc    Download contest or individual problems
init  Copy example files into the current working directory.
r     Run code against contest problems.
sp    Create an empty solution file from a starter file in the source files folder.
```
Each one has different arguments that can be viewed by looking at the help messages.

## Usage
We will use [Codeforces Contest 198](https://codeforces.com/contest/198)
as a running example. This will demonstrate the workflow of using `ecfr` as a
command line utility.

### Initialization
The first thing we must do is create an environment config file and provide
starter files for languages we would like to use. Fortunately, `ecfr` includes
a sample config and starter files with reasonable defaults. To use them, we must
simply navigate to the directory we would like to do the contest in and run
```sh
$ ecfr init
```

### Download Sample Tests
Now that we have starter files and a basic config, we need to download the contest.
We can download contest 198 using the following command.
```sh
$ ecfr dc 198
```
To download just problem A, we can add the `-p A` arguments. This creates a
`.xml` for each problem downloaded, including the sample test cases.
You may add additional test cases by following the format of the `.xml` files.
By default, these are put into the `contest_files` directory.

### Starting a Problem
Now that we have our problem files downloaded, we can start coding. To do this,
we need to copy a starter file for our desired language and give it a descriptive
name, according to the problem we will be solving. Let's say we wish to do problem A
in the contest we downloaded in Python. We can create our file with the following
command:
```sh
$ ecfr sp A py
```
This will create a file called `A.py` in the `source_files` directory (the location can
be set in the config). Here, `py` is the file extension `ecfr` will look for when making
a copy of the starter file. The directory for starter files can be set in the config file.

### Running the Tests
Once we have our starter file and finish coding, we want to run the tests that we downloaded.
To do this, we simply run the following command:
```sh
$ ecfr r A
```
With this command, `ecfr` will look in the `source_files` directory for files matching problem
A. When a file is found, it will automatically compile and run it against all test cases in
the `A.xml` file.

### Submitting
Once all test cases pass, we wish to submit our source file. This is as easy as selecting
`A.py` and uploading the file on the contest submission page. Make sure to select the
appropriate programming language. If you are using Python 3, be sure and submit to PyPy3,
as this generally runs much faster.

## Configurations
The file `conf.json` contains the used compile and execute commands. More commands
can be added for different languages. In addition, names of directories for source files,
contest files, and executables can be changed.

## About
This tool is only verified on Linux and is still considered in beta. Once more testing
has been done and more test cases have been added, we will proceed with a full release.

## Acknowledgements
Thanks to [sayuan](https://github.com/sayuan) and all of the contributors to the original
[CodeforcesRunner](https://github.com/sayuan/CodeforcesRunner) project.
Additional thanks to [dhashe](https://github.com/dhashe) for help testing this project.
