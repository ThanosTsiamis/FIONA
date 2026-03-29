# FIONA: FInding Outliers iN Attributes

<img src="client_side/public/LogoFIONA.png" alt="A minimalistic one-line of a woman fused into a tree" width="200" title="Logo of FIONA"/>

FIONA, short for "Finding Outliers in Attributes" is an innovative framework designed for the detection of categorical
outliers within datasets. This configuration-free and user-friendly tool specializes in identifying unusual patterns in
data attributes, making it invaluable for data analysis and decision-making processes. FIONA's key strengths lie in its
ability to capture syntactic structures, its deterministic nature, and its adaptability to various encodings and
languages, making it a powerful and accessible tool for outlier detection in diverse datasets.

FIONA is the Master's Thesis of Thanos Tsiamis
at Utrecht University for the program of Computing Science. It was developed under
the supervision of Dr. A.A.A. (Hakim) Qahtan for the academic year of 2022 - 2023. 

## How to run

There are 2 ways to run the project: (i) [Locally](#local-execution) and (ii) in [Docker](#docker).

**We strongly advise towards the second way (Docker) due to its simplicity.**

### Local execution

**Important Note**: Make sure that Node.js and Python are installed in your system.

For the **front end**:

One-liner in terminal: `cd client_side && npm install && npm run dev`

On Windows, use semicolons if needed.

- Open a terminal and cd to the `client_side` folder
- Run the command `npm install`
- Run the command `npm run dev`

For the **back end**:

One-liner in terminal: `pip install -r api/requirements.txt && python -m api.main`

- Open a second terminal and stay in the project root
- Install the necessary requirements from `api/requirements.txt`
- Run the command `python -m api.main`

The frontend runs on [http://localhost:3000](http://localhost:3000) and the backend runs on [http://localhost:5000](http://localhost:5000).

### Running tests

From the project root, run:

- `python3 -m unittest discover -s tests -v`
- `npm --prefix client_side test`
- `npm --prefix client_side run lint`

Or use the root package helper:

- `npm run test:api`
- `npm run test:frontend`
- `npm run lint:frontend`

These tests currently cover API helper logic, route-level upload/fetch/history flows, Flask integration coverage when backend dependencies are installed, dataset-based regression checks, and frontend page behavior.

### Docker

**_Important Note_**: A docker installation is required for it to work. More info on how to
install it [here](https://docs.docker.com/get-docker/).

- Once Docker is installed, navigate to the directory where you have the FIONA project files.

- Open a terminal or command prompt.

- For local development with bind mounts and live code changes, run:
  `docker compose up --build`
- For a more production-like run that uses the production Compose file and the production frontend image target, run:
  `docker compose -f docker-compose.prod.yml up --build`

Please make sure that approximately 2GB of storage are available for the 2 docker containers.

The default [docker-compose.yml](/Users/thanostsiamis/PycharmProjects/FIONA/docker-compose.yml) is now development-focused and always builds from local source. The production-style setup lives in [docker-compose.prod.yml](/Users/thanostsiamis/PycharmProjects/FIONA/docker-compose.prod.yml).
Container health checks are enabled for both services. The API health endpoint is available at [http://localhost:5000/api/health](http://localhost:5000/api/health).

## How to use

After the system is up and running, head to [http://localhost:3000](http://localhost:3000) and upload the file you want
outliers to be detected in the corresponding form.

## Note

Development was done on Windows 10 with Firefox and Chrome.

## Contact

We're always looking to improve our project, so any input you have is greatly appreciated. If you encounter any issues
or bugs, please report them on our GitHub issues page.
