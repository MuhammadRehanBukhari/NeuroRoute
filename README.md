<div align="center">



\# 🧠 NeuroRoute



\### Neuromorphic model conversion, placement, multicast routing, and MLOps



\[!\[Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-ANN-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)

\[!\[PyNN](https://img.shields.io/badge/PyNN-Brian2-6A5ACD)](https://neuralensemble.org/PyNN/)

\[!\[SpiNNaker](https://img.shields.io/badge/sPyNNaker-Virtual\_Board-00A67E)](https://spinnakermanchester.github.io/)

\[!\[MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow\&logoColor=white)](https://mlflow.org/)

\[!\[DVC](https://img.shields.io/badge/DVC-Data\_Versioning-945DD6?logo=dvc\&logoColor=white)](https://dvc.org/)

\[!\[Tests](https://img.shields.io/badge/Tests-47\_Passing-brightgreen)](#testing)



NeuroRoute converts a molecular-classification neural network into a spiking

network and evaluates how it can be partitioned, placed, and routed across a

SpiNNaker-style neuromorphic architecture.



</div>



\---



\## 📑 Contents



\- \[Why NeuroRoute?](#-why-neuroroute)

\- \[System architecture](#-system-architecture)

\- \[Features](#-features)

\- \[Results](#-results)

\- \[Project structure](#-project-structure)

\- \[Installation](#-installation)

\- \[Running the project](#-running-the-project)

\- \[MLOps workflow](#-mlops-workflow)

\- \[Testing](#-testing)

\- \[SpiNNaker integration](#-spinnaker-integration)

\- \[Limitations](#-limitations)



\## 🎯 Why NeuroRoute?



Neuromorphic systems process information using spikes and distributed

computation. Deploying an SNN requires more than training a model: populations

must be partitioned into machine vertices, assigned to chips, connected using

multicast routes, and checked against hardware limits.



NeuroRoute provides an end-to-end experimental workflow for studying those

deployment decisions.



\## 🏗 System architecture



```mermaid

flowchart TD

&#x20;   A\["BBBP molecules"] --> B\["RDKit fingerprints"]

&#x20;   B --> C\["PyTorch ANN"]

&#x20;   C --> D\["snnTorch SNN"]

&#x20;   D --> E\["PyNN simulation"]

&#x20;   D --> F\["Population partitioning"]

&#x20;   F --> G\["Connectivity-aware placement"]

&#x20;   G --> H\["Multicast route trees"]

&#x20;   H --> I\["Bandwidth and timing simulation"]

&#x20;   C --> J\["MLflow Registry"]

&#x20;   A --> K\["DVC"]

&#x20;   F --> L\["Optuna optimization"]

&#x20;   J --> M\["Deployment bundle"]

&#x20;   I --> M

```



\## ✨ Features



| Area | Implementation |

|---|---|

| Data | BBBP validation, RDKit fingerprints and deterministic splits |

| ANN | PyTorch binary molecular classifier |

| SNN | ANN-to-SNN conversion with snnTorch |

| PyNN | Local simulation through the Brian2 backend |

| SpiNNaker | Official sPyNNaker and PACMAN virtual-board mapping |

| Partitioning | Population-to-machine-vertex decomposition |

| Placement | Round-robin and connectivity-aware strategies |

| Routing | Mesh routes and multicast route trees |

| Hardware model | Chip, core, router-table and link constraints |

| Traffic | Bandwidth, queue, latency and packet-drop simulation |

| Optimization | Optuna partition hyperparameter search |

| MLOps | MLflow experiments, artifacts and Model Registry |

| Data versioning | DVC-tracked raw and processed data |

| Deployment | Versioned bundle, checksums and independent validation |

| Quality | 47 automated tests |



\## 📊 Results



\### ANN and SNN performance



| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |

|---|---:|---:|---:|---:|---:|

| PyTorch ANN | \*\*0.8660\*\* | \*\*0.9143\*\* | 0.9447 | 0.8761 | \*\*0.9091\*\* |

| Converted SNN | 0.7124 | 0.9055 | 0.9679 | 0.6453 | 0.7744 |



The SNN retained a strong ROC-AUC while using spike-based computation.



\### Placement optimization



| Metric | Round-robin | Connectivity-aware | Change |

|---|---:|---:|---:|

| Used chips | 16 | \*\*10\*\* | −37.50% |

| Independent unicast hops | 310 | \*\*72\*\* | −76.77% |

| Multicast tree links | 152 | \*\*72\*\* | −52.63% |

| Maximum router entries | \*\*35\*\* | 36 | +2.86% |

| Maximum link route load | 24 | 24 | No change |



> Connectivity-aware placement substantially reduced communication distance

> while remaining within router capacity.



\### Link simulation



| Metric | Result |

|---|---:|

| Active directed links | 22 |

| Source packets | 21,054.69 |

| Link transmissions | 93,003.30 |

| Peak utilization | 15.36% |

| Average route latency | 0.408 ms |

| Maximum route latency | 0.666 ms |

| Maximum queue | 0 packets |

| Dropped packets | \*\*0\*\* |

| Overloaded links | \*\*0\*\* |



\### Hyperparameter optimization



Optuna selected:



```yaml

neurons\_per\_core: 64

max\_application\_cores\_per\_chip: 7

used\_chips: 3

machine\_vertices: 20

multicast\_tree\_links: 14

maximum\_router\_entries: 19

router\_overflow\_events: 0

objective\_score: 55.25

```



\## 📁 Project structure



```text

neuroroute/

├── configs/                     # Experiment and hardware configurations

├── data/                        # DVC-managed datasets

├── src/neuroroute/

│   ├── data\_prep.py             # BBBP preprocessing

│   ├── train\_ann.py             # ANN training and MLflow logging

│   ├── convert\_snn.py           # ANN-to-SNN conversion

│   ├── pynn\_demo.py             # PyNN/Brian2 example

│   ├── partition\_sim.py         # Population partitioning

│   ├── placement.py             # Placement algorithms

│   ├── routing.py               # Point-to-point routing

│   ├── multicast.py             # Multicast route trees

│   ├── link\_sim.py              # Bandwidth and latency simulation

│   ├── hpo\_partition.py         # Optuna optimization

│   ├── promote\_model.py         # MLflow model promotion

│   ├── package\_deployment.py    # Deployment packaging

│   └── validate\_deployment.py   # Bundle validation

├── tests/                       # Automated tests

├── requirements-lock.txt

├── requirements-spinnaker-lock.txt

└── README.md

```



\## ⚙️ Installation



\### Main environment



```cmd

py -3.11 -m venv .venv

.venv\\Scripts\\activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements-lock.txt

```



\### Separate sPyNNaker environment



```cmd

py -3.11 -m venv .venv-spinnaker

.venv-spinnaker\\Scripts\\activate

python -m pip install -r requirements-spinnaker-lock.txt

```



The separate environment prevents sPyNNaker dependencies from conflicting with

the main ML environment.



\## ▶️ Running the project



Activate the main environment:



```cmd

.venv\\Scripts\\activate

```



\### 1. Start MLflow



```cmd

start "NeuroRoute MLflow" cmd /K "call .venv\\Scripts\\activate \&\& mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000"

```



Open \[http://127.0.0.1:5000](http://127.0.0.1:5000).



\### 2. Prepare data



```cmd

python -m src.neuroroute.data\_prep --config configs\\baseline.yaml

```



\### 3. Train the ANN



```cmd

python -m src.neuroroute.train\_ann --config configs\\baseline.yaml

```



\### 4. Convert to an SNN



```cmd

python -m src.neuroroute.convert\_snn --config configs\\snn.yaml

```



\### 5. Run the PyNN demonstration



```cmd

python -m src.neuroroute.pynn\_demo

```



\### 6. Compare placement strategies



```cmd

python -m src.neuroroute.partition\_compare

```



\### 7. Simulate link traffic



```cmd

python -m src.neuroroute.link\_sim --config configs\\hardware.yaml

```



\### 8. Optimize partitioning



```cmd

python -m src.neuroroute.hpo\_partition

```



\### 9. Validate deployment



```cmd

python -m src.neuroroute.validate\_deployment

```



\## 🔄 MLOps workflow



```mermaid

flowchart LR

&#x20;   A\["DVC data"] --> B\["Training"]

&#x20;   B --> C\["MLflow run"]

&#x20;   C --> D\["Candidate"]

&#x20;   D --> E\["Champion"]

&#x20;   E --> F\["Validated bundle"]

```



MLflow tracks:



\- Configurations and hyperparameters

\- Training and validation curves

\- ANN and SNN metrics

\- Partitioning comparisons

\- Optuna parent and child runs

\- Models and artifacts

\- Deployment-validation events



The ANN is registered as:



```text

NeuroRoute-BBBP-ANN

```



Promotion uses the `candidate` and `champion` aliases.



\## ✅ Testing



Run the complete test suite:



```cmd

pytest -v

```



Verified result:



```text

47 passed in 19.13s

```



Tests cover data preparation, model output, artifact loading, partition

constraints, placement, routing, multicast trees, bandwidth, latency, queues,

and packet drops.



\## 🧩 SpiNNaker integration



NeuroRoute demonstrates:



\- PyNN API familiarity using Brian2

\- Official sPyNNaker installation

\- Virtual SpiNNaker-5 board setup

\- PACMAN graph partitioning and mapping

\- Integration-readiness checks for the SpiNNaker toolchain



A physical board or authorized spalloc account was not available, so the

project does \*\*not\*\* claim physical-hardware execution.



\## ⚠️ Limitations



\- No physical SpiNNaker/spalloc execution

\- Virtual mapping uses the installed SpiNNaker-5 toolchain

\- Router simulation is architectural, not cycle-accurate

\- No claim of bit-exact SpiNNaker2 router behavior

\- ANN and SNN experiments used CPU execution

\- Demonstration currently uses only the BBBP dataset



\## 🔮 Future work



\- Execute the mapped network through an authorized spalloc service

\- Validate simulation estimates against hardware counters

\- Add cycle-level router arbitration

\- Add fault-aware placement and rerouting

\- Support additional molecular datasets

\- Add continuous integration and container deployment



\## 📜 License



This repository is intended for educational and research use.



\---



<div align="center">



\*\*NeuroRoute — from neural models to neuromorphic routes\*\*



</div>

