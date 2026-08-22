\# NeuroRoute



NeuroRoute is an experimental neuromorphic deployment pipeline that converts a molecular classification ANN into a spiking neural network and studies its placement, multicast routing, router pressure, link bandwidth, and timing on a SpiNNaker-style architecture.



\## Project capabilities



\- Reproducible BBBP molecular preprocessing

\- PyTorch ANN training

\- ANN-to-SNN conversion with snnTorch

\- PyNN simulation using the Brian2 backend

\- Official sPyNNaker virtual-board integration

\- PACMAN virtual mapping

\- SpiNNaker-style machine topology

\- Population partitioning into machine vertices

\- Naive and connectivity-aware placement

\- Multicast routing-tree construction

\- Router-table pressure validation

\- Link bandwidth, queue, latency, and packet-drop simulation

\- Optuna hyperparameter optimization

\- MLflow experiment tracking and Model Registry promotion

\- DVC dataset versioning

\- Deployment packaging and validation

\- Automated tests



\## Results



\### ANN baseline



\- Test ROC-AUC: 0.9143

\- Test accuracy: 0.8660

\- Test precision: 0.9447

\- Test recall: 0.8761

\- Test F1: 0.9091



\### Converted SNN



\- Test ROC-AUC: 0.9055

\- Test accuracy: 0.7124

\- Test F1: 0.7744



\### Routing optimization



| Metric | Naive | Connectivity-aware |

|---|---:|---:|

| Used chips | 16 | 10 |

| Independent unicast hops | 310 | 72 |

| Multicast tree links | 152 | 72 |

| Maximum router entries | 35 | 36 |

| Maximum link route load | 24 | 24 |



The optimized placement reduced unicast hops by 76.77% and multicast links by 52.63%.



\### Hyperparameter optimization



The best partition configuration used:



\- 64 neurons per core

\- 7 application cores per chip

\- 3 chips

\- 20 machine vertices

\- 14 multicast tree links

\- 19 maximum router entries

\- No router overflow



\### Link simulation



\- Active directed links: 22

\- Peak link utilization: 15.36%

\- Dropped packets: 0

\- Maximum queue: 0 packets

\- Average route latency: 0.408 ms

\- Maximum route latency: 0.666 ms



\## Installation



The main project uses Python 3.11.



```cmd

py -3.11 -m venv .venv

.venv\\Scripts\\activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements-lock.txt

