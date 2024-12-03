# MoveSim
Codes for paper in KDD'20 (AI for COVID-19): Learning to Simulate Human Mobility

## Datasets

* **GeoLife**: This GPS trajectory dataset was collected by the MSRA GeoLife project with 182 users in a period of over five years.  
  * Link: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

## Requirements

* **Python = 3.10.9** 
* **opacus = 1.2.0** 
* **numpy = 1.24.3** 
* **scipy = 1.10.1** 
* **pytorch = 2.0.1** 
* **pytorch-cuda = 11.8**
## Usage

Pretrain and train new model:

`python main.py --pretrain --data=geolife`

**Other commands**
* --epsilon, -e: desired privacy budget, if -1 means train without DP.
* --delta: desired delta in (epsilon, delta) DP.
* --skipm: Skip M1 and M2 matrix generation.
* --load: Load pretrained models (form folder pretrain)
* --cuda: either 'cpu' if no cuda available or the number of the GPU when listing cuda devices.

Evaluation with generated data:

`python evaluation.py`
