# STRAT


---

## Benchmark Datasets

We evaluate **STRAT** across six benchmark datasets:
* **Amazon**: `Beauty`, `Sports & Outdoors`, `Video Games`, `Toys & Games`
* **Yelp**
* **MovieLens-1M** (`ML-1M`)

### Data Preparation

You can obtain the processed interaction files (`.inter`) in two ways:

* **Automatic Download**: RecBole downloads supported datasets automatically when you first run them, storing files in the `dataset/` folder.
* **Manual Download**: You can download files directly from official sources:
  * **GitHub**: [RUCAIBox/RecDatasets](https://github.com/RUCAIBox/RecDatasets)
  * **Google Drive**: [Processed Datasets](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing)
  * **Baidu Netdisk**: [Baidu Netdisk Link](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g) (Code: `e272`)

> [!TIP]
> When downloading manually, extract datasets into `dataset/<dataset_name>/` (e.g., `dataset/beauty/`, `dataset/ml-1m/`).

---

## Quick Start

Run `run_recbole.py` from the project root to train and evaluate **STRAT**:

```bash
python run_recbole.py --model=STRAT --dataset=<dataset_name>
```

### Examples

* **Train on Beauty with default settings**:
  ```bash
  python run_recbole.py --model=STRAT --dataset=beauty
  ```

* **Train with custom parameters** (e.g., learning rate and embedding size):
  ```bash
  python run_recbole.py --model=STRAT --dataset=beauty --learning_rate=0.001 --embedding_size=64
  ```

---

## Best Hyperparameters

Default model settings are configured in [recbole/properties/model/STRAT.yaml](recbole/properties/model/STRAT.yaml).

The best parameter combinations for each dataset are listed below. You can copy and run the corresponding command directly:

| Dataset | `lambda_freq_init` | `lambda_time_init` | `laplacian_mix_init` | `laplacian_tau_init` | `routing_strength` | Run Command |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Beauty** | `0.2` | `0.05` | `0.5` | `0.05` | `0.1` | `python run_recbole.py --model=STRAT --dataset=beauty --lambda_freq_init=0.2 --lambda_time_init=0.05 --laplacian_mix_init=0.5 --laplacian_tau_init=0.05 --routing_strength=0.1` |
| **Sports** | `0.1` | `0.2` | `0.3` | `2.0` | `1.0` | `python run_recbole.py --model=STRAT --dataset=sports --lambda_freq_init=0.1 --lambda_time_init=0.2 --laplacian_mix_init=0.3 --laplacian_tau_init=2.0 --routing_strength=1.0` |
| **Video** | `0.2` | `0.1` | `0.01` | `2.0` | `0.75` | `python run_recbole.py --model=STRAT --dataset=video --lambda_freq_init=0.2 --lambda_time_init=0.1 --laplacian_mix_init=0.01 --laplacian_tau_init=2.0 --routing_strength=0.75` |
| **Toys** | `0.1` | `0.2` | `0.05` | `0.05` | `0.1` | `python run_recbole.py --model=STRAT --dataset=toys --lambda_freq_init=0.1 --lambda_time_init=0.2 --laplacian_mix_init=0.05 --laplacian_tau_init=0.05 --routing_strength=0.1` |
| **Yelp** | `0.02` | `0.1` | `0.05` | `2.0` | `0.25` | `python run_recbole.py --model=STRAT --dataset=yelp --lambda_freq_init=0.02 --lambda_time_init=0.1 --laplacian_mix_init=0.05 --laplacian_tau_init=2.0 --routing_strength=0.25` |
| **ML-1M** | `0.05` | `0.05` | `0.1` | `0.5` | `0.75` | `python run_recbole.py --model=STRAT --dataset=ml-1m --lambda_freq_init=0.05 --lambda_time_init=0.05 --laplacian_mix_init=0.1 --laplacian_tau_init=0.5 --routing_strength=0.75` |

---

## Acknowledgments

This codebase is built upon [RecBole](https://github.com/RUCAIBox/RecBole). We thank the authors for their open-source contribution.
