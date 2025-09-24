<!-- ABOUT THE PROJECT -->
## SAMVAE🪇

Survival Analysis Multimodal model based on Variational Autoencoders. 

This repository provides:
* Necessary scripts to train SAMVAE for survival analysis and competing risks.
* A guide for downloading clinical, omics, and image data from the TCGA portal.
* Scripts for data preparation and preprocessing.
* Validation metrics (C-index and IBS) adapted from PyCox.
* A script to generate plots and interactive HTML visualizations.
* A script to generate result tables as presented in the paper.

For more details, see the full paper: [Deep Survival Analysis in Multimodal Medical Data](https://doi.org/10.48550/arXiv.2507.07804).

<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Ubuntu
* Python 3.10.0
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/AlbaGarridoLopezz/SAMVAE.git
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You can specify different configurations and training parameters for SAMVAE models in `utils.py`.  
For detailed instructions, see `experiments_guide.ipynb`.


To preprocess data, run the following command:
   ```sh
   python survival_analysis/preprocess_data.py
   ```

To train/test SAMVAE and show results, run the following command:
   ```sh
   python survival_analysis/main_samvae.py
   ```
To view the result tables, check the `.txt` files located in the `results` folder for each model and parameter combination.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Interactive HTMLs -->
🔗 [Multimodal Survival Analysis Models for Breast Cancer vs Kaplan-Meier](https://albagarridolopezz.github.io/SAMVAE/interactive_plot_brca_sa.html)

🔗 [Multimodal Survival Analysis Models for Lower Grade Glioma vs Kaplan-Meier](https://albagarridolopezz.github.io/SAMVAE/interactive_plot_lgg_sa.html)

🔗 [Best vs Worst Prognosis Patients in Breast Cancer Cohort](https://albagarridolopezz.github.io/SAMVAE/best_vs_worst_prognosis_interactive.html)


[//]: # (<!-- LICENSE -->)

[//]: # (## License)

[//]: # ()
[//]: # (Distributed under the XXX License. See `LICENSE.txt` for more information.)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- CONTACT -->
## Contact

Alba Garrido - alba.garrido.lopez@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[//]: # (<!-- ACKNOWLEDGMENTS -->)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)
