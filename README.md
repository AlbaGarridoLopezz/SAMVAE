<!-- ABOUT THE PROJECT -->
## SAMVAE🪇

Survival Analysis Multimodal model based on Variational Autoencoders. 

This repository provides:
* Download Guide for the multimodal Data
* Necessary scripts to train SAMVAE and see results 
* Validation metrics (C-index and IBS) adapted from PyCox.
* A script to generate result tables and plots as presented in the paper.
  
For more details, see full paper [here](https://arxiv.org/abs/2312.14651).


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
   git clone https://github.com/github_username/repo_name.git
   ```


Already trained SAVAE and SOTA models and dictionaries with results can be found in /savae/survival_analysis/.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You can specify different configurations or training parameters in utils.py for both SAVAE and State-Of-The-Art (SOTA) models. 

To preprocess data, run the following command:
   ```sh
   python survival_analysis/preprocess_data.py
   ```

To train/test SAMVAE and show results, run the following command:
   ```sh
   python survival_analysis/main_savae.py
   ```
To train/test SOTA models and show results, run the following command:
   ```sh
   python survival_analysis/main_sota.py
   ```

If you want to display comparison results between SAVAE and SOTA models, run the following command:
   ```sh
   python survival_analysis/results_display.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Interactive HTMLs -->
🔗 [Multimodal Survival Analysis for BRCA](https://AlbaGarridoLopezz/SAMVAE/savae-main/htmls/interactive_plot_brca_sa.html)


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
