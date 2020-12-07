# SOME
SOME: Reference-less Sub-Metrics Optimized for Manual Evaluations of Grammatical Error Correction  
Paper: https://www.aclweb.org/anthology/2020.coling-main.573.pdf

# Dependencies
- Python >= 3.6.0
- Pytorch >= 1.3.1
- transformers >= 3.0.2

# Trained models and Dataset
- Download trained model [here](https://drive.google.com/file/d/1uoAReQK3f5g9CEy8rV4haSzXll8NqVHW/view?usp=sharing).
- These model are trained on [TMU-GFM-Dataset](https://github.com/tmu-nlp/TMU-GFM-Dataset).

# How to use

```
python some.py [hypothesis file] [source file] [output_file] \
    --g-dir [directry path of grammar model] \
    --f-dir [directry path of fluency model ] \
    --m-dir [directry path of meaning model]
```
More option can be found ```python some.py -h```.  
The default weight of each model are tuned with sentence-level Kendall tau on [Grundkiewicz et al. (2015).](https://www.aclweb.org/anthology/D15-1)
More details can be found the paper.