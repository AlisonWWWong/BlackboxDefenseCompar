<img src="transfer/plt/plot/logo.png" style="height: 60px;" align="right">

# BlackboxBench: A Comprehensive Benchmark of Defense Methods for Black-box Adversarial Attacks

<p align="center">
<img src="https://img.shields.io/badge/benchmark-lightseagreen" style="height: 20px;"> <img src="https://img.shields.io/badge/contributions-welcome-lemonchiffon.svg" style="height: 20px;">
</p>

<p align="center">
<a href="https://arxiv.org/abs/2312.16979"> Paper </a >•<a href="https://blackboxbench.github.io/index.html"> Leaderboard </a >
</p >

**BlackboxBench** is a comprehensive benchmark containing mainstream adversarial <u>**black-box attack methods**</u> implemented based on [PyTorch](https://pytorch.org/). It can be used to evaluate the adversarial robustness of any ML models, or as the baseline to develop more advanced attack and defense methods.


✨ BlackBoxBench will be continously updated by adding more attacks. ✨

✨ **You are welcome to contribute your black-box attack methods to BlackBoxBench!!**  See [how to contribute](#how-to-contribute)✨


---
<font size=5><center><b> Contents </b> </center></font>

- [Transfer-based attacks](#transfer-based-attacks)
    - [Quick start](#transfer-quick-start)
    - [Supported attacks](#transfer-supported-attacks)
    - [Supported datasets](#transfer-supported-datasets)
    - [Supported models](#transfer-supported-models)
- [Query-based attacks](#query-based-attacks)
   - [Quick start](#query-quick-start)
   - [Supported attacks](#query-supported-attacks)
   - [Supported datasets](#query-supported-dataset)
   - [Supported models](#query-supported-models)
- [Analysis Tools](#analysis-tools)
- [How to contribute](#how-to-contribute)
- [Citation](#citation)
- [Copyright](#copyright)
---



---
## <span id="query-based-attacks">Defense methods for query-based attacks</span>

### <span id="query-quick-start">💡Quick start</span>

For `Requirements` and `Quick start` of query-based black-box adversarial attacks in BlackboxBench, please refer to the README [here](query/README.md). 

### <span id="query-supported-attacks">💡Supported attacks</span>




---

## <span id="analysis-tools">Analysis tools</span>

Analysis tools will be released soon!



---

## <span id="how-to-contribute">How to contribute</span>

You are welcome to contribute your black-box attacks or defenses to BlackBoxBench! 🤩

In the following sections there are some tips on how to prepare you attack.

### 🚀 Adding a new transfer-based attack

##### 👣 Core function

We divide various efforts to improve I-FGSM into four distinct perspectives: data, optimization, feature and model. Attacks belonging to different perspectives can be implemented by modifying below blocks:

[input_transformation.py](transfer/input_transformation.py): the block registering various input transformation functions. Attacks from data perspective are most likely to happen here. For example, the key of DI-FGSM is randomly resizing the image, so its core function is defined here:

```
@Registry.register("input_transformation.DI")
def DI(in_size, out_size):
    def _DI(iter, img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        ...
        return padded
    return _DI
```

[loss_function.py](transfer/loss_function.py): the block registering various loss functions. Attacks from feature perspective are most likely to happen here. For example, the key of FIA is designing a new loss function, so its core function is defined here:

```
@Registry.register("loss_function.fia_loss")
def FIA(fia_layer, N=30, drop_rate=0.3):
    ...
    def _FIA(args, img, true_label, target_label, ensemble_models):
        ...
        return -loss if args.targeted else loss
    return _FIA
```

[gradient_calculation.py](transfer/gradient_calculation.py): the block registering various ways to calculate gradients. Attacks from optimization perspective are most likely to happen here. For example, the key of SGM is using gradients more from the skip connections, so its core function is defined here:

```
@Registry.register("gradient_calculation.skip_gradient")
def skip_gradient(gamma):
		...
    def _skip_gradient(args, iter, adv_img, true_label, target_label, grad_accumulate, grad_last, input_trans_func, ensemble_models, loss_func):
        ...
        return gradient
    return _skip_gradient
```

[update_dir_calculation.py](transfer/update_dir_calculation.py): the block registering various ways to calculate update direction on adversarial examples. Attacks from optimization perspective are most likely to happen here. For example, the key of MI is using the accumulated gradient as update direction, so its core function is defined here:

```
@Registry.register("update_dir_calculation.momentum")
def momentum():
    def _momentum(args, gradient, grad_accumulate, grad_var_last):
        ...
        return update_dir, grad_accumulate
    return _momentum
```

[model_refinement.py](transfer/model_refinement.py): the block registering various ways to refine the surrogate model. Attacks from model perspective are most likely to happen here. For example, the key of LGV is finetune model with a high learning rate, so its core function is defined here:

```
@Registry.register("model_refinement.stochastic_weight_collecting")
def stochastic_weight_collecting(collect, mini_batch=512, epochs=10, lr=0.05, wd=1e-4, momentum=0.9):
    def _stochastic_weight_collecting(args, rfmodel_dir):
        ...
    return _stochastic_weight_collecting
```

Design your core function and register it in the suitable `.py` file to fit into our unified attack pipeline.

##### 👣 Config file

 You should also fill a json file which is structured in the following way and put it in `transfer/config/<DATASET>/<TARGET>/<L-NORM>/<YOUR-METHOD>.py`. Here is an example from [transfer/config/NIPS2017/untargeted/l_inf/I-FGSM.json](transfer/config/NIPS2017/untargeted/l_inf/I-FGSM.json)):

```
{
  "source_model_path": ["NIPS2017/pretrained/resnet50"],
  "target_model_path": ["NIPS2017/pretrained/resnet50",
                        "NIPS2017/pretrained/vgg19_bn",
                        "NIPS2017/pretrained/resnet152"],
  "n_iter": 100,
  "shuffle": true,
  "batch_size": 200,
  "norm_type": "inf",
  "epsilon": 0.03,
  "norm_step": 0.00392157,
  "seed": 0,
  "n_ensemble": 1,
  "targeted": false,
  "save_dir": "./save",

  "input_transformation": "",
  "loss_function": "cross_entropy",
  "grad_calculation": "general",
  "backpropagation": "nonlinear",
  "update_dir_calculation": "sgd",
  "source_model_refinement": ""
}
```

Make sure your core function is well specified in the last six fields.



---

## <span id="citation">Citation</span>

If you want to use BlackboxBench in your research, cite it as follows:

```
@misc{zheng2023blackboxbench,
      title={BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks}, 
      author={Meixi Zheng and Xuanchen Yan and Zihao Zhu and Hongrui Chen and Baoyuan Wu},
      year={2023},
      eprint={2312.16979},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```



---

## <span id="copyright">Copyright</span>

The source code of this repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data ([SCLBD](http://scl.sribd.cn/index.html)) at The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.
