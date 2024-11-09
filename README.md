# LLM Projector
![colorband](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/colorband.png)
[![plot](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/plot.gif)](https://htmlpreview.github.io/?https://github.com/vitalstarorg/projector/blob/master/nbs/plot.html)
## What are you seeing...
The image shows how LLM GPT2 transforms the prompt "Alan Turing theorized that the" through each layers of transformer to find the next word "universe".
 
Since the dimension of the embedding vectors are 786 in GPT2, we project these higher dimensional vectors to a 3d space for ease of visualization. The little fluffy cloud are the 50257 encoded tokens in the LLM. Both the origin of higher dimension and center of the embedding cloud are projected as a black dot and a black cross. If you click on the animated image, it will show you the 3d interactive graph that you can further examine the details of each transition. For example, if you follow the blue dots from te5, pe5, 0, 1, 2 ... 11, you will see how the last word in the prompt, the token "the" got transformed to "universe" after 12 layers of the transformer.
## Motivation
A picture worth a thousand words. With this interactive 3d projection done by PCA, it helps us to understand the effect of each transformation through a LLM transformer. Since it is a linear transformation from 768d to 3d, it gives us clues on the function of each element inside the LLM e.g. position encoding, layer norm, attention, MLP and the final logit by examining their relative and changing positions. The picture above is just one use case of this projector library.
## Projector
Projector is a research tool. We develop this to help us to understand layers of LLM transformation at higher dimensions by visualizing its effect through projected 3d vector space. Emphasis is to make intuitive use of the library objects e.g. _projector_, _projection_ and _trace_, in order to avoid misinterpretation of these vectors.
### Main Objects
#### Projector
#### Operator
#### Trace
#### Inference
#### Projection
### Developer Note
#### Setup for Unit Tests and Notebook
If you want to get your hand dirty to test out the library, here are the steps
```bash
# Setup Python Env
conda create -y --name=prj "python=3.9"
conda activate prj
pip install -r requirements-darwin.txt	# for Apple Silicon
pip install -r requirements-linux.txt	# for Linux


# Run unit tests
export LLM_MODEL_PATH=$(pwd)/model		# directory for storing models
rm project*.zip					# remove cached files
pytest --disable-warnings --tb=short tests
	# you should see 3 test suites
# tests/test_00projector.py   ... passed
# tests/test_01gpt2.py        ... passed
# tests/test_02gpt2np.py      ... failed (pls refers to next section)


# Run the notebook to generate the image and plot
jupyter lab -y --NotebookApp.token='' --notebook-dir=. nbs/projector.ipynb
```
#### Use of Numpy
We have implemented the transformer using both numpy and pytorch. Numpy was our initial implementation by following [picoGPT](https://github.com/jaymody/picoGPT). We put the test harness on this implementation to make sure all math is right. Based on this harness, we reimplement the same object using pytorch as Huggingface is using pytorch. We hope we could reuse the same library for other Huggingface models.
In order to setup to use Numpy, we need to download the GPT2 checkpoint files and put them in the model directory. Then the 3rd test should pass.
```bash
cd model
mkdir gpt2-chkpt
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/checkpoint
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.index
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.meta
wget https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
```
#### Use of TDD
This is an on-going project to support our LLM research. We employ TDD to guide this development and also serve as our main documentation, so you will find all available features illustrated in the form of unit tests.  We may develop more notebooks and other resources in the future to illustrate the use of each feature.  Additional features and tests will be added when we need more visualization in the cause of this research.
Please refers to this [notebook](https://github.com/vitalstarorg/projector/blob/main/nbs/projector.ipynb) to see how the image and the plot got generated.
#### Use of Builder Pattern
You may find one main design pattern driving through most code is the builder pattern. In general, most object method named in as a noun is a property method. When it is called without argument, it is a getter for retrieving its value. When it is called with an argument, it is a setter for setting value, and returns the object initiating the call. For example, every object would have a name() method.
```python
pj = Projector()
projector.name('pj')              # set its name
projector.name()                  # get its name


# Alternatively we can do this
pj = Projector().name('pj')
```
#### Use of Smallscript
Smallscript is a library that implements a small language called Smalltalk as a gluing script to orchestrate different objects together to express high level expression. For example, the whole GPT2 transformer can be expressed as follows.
```python
infer = model.inference().prompt("Alan Turing theorized that the")
x = infer.ssrun("""
       self wte | wpe
           | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum
           | layer: 1 | layer: 2 | layer: 3
           | layer: 4 | layer: 5 | layer: 6 | layer: 7
           | layer: 8 | layer: 9 | layer: 10 | layer: 11
           | fnorm""")            # x is the tensor output of the transformer.
trace = pj.newTrace().fromVectors(x)   # projector creates a trace for visualization.
trace.show()                      # show the projected vector.
```
**Note**
- The pipe character works like a Linux pipe, piping the return value for the next method. Since we are using a builder pattern, they are all piping the same @infer object.
- `| lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum` is the expansion of `layer: 0` into finer details for illustration purpose.
