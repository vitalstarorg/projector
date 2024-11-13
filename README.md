# LLM Projector
![colorband](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/colorband.png)
[![plot](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/plot.gif)](https://htmlpreview.github.io/?https://github.com/vitalstarorg/projector/blob/master/nbs/plot.html)
## What are you seeing...
The image shows how GPT2 transforms the prompt "Alan Turing theorized that the" through each layers of transformer to find the next word "universe".
 
Since the dimension of the embedding vectors are 786d in GPT2, we project these higher dimensional vectors to a 3d space for ease of visualization. The little fluffy cloud are the 50257 encoded tokens in the LLM. Both the origin of the higher dimension and center of the embedding cloud are projected as a black dot and a black cross. If you click on the animated image, it will show you the 3d interactive graph that you can further examine the details of each transition. For example, if you follow the blue dots from te5, pe5, 0, 1, 2 ... 11, you will see how the last word in the prompt, the token "the" got transformed to "universe" after 12 layers of the transformer.
## Motivation
A picture worth a thousand words. With this interactive 3d projection done by PCA, it helps us to understand the effect of each transformation through a LLM transformer. Since it is a linear transformation from 768d to 3d, it gives us clues on the function of each element inside the LLM e.g. position encoding, layer norm, attention, MLP and the final logit by examining their relative and changing positions. The picture above just shows one scenario to illustrate the use of this library. The same library can be used to investigate the effect of the attention mechanism, and compare the effects of using different similarity schemes, and etc. In time, we will share our findings in notebooks.
## Projector
Projector is a research tool. We develop this to help us to understand layers of LLM transformation at higher dimensions by visualizing its effect through projected 3d vector space. Emphasis is to make intuitive use of the library objects e.g. _projector_, _projection_ and _trace_, to avoid misinterpretation of these vectors when we examine the effects with different LLM architectures.
### Main Objects
#### Projector
`Projector` is the main object that aggregates a few helper objects to manipulate and project high dimensional vectors to a 3d space. It can render a single vector or a tensor of vectors in one rendering. It also provides interactivity to help to explore these vectors and their relationship. The main method to provide the interactivity is `Projector.onClick()`.


The vector visualization is through a trace object created by the projector. The trace object will carry a few key objects from the projector to project and render a vector to the 3d space e.g. projection, similarity, inference, etc. 


`Projector` will first set up a stage using plotly FigureWidget with a token embedding cloud as a background before rendering a projected vector from the high dimensional embedding space. This embedding cloud serves as a reference to help us to visualize embedding vectors in relationship to this cloud.


`Projector` also sets up a `ColorShape` to keep track of the color and shape used in rendering. We can use different colors and shapes to assign different meanings for each rendering.


`Projector` can save some of its helper objects which hold on to the state of the projection to a cache file e.g. PCA projection, current view and camera position so that we can consistently render these vectors and understand their transitions through inferencing.


Below is a typical setup to use a projector to visualize the projection from a single 768d vector in 3d space. In this case we use the origin. You may consult this [notebook](https://github.com/vitalstarorg/projector/blob/main/nbs/projector.ipynb) to see how we use the library to produce the above animation.
```python
model = GPT2Operator().name("gpt2").downloadModel()	# download gpt2 from huggingface
pj = Projector().name('projector').model(model)
pj.loadCache()                                          # load cached view and camera
pj.showEmbedding()                                      # show embedding cloud
trace = pj.newTrace().name('origin'). \
            label('origin').color('black'). \
            fromVectors(0).show()
```
#### Operator
`Operator` encapsulates a GPT model loading and model parameters access. It also provides a pure form of the GPT2 transformer implementation. Since it works like an adaptor behaving like a gpt model object, we always name this object `model` in our code. We name this class `Operator` to avoid confusion with Huggingface `Model`.


We have two implementations. `GPT2Operator` is a PyTorch implementation as our default. `GPT2OperatorNP` is a Numpy implementation. These math implementations are verified and validated by unit tests to ensure its correctness. This harness helps to ensure the math is correct when we are manipulating the high level objects during experiments. We use this to examine different transformer schemes e.g. redefine the transformer architecture, changing similarity measure with different projection, changing the normalization, etc.


`Operator` provides quick access to the nearest tokens of a high dimension vector with some additional measurements. This helps to keep track and appreciate the transitions on how the transformer transforms the embedding vectors in the higher dimensional space.
#### Trace
`Trace` objects are created by the `Projector` object which carries its state objects for both projection and rendering. We can create a trace for a single vector in embedding space, or we can create a trace with multiple vectors as a list or tensor.


`Trace` objects collaborate with the projector as a visitor to encapsulate its rendering logic and customization within. So we have another variant called `Line` to show different ways of connecting these vectors together e.g. line or radial segments with optional arrows.


`Trace` also provides access to the calculation of neighboring vectors based on the defined similarity and their corresponding angles.


`Trace` also implemented the final linear transformation of gpt2. The significance of this implementation is crucial in the visualization. Since most transformed vectors are projected out of visualization range i.e. their norms are large. It is this final transformation, to bring them close to the embedding cloud; therefore, this final transformation is applied to these intermediate vectors in the above animation.


In order to understand the true effect of each layer of the transformer to these vectors, `Trace` allows us to define a uniform weight and bias instead of the original final linear transformation. Therefore you may find a choice of use of different linear transforms in our code.
```python
lnfw = pj.model().modelParams().getValue('ln_f.w')
lnfb = pj.model().modelParams().getValue('ln_f.b')
pj.wOffset(0.4); pj.bOffset(lnfb);     # using uniform weight with original bais
# or
pj.wOffset(lnfw); pj.bOffset(lnfb);    # using original weight and bias
```
#### Projection
`Projection` encapsulates the projection state and logic to make it a changeable component in our research. Currently we are using the first 3 PCA components as our final projection. Surprisingly we found that this calculation is highly sensitive, and susceptible to truncation errors. The propagated deviation would cause a significant change in our visualization. We will further investigate its numerical stability. One thing we can tell, there are a few dimensions in the embedding cloud that contribute a significant portion of the vector norms. In theory it will destabilize the GPT2 training and inference as it shares the same normalization procedure. Hope to finalize and share our findings soon.


The saving and loading of this object in the `Projector` helps to keep this state consistent in different experiments and inferencing. If we use a larger GPT model, it would reduce the computation during the setup time for a projector.
#### Inference
Inference encapsulates the whole transformer process from transforming a prompt to the final predictions. This encapsulation is especially important to safeguard the validity and accuracy of the math when we are composing different transformer architecture using the same model e.g. skip, repeat and swap layers.


Below is the GPT2 architecture. It is different from the original transformer, particularly the position of the layer norm.


<div align="center">
    <img src="https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/gpt2-architecture.png" alt="gpt2 architecture" width="25%">
</div>


The following code is the representation of this transformer in Python. 
```python
infer = self.model.inference().prompt("Alan Turing theorized that the")


infer.wte().wpe()
for layer in range(infer.nlayer()):
    infer.lnorm1(layer).attn(layer).sum()
    infer.lnorm2(layer).ffn(layer).sum()
infer.fnorm()                # final normalization
infer.logits()               # output 50257d logits
infer.argmax()               # next likely predicted token ids
infer.generation()           # next likely predicted tokens


# Alternative, we can express the transformer using Smallscript
infer.ssrun("""self wte | wpe
       | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum
       | layer: 1 | layer: 2 | layer: 3
       | layer: 4 | layer: 5 | layer: 6 | layer: 7
       | layer: 8 | layer: 9 | layer: 10 | layer: 11
       | fnorm | logits""")  # output 50257d logits
```
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
     # tests/test_02gpt2np.py      ... failed (as expected)


# Run the notebook to generate the image and plot
jupyter lab -y --NotebookApp.token='' --notebook-dir=. nbs/projector.ipynb
```
#### Setup Docker
A simple docker file is provided so that you can do the same thing using Docker.
```bash
docker build -t projector -f Dockerfile .
docker run -it -p 8888:8888 \
     -v $(pwd):/home/ml/projector \
     -v $HOME/model:/home/ml/model \
     --name projector --rm projector /bin/bash


# Inside the container
export LLM_MODEL_PATH=$HOME/model
cd $HOME/projector
rm project*.zip
pytest --disable-warnings --tb=short tests
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
You may find one main design pattern driving through most code is the builder pattern. In general, most object method named in as a noun is a property method. When it is called without argument, it is a getter for retrieving its value. When it is called with an argument, it is a setter for setting value, and returns the object initiating the call. For example, every object would have a name() method. One reason for adopting this pattern is to make setting up a complex object e.g. projector with a bunch of helper objects intuitively. Below is an example.
```python
pj = Projector()
projector.name('pj')              # set its name
projector.name()                  # get its name


# Alternatively we can do this
pj = Projector().name('pj')
```
#### Use of Smallscript
Smallscript is a library that implements a small language, a variant of Smalltalk, as a gluing script to orchestrate different objects together to express high level concepts intuitively. For example, the whole GPT2 transformer can be expressed as follows. With the help from underlying object design, it allows us to experiment different LLM architectures at ease without worrying of misinterpreting the vectors during visualization i.e. we could believe what we see.
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
- Usually the last method is retrieving the final output other than the original object e.g. `self wte | wpe | layer:0 | … | fnorm | logits`.
- This particular line of script `| lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum` is a detailed expansion of `layer: 0` for illustration purposes.
