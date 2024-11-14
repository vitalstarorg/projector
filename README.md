# LLM Projector
![colorband](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/colorband.png)
[![plot](https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/plot.gif)](https://htmlpreview.github.io/?https://github.com/vitalstarorg/projector/blob/master/nbs/plot.html)
## What Are You Seeing? Click to See More...
The image above illustrates how GPT-2 transforms the prompt "_Alan Turing theorized that the_" through each layer of transformer model to predict the next word, which is "_universe_".

We project the 768D embedding vectors into a 3D space for visualization. The "fluffy cloud" represents the 50,257 embedded tokens in GPT-2. The origin and the center of the embedding cloud are marked with a black dot and a black cross, respectively. Clicking on the animated image will open an interactive 3D graph that lets you explore the details of each transformation. By following the blue dots from "te5, pe5, 0, 1, 2 ... 11," you can see how the last word in the prompt, the token "_the_", got transformed to "_universe_" after passing through the 12 transformer layers.

## Motivation
A picture is worth a thousand words. This interactive 3D projection, created using PCA, helps us understand the effect of each transformation within a large language model (LLM) transformer. Since it's a linear transformation from 768D to 3D, it gives us insights into the function of various components of the LLM, such as positional encoding, layer normalization, attention, and multi-layer perceptron (MLP) modules, as well as the final logits, by observing their relative and evolving positions. The picture above is just one scenario demonstrating the library's capabilities. The same library can be used to investigate the effect of attention mechanisms, compare different similarity measures, and more. We will share our findings in future notebooks.

## Projector

The `Projector` is a research tool developed to help us understand the layers of LLM transformations at higher dimensions by visualizing their effect in projected 3D vector space. The focus is on designing an intuitive library objects such as _projector_, _projection_, and _trace_ to avoid misinterpreting these vectors when examining different LLM architectures.

### Main Objects
#### Projector
`Projector` is the main object that aggregates several helper objects to manipulate and project high-dimensional vectors into a 3D space. It can render a single vector, a list of vectors or a tensor in a single rendering. It also provides interactivity to help explore these vectors and their relationships. The main method for interactivity is `Projector.onClick()`.

The vector visualization uses a trace object created by the projector. This trace object retains essential information from the projector to project and render vectors in 3D space, such as _projection_, _similarity_, and _inference_.

`Projector` first sets up a stage using Plotly's FigureWidget, with the token embedding cloud serving as a background reference to help visualize embedding vectors in relation to the cloud.

Additionally, `Projector` establishes a `ColorShape` to manage the color and shape used in rendering. Different colors and shapes can represent different meanings during visualization.

The `Projector` can save some of its helper objects, which hold the state of the projection, into a cache file (e.g., PCA projection, current view, and camera position). This ensures consistent rendering and facilitates understanding the transitions during inference.

Below is a typical setup to use a projector to visualize a projection from a single 768D vector in 3D space. In this case, we use the origin. For more details, refer to this [notebook](https://github.com/vitalstarorg/projector/blob/main/nbs/projector.ipynb).

```python
model = GPT2Operator().name("gpt2").downloadModel()  # download gpt2 from huggingface
pj = Projector().name('projector').model(model)
pj.loadCache()                                          # load cached view and camera
pj.showEmbedding()                                      # show embedding cloud
trace = pj.newTrace().name('origin'). \
            label('origin').color('black'). \
            fromVectors(0).show()
```

#### Operator

`Operator` encapsulates GPT model loading and provides access to model parameters. It also offers a simplified implementation of the GPT-2 transformer. Since it acts as an adapter, behaving like a GPT model object, it is named `model` in our code. We use the name `Operator` to avoid confusion with Huggingface's `Model`.

Two implementations are available: `GPT2Operator` (a PyTorch implementation, which is the default) and `GPT2OperatorNP` (a Numpy implementation). These implementations are verified through unit tests to ensure correctness, which is crucial for manipulating high-level objects during experiments. This setup is used to examine various transformer schemes, such as redefining the architecture, changing similarity measures, or altering normalization techniques.

The `Operator` provides quick access to the nearest tokens of a high-dimensional vector, along with additional metrics to help track transformations during the embedding process.

#### Trace

`Trace` objects are created by the `Projector` and retain its state objects for projection and rendering. A trace can be created for either a single vector or multiple vectors represented as a list or tensor.

`Trace` objects work alongside the projector to encapsulate rendering logic and customization. Variants like `Line` allow different ways to connect vectors, such as lines or radial segments, with optional arrows.

`Trace` also calculates neighboring vectors based on the defined similarity and their corresponding angles.

Additionally, `Trace` implements the final linear transformation of GPT-2. This transformation is crucial for visualization because most transformed vectors are projected outside the visualization range (i.e., their norms are large). The final transformation brings them closer to the embedding cloud, enabling us to observe intermediate vectors. `Trace` allows us to use a uniform weight and bias instead of the original transformation, providing flexibility in visualizing different linear transforms.

```python
lnfw = pj.model().modelParams().getValue('ln_f.w')
lnfb = pj.model().modelParams().getValue('ln_f.b')
pj.wOffset(0.4); pj.bOffset(lnfb);     # using uniform weight with original bias
# or
pj.wOffset(lnfw); pj.bOffset(lnfb);    # using original weight and bias
```

#### Projection

`Projection` encapsulates the projection state and logic, making it a flexible component in our research. Currently, we use the first three principal components (PCA) as our final projection. We found this calculation to be highly sensitive to truncation errors, which can significantly affect the visualization. We plan to investigate its numerical stability further. Interestingly, a few dimensions in the embedding cloud contribute substantially to vector norms, which could, in theory, destabilize GPT-2 training and inference due to shared normalization procedures. We hope to finalize and share our findings soon.

Saving and loading the `Projection` object within the `Projector` helps maintain consistency across different experiments and inference runs. For larger GPT models, it also reduces computation time during projector setup.

#### Inference

`Inference` encapsulates the entire transformer process, from transforming a prompt to making final predictions. This encapsulation is crucial to safeguard the mathematical validity and accuracy when experimenting with different transformer architectures using the same model, such as skipping, repeating, or swapping layers.

Below we show both the original encoder-decoder transformer and decorder only GPT2 architectures for comparison. In GPT2, the positioning of the layer normalization is different.

<table style="border: none;">
  <tr>
    <td align="center">
      <b>Original Transformer Architecture</b>
      <img src="https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/ModalNet-21.png" width="300"><br>
    </td>
    <td align="center">
      <b>GPT2 Architecture</b>
      <img src="https://raw.githubusercontent.com/vitalstarorg/projector/refs/heads/main/nbs/gpt2-architecture.png" width="300"><br>
    </td>
  </tr>
</table>
The following code represents this transformer in Python:

```python
infer = self.model.inference().prompt("Alan Turing theorized that the")

infer.wte().wpe()
for layer in range(infer.nlayer()):
    infer.lnorm1(layer).attn(layer).sum()
    infer.lnorm2(layer).ffn(layer).sum()
infer.fnorm()                # final normalization
infer.logits()               # output 50257-dimensional logits
infer.argmax()               # next likely predicted token ids
infer.generation()           # next likely predicted tokens

# Alternatively, we can express the transformer using Smallscript
infer.ssrun("""self wte | wpe
       | lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum
       | layer: 1 | layer: 2 | layer: 3
       | layer: 4 | layer: 5 | layer: 6 | layer: 7
       | layer: 8 | layer: 9 | layer: 10 | layer: 11
       | fnorm | logits""")  # output 50257-dimensional logits
```

### Developer Note

#### Setup for Unit Tests and Notebook

To test out the library, follow these steps:

```bash
# Setup Python Env
conda create -y --name=prj "python=3.9"
conda activate prj
pip install -r requirements-darwin.txt    # for Apple Silicon
pip install -r requirements-linux.txt     # for Linux

# Run unit tests
export LLM_MODEL_PATH=$(pwd)/model        # directory for storing models
rm project*.zip                           # remove cached files
pytest --disable-warnings --tb=short tests
     # you should see 3 test suites
     # tests/test_00projector.py   ... passed
     # tests/test_01gpt2.py        ... passed
     # tests/test_02gpt2np.py      ... failed (as expected)

# Run the notebook to generate the image and plot
jupyter lab -y --NotebookApp.token='' --notebook-dir=. nbs/projector.ipynb
```

#### Setup Docker

A simple Dockerfile is provided to run the same setup using Docker.

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

We implemented the transformer using both NumPy and PyTorch. The NumPy implementation was inspired by [picoGPT](https://github.com/jaymody/picoGPT). We put a test harness on this implementation to ensure mathematical correctness. Based on this harness, we re-implemented it using PyTorch for compatibility with Huggingface. We hope to reuse this library with other Huggingface models.

To set up and use NumPy, download the GPT-2 checkpoint files and place them in the model directory. Then the third test should pass.

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

This project is ongoing and employs Test-Driven Development (TDD) to guide its development. TDD also serves as the main documentation, so all available features are illustrated as unit tests. We may develop additional notebooks and other resources to demonstrate each feature. More features and tests will be added as new visualizations are required for our research. Refer to this [notebook](https://github.com/vitalstarorg/projector/blob/main/nbs/projector.ipynb) to see how the image and plot were generated.

#### Use of Builder Pattern

A primary design pattern employed in most of the code is the builder pattern. Generally, methods named as nouns are property methods. When called without arguments, they act as getters, and when called with arguments, they act as setters and return the calling object. For example, every object has a `name()` method.

One reason for using this pattern is to make setting up complex objects (e.g., a projector with multiple helper objects) more intuitive. For example:

```python
pj = Projector()
projector.name('pj')              # set its name
projector.name()                  # get its name

# Alternatively, we can do this
pj = Projector().name('pj')
```

#### Use of Smallscript

Smallscript is a library that implements a variant of Smalltalk, used as a scripting language to intuitively orchestrate different objects. For example, the whole GPT2 transformer can be expressed as follows. With the help from underlying object design, it allows us to experiment different LLM architectures at ease without worrying of misinterpreting the vectors during visualization i.e. we could believe what we see.

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

- The pipe character works like a Linux pipe, passing the return value to the next method. Since we use a builder pattern, all methods return the same `@infer` object.
- Typically, the last method retrieves the final output rather than returning the original object (e.g., `self wte | wpe | layer:0 | … | fnorm | logits`).
- The line `| lnorm1: 0 | attn: 0 | sum | lnorm2: 0 | ffn: 0 | sum` is a detailed expansion of `layer: 0`, shown for illustration purposes.

