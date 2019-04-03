# PyTorch
By [Ziyu Bao](https://github.com/ZiyuBao), [Tian Tian](https://github.com/ttup7777), Yuanhao Xie, [Zhao Yin](https://github.com/zhaoyin666) from TU Delft.

![](https://i.imgur.com/ZcKU3XT.png)


# <a id="abstract">Abstract</a>
Here's the abstract. // TODO

# <a id="toc">Table of Contents</a>
1. [Introduction](#intro)
2. [Stakeholder Analysis](#stakeholder_ana)
3. [Context View](#context_view)
4. [Development View](#dev_view)
5. [Technical Debt](#td)
6. [Deployment View](#deploy_view)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Appendix](#appendix)

# <a id="intro">1. Introduction</a>
Here's the introduction for the report. // TODO

# <a id="stakeholder_ana">2. Stakeholder Analysis</a>
In this section, we will identify the stakeholders of pytorch, perform an PR analysis and look beyond Rozanski and Woods classification to identify any other relevant parties. What's more, we will present a Power vs. Interest grid.

## Stakeholders

According to stakeholders in Chapter 9 of Rozanski and Woods[2], we classified different types of stakeholders of pull-request and explain their contributions or influences.
As we know, a software is not only for using, but contain many steps to maintain it. For example, it need to be enhanced, maintain, test, etc. Thus, all those activities will involve stakeholders with different types. Here, we classify them into 11 different types that inspired by [2]. Table below will specifically introduced each type of stakeholders, include a short summary and specific description.

<table>
  <tr>
    <th>Type</th>
    <th>Stakeholders</th>
    <th>Summary</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="4">Developers</td>
    <td rowspan="4">Core developers(@fmassa,@apaszke, etc.) and contributors</td>
    <td rowspan="4">Developers are the largest group of all stakeholders. Anyone involved in the software development  can be seen as developer, including a lifecycle from software design, coding, testing to acceptance.</td>
    <tr>
       <td>@apaszke @fmassa are core developers. They are almost involved in every pull request. They are also responsible for merging the accpted pull requests. There are also other core developers in charge of the daily development of pytorch</td>
    </tr>
  </tr> 
  <tr>
  
  <tr>
    <td>Up to now, there are 994 contributors developed Pytorch on GitHub. Those contributors all made their contributions on system design or code writing and modification. </td>
  </tr>
  
  </tr>
  
   
  
  <tr>
    <td>Acquirers</td>
    <td>Senior sponsors, Strategy makers</td>
    <td>Acquirers oversee the procurement of the software and give the financial support. This kind of stakeholders usually is the most important part, because they can control the future roadmap of the software.</td>
    <td><a href="https://www.facebook.com">Facebook</a> is the main project sponsor for PyTorch development. In addition, <a href="https://aws.amazon.com/">AWS</a>, <a href="https://www.google.cn/">Google</a> and <a href="https://www.microsoft.com/">Microsoft</a> are deepening its investment in PyTorch 1.0 by providing stronger support for their cloud platform, products and services framework. </td>
  </tr>
  <tr>
    <td>Assessors</td>
    <td>Developers, internal assessment department or external legal entities</td>
    <td>Assessors supervise whether the development or test of the software meets the legal regulations. </td>
    <td>Developers, in addition to developing the system, also assess the systems's conformance to standards and legal regulations to make sure its well-development. There are also external legal departments that always check the system legal compliance.</td>
  </tr>
  <tr>
    <td>Communicators</td>
    <td>Communities, developers</td>
    <td></td>
    <td>Communicators explain the system to other stakeholders. <a href="https://www.GitHub.com">GitHub</a>, <a href="https://discuss.pytorch.org/">PyTorchDiscuss</a> and <a href="https://slack.com/">Slack</a> is three official communities for stakeholders to ask questions and answer the questions about PyTorch. In addition, PyTorch tutorials can be found in many webstations. Developers or teachers can explain everything about PyTorch.</td>
  </tr>
  <tr>
    <td>Maintainers(**to be edited**)</td>
    <td>@t-vi, @AlexanderRadionov, @lantiga and other developers</td>
    <td>@t-vi improved the functionality of PyTorch, for example, implicit Function Theorem and Implicit Functions in PyTorch. @AlexanderRadionov introduced simplistic implementation of batchnorm fusion for CNN in PyTorch. @lantiga is working on the Python module which is used for compiling static PyTorch graphs to C </td>
  </tr>
  <tr>
    <td>Production engineers</td>
    <td>@Yinghai Lu and @Duc Ngo and other engineers</td>
    <td>@Yinghai Lu and @Duc Ngo are software engineers of PyTorch. They takes charge of tensors and dynamic neural networks in python with strong GPU acceleration.  </td>
  </tr>
  <tr>
    <td>Suppliers</td>
    <td>Ecosystem members, GitHub</td>
    <td>PyTorch tap into a rich ecosystem of tools, libraries, and more to support, accelerate, and explore AI development. <a href="https://allennlp.org/">AllenNLP</a> is a open-source research library built on PyTorch for designing and evaluating deep learning for NLP. <a href="https://github.com/pytorch/elf">ELF</a> is a platform for game research that allows developers to test and train their algorithms in various game environments. <a href="https://docs.fast.ai/">fastai</a>, <a href="https://github.com/zalandoresearch/flair">Flair</a>, <a href="https://github.com/pytorch/glow">Glow</a>, <a href="https://gpytorch.ai/">GPyTorch</a>, <a href="https://github.com/horovod/horovod">Horovod</a>, <a href="https://github.com/pytorch/ignite">Ignite</a>, <a href="http://parl.ai/">ParlAi</a>, <a href="http://pyro.ai/">PyroAi</a>, <a href="http://tensorly.org/stable/home.html">TensorLy</a> and <a href="https://github.com/pytorch/translate">Translate</a> are all suppliers provide services for PyTorch. What's more, <a href="https://www.GitHub.com">GitHub</a> is the main supplier for developing.</td>
  </tr>
  <tr>
    <td>Support staff</td>
    <td>Developers, Teachers, Customer service</td>
    <td>Support staff include help desk, technical support, customer service departments and etc. Developers have the right to give technical support on commnuities. Teachers provide the service to users for running the system. Customer service of PyTorch provide service for solving the problems from users.</td>
  </tr>
  <tr>
    <td>System administrators</td>
    <td>Main engineers @Yinghai Lu and @Duc Ngo and other engineers, core developers</td>
    <td>The main engineers take charge of the operation of PyTorch. Core developers control the evolution and development of different projects on PyTorch.</td>
  </tr>
  <tr>
    <td>Testers</td>
    <td>Developers @MlWoo</td>
    <td>@MlWoo used PyTorch to Benchmark the CNN part of projects on different mainstreaming CPU platform</td>
  </tr>
    <tr>
    <td>Users</td>
    <td>All developers and some organizations using PyTorch </td>
    <td>Alibaba (Arena): Alibaba cloud's deep learning solution-Arena supports the PyTorch frameworks. Stanford University is using Pytorch on their researches of new algorithmic approaches. UDACITY is using PyTorch to educate the next wave of AI innovators. Salesforce is using PyTorch to push the state of the art in NLP and Multi-task learning.</td>
  </tr>
</table>

## PR Analysis

An codification process is in the appendix.

The author of the pull request creates a new functionality and shows the usage to whom will concern. After the showcase, two main members of PyTorch have a lot of reviews, suggestions, technical questions and vital decisions in the conversations of the pull request which are basically about, in the code level, improvements, small bugs, big issues, abstract representations (e.g., one attribute should morally be included in another class) and connections with other parts of PyTorch. This issue-solution pattern as an atom or a routine circle is repeated many times before a final approvement is reached to ensure the whole process trackable and efficient. Experienced reviewers largely influence the enhancement and complement of the new functionality or the code to be specific. 

Looking over all the conversations during which the code base of the original pull request get improved and reconstructed, I notice that the members of PyTorch are quite expected to see this new change thus the whole process is going quite fast and active. Meanwhile, the author is able to fully understand the suggestions, follows them strictly and comments every subtle change to make things clear, which saves quite a lot of time. With efficient and extensive cooperative efforts of the author and reviewers, the pull request becomes good enough and is naturally ready to merge.

After this kind of laborious analysis, I am quite sure what it is about and what is going on within this specific pull request. Besides, whether a pull request would be accepted depends a lot on the main developers' perspective and its external connection with the interest of the open source project.



## Going beyond Rozanski and Woods classification:

#### Competitor: 
TensorFlow is based on Theano and developed by Google. Compare to Pytorch, Tensorflow has a larger community and it creates a static graph instead of a dynamic graph. 

#### Founders: 
PyTorch is based on Torch and developed by Facebook. The original authors are Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan.


#### Integrators/reviewers in PyTorch:
The core developers such as @apaszke @fmassa are integrators. They're architects of PyTorch and almost involve in every pull request. They're responsible for reviewing the pull requests, decide to accept a pull request and merge the branches or not.

#### Contact Persons
The project has many developers who are the main members of the team. If you want to contact them, you can go to their github pages and find their email addresses if they have provided them, else you can just leave a comment in the discussion they involve. Here is a non-exaustive list: Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Kopf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

## Power vs Interest Grid:
The Figure 1 shows the Power & Interest grid.

![](https://i.imgur.com/FZS6FyO.png)
Figure 1. Power & Interest grid


**High power and high interest**: Core developers who have the right to accept the modification of the project or not as well as administrators and production engineers have the maximum power. Testers, maintainers, and assessors who take responsibility for testing the system, managing the evolution, and overseeing the system's conformance to standard respectively. Therefore, they have a slightly lower power compared with core developers. Testers, maintainers assessors have the same level interest as core developers. Founders belong to this category, however, their power and interest are lower than the others in this category. Acquires who oversee the procurement of the system and give the financial support. They usually are the most important part, as they can control the future roadmap of the software. Therefore, they had relatively high interest and high power.

**High power and low interest**: Since Facebook is the first developer of Pytorch, it has the highest level of power as PyTorch which is developed and mainly sponsored by Facebook.
**Low power and high interest**: Apparently, User is a typical example. Users implement PyTorch. Therefore, they must have high interest. They make use of PyTorch, but do not have that high power compared with other stakeholders.
**Low power and low interest**: Suppliers such as have quite few power and are not that interested in PyTorch. For example, Github provides the platform for developers to work together for PyTorch.
(to be continued)
**Communicators and Support Staff**: Communicators mainly focus on creating documentation and training material to explain the PyTorch. In general, they do not have other power. Support Staff helps the user to run the system. They have no decision power. Therefore, both of these two types of stakeholders are mildly interested in PyTorch with relatively low power. 

## StakeHolder analysis conclusion
We can see that the this project involve many types of stakeholders. There are also types of stakeholders beyond Rozanski and Woods classification. They have different interest and power to the project.

# <a id="context_view">Context view</a>
The context view of PyTorch defines the relationships, dependencies, and interactions between PyTorch and its environment as show in Figure 2.

![context_view](https://i.imgur.com/mkVgvLe.png)

## System Scope and Responsibilities
PyTorch is created and developed primarily by Facebook's artificial intelligence group as an open-source deep learning library platform that provides a seamless path from research prototyping to production deployment. Everyone could use it to build its own customized neural networks or perform fast matrix operations on GPUs using the torch component. PyTorch is a Python package that provides two high-level features [[1]](#pytorch_github):
1. Tensor computation (like NumPy) with strong GPU acceleration
2. Deep neural networks built on a tape-based autograd system

## External Entities
Communication: Communications are mainly done in <a href="https://www.GitHub.com">GitHub</a>, <a href="https://discuss.pytorch.org/">PyTorchDiscuss</a> and <a href="https://slack.com/">Slack</a>. In addition, many websites provide tutorials of Pytorch. Communication can also be done among developers and via teachers.

Storage: <a href="https://aws.amazon.com/">AWS</a>, <a href="https://www.google.cn/">Google</a> and <a href="https://www.microsoft.com/">Microsoft</a> all have provided support of their cloud platforms for data storage of PyTorch.

Tools: [Horovod](https://github.com/uber/horovod) is a distributed training framework that can be used by PyTorch. [Pytorch Geometry](https://github.com/arraiy/torchgeometry) is a library of PyTorch for geometric computer vision. [TensorBoardX](https://github.com/lanpa/tensorboardX) is a visulization tool that can log events happening e.g. during training of a neural network. [Translate](https://github.com/lanpa/tensorboardX) extends PyTorch functionality to train for machine translation models.

Users: PyTorch is used from industry to acedemia. When it is used in industry, it is used as part of the core business of companies, like Alibaba or Salesforce, to support deep learning framework. When it is used in academia, PyTorch can support researches of new algorithmic approaches, like in Stanford University. These users in turn also provide feedback or contribute in the project on Github, which makes them contributors as well.

# <a id="dev_view">Development View</a>
The development view of PyTorch describes its code structure and dependencies, build and configuration of deliverables, system-wide design constrains and system-wide standards to ensure technical integrity [[2]](#book). We start off by introducing the module structure model of the architectural components from low to high API levels and the overview of the components of PyTorch as a Python library at user level. Second, we explain the common design models in PyTorch that maximizes commonality across modules and functionalities in the process of development. Finally, we end up this chapter by walking through the codeline model that makes up the code of PyTorch.

## Module Structure Model
The main structure of PyTorch in a architectural view is shown in Figure [3](#module_model).

![#module_model](https://i.imgur.com/iGWbOXL.png)

**Figure 3.** Pytorch Architecture. Inspired by [[3]](#learning_pytorch_book).

The top-level Python library of PyTorch (please refer to the following section) exposes easy-to-understand API for users to quickly perform operations on tersors, build and train a deep neural network. This library provides interface but doesn't really execute the computations. Instead, it delivers this job down to its efficient computation engines written in C++. 

These engines on the middle-level of module structure consist of `autograd` for computing DCG and providing auto-differentiation, `JIT` (just-in-time) compiler for optimizing computation steps that are traced, `ATen` as a C++ tensor library that wraps low-level C library for PyTorch (no autograd support). 

The low-level C or CUDA library does almost all the intensive computations assigned by upper-level API. These libraries provide efficient data structures, the tensors (a.k.a. multi-dimensional arrays), for CPU and GPU (TH and THC, respectively), as well as stateless functions that implement neural network operations and kernels (THNN and THCUNN) or wrap optimized libraries such as NVIDIA’s cuDNN [[3]](#learning_pytorch_book). 

ATen wraps those low-level libraries in C++ and then exposes to the high-level Python API. Similarly, the neural network function libraries (low-level) are automatically wrapped towards the engine and Python API (see the two curved arrows). In other words, the low-level libraries can be utilized not only by its standard wrapper ATen but also top-level Python APIs and mid-level engines. This kind of connection keeps the code loosely coupled, decreasing the overall complexity of the system and encouraging further development [[3]](#learning_pytorch_book).

## Component Overview
PyTorch as a libary consists of the following components (also see Figure 3 for the connection with the module structure):
- **torch:** a Tensor library like NumPy, with strong GPU support [[1]](#pytorch_github). It contains data structures for multi-dimensional tensors , defines many mathematical operations based on these tensors, and provides many utilities for efficient serialization of Tensors and arbitrary types, and other useful utilities. Different from its analogue NumPy, all the data structures and tensor operations can be seamlessly performed from CPU to GPU which would accelerate the computation by a huge amount.
- **torch.autograd:** a tape-based automatic differentiation library that supports all differentiable Tensor operations in torch [[1]](#pytorch_github). This functionality greatly differs PyTorch from other machine learning or deep learning frameworks like TensorFlow [[4]](#tensorflow), Caffe [[5]](#caffe) and CNTK [[6]](#cntk) which require users to restart from scratch at beginning in order to change some minor behaviors of the neural network once created. While in PyTorch, a  technique called reverse-mode auto-differentiation is adopted to facilitate differentiation process so that the computation graph is computed in the fly which leaves users more time to implement their ideas.
- **torch.nn:** a neural networks library deeply integrated with autograd designed for maximum flexibility [[1]](#pytorch_github). This component or module in PyTorch provides a high level for us to build and train a deep neural network easily without pain. It contains many types of neural network layers like convolutional layers, recurrent layers, padding layers and normalization layers, and also a large amount of utilities facilitating functional operations on the network like loss functions and distance functions.
- **torch.multiprocessing:** Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training [[1]](#pytorch_github). This component wraps native Python multiprocessing module using shared memory to provide shared views on the same data in different processes. Its 100% API compatibility with the original module make it possible to have all the tensors sent through the queues or shared via other mechanisms, moved to shared memory.
- **torch.utils:** DataLoader, Trainer and other utility functions for convenience [[1]](#pytorch_github). It consists of five submodules - `torch.utils.bottleneck` for debugging bottlenecks in the program, `torch.utils.checkpoint` for checkpointing a model or part of the model, `torch.utils.cpp_extension` for creating C++ extension on the program and `torch.utils.data` for data generating and loading and other fancy operations on the data, and `torch.utils.dlpack` for conversion between a tensor and a decoded DLPack (an open in-memory tensor structure to for sharing tensor among frameworks [[7]](#dlpack)).

## Common Design Model
This section uses a common design model to analyze how PyTorch tries to achieve its developmental approach.
### Common processing
Common processing identifies tasks that benefit greatly from using a standard approach across all system elements.

**Use of third-party libraries.** PyTorch makes use of third-party libraries for build, doc generation, high-level operations and etc. They are:
- **python-future:** a missing compatibility layer between Python 2 and Python 3 [[8]](#python-future).
- **numpy:** a fundamental package needed for scientific computing with Python [[9]](#numpy).
- **pyyaml:** a next generation YAML parser and emitter for Python [[10]](#pyyaml).
- **setuptools:** a fully-featured, actively-maintained, and stable library designed to facilitate packaging Python projects [[11]](#setuptools).
- **six:** a Python 2 and 3 compatibility library that provides utility functions for smoothing over the differences between the Python versions with the goal of writing Python code that is compatible on both Python versions [[12]](#six).
- **typing:** a Python library support for type hints.
- **matplotlib:** a Python 2D plotting library which produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms [[13]](#matplotlib).
- **Sphinx:** a tool that makes it easy to create intelligent and beautiful documentation for Python projects (or other documents consisting of multiple reStructuredText sources) [[14]](#sphinx).
- **sphinxcontrib-katex:** a Sphinx extension for rendering math in HTML pages.
- **breathe:** an extension to reStructuredText and Sphinx to be able to read and render the Doxygen xml output [[15]](#breathe).
- **exhale:** an automatic C++ library API documentation generator using Doxygen, Sphinx, and Breathe.

**Documentation management.** Well documented code benefits code readability, practical implementation and issue tracing. PyTorch uses Google style for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups. For C++ documentation it uses Doxygen and then convert it to Sphinx via Breathe and Exhale [[1]](#pytorch_github).


### Standard design approaches
Standard design approaches identifies how to deal with situations where implementations of a certain aspect of an element will have a system-wide impact.

**API design standardization.** (resource from [[22]](#blog)) The essential unit of PyTorch is the Tensor. As the Tensor is used in Python for users but implemented in C, it needs API design to wrap it up so that user can interact with it from the Python shell. PyTorch first extends the Python interpreter by CPython’s framework to define a Tensor type that can be manipulated from Python. The formula for defining a new type is as follows: 1) create a struct that defines what the new object will contain and 2) define the type object for the type. Then PyTorch wraps the C libraries that actually define the Tensor’s properties and methods by CPython backend’s conventions. The `Tensor.cpp` file defines a “generic” Tensor and PyTorch uses it to generate Python objects for all the permutations of types (`float`, `integer`, `double` and etc.). PyTorch takes the custom YAML-formatted code and generates source code for each method by processing it through a series of steps using a number of plugins. Finally to generate a workable application, PyTorch takes a bunch of source/header files, libraries, and compilation directives to build an extension using Setuptools.


**Contributing standardization.** As PyTorch is an open source which means that everyone is free to contribute to the repository. In order to keep maintainability, reliability, and technical cohesion of the system, the PyTorch Governance (consisting of code maintainers, core developers and some other contributors) composed a contributing tutorial to standardize the design process. The tutorial provides useful guidelines for coding, parameter configuration, testing, writing documentation and all other tips and rules for qualified contribution.

## Codeline Model
### Source Code Structure
PyTorch has its directory structure of code to make it easy for developers locate the code they want to change. We show its main directory below while the full directory could be seen on [here](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#codebase-structure).
```
Pytorch Code Structure
├── c10/ - Core library files that work everywhere.
├── aten/ - C++ tensor library for PyTorch (no autograd support).
|   └── src/
|       ├── TH/ THC/ THNN/ THCUNN/ - Legacy library code from the original Torch.
|       └── ATen/
|           ├── core/ - Core functionality of ATen.
|           └── native/ - Modern implementations of operators.
|               ├── cpu/ - CPU implementations (compiled with processor-specific 
|               |   instructions) of operators.
|               ├── cuda/ - CUDA implementations of operators.
|               ├── sparse/ - CPU and CUDA implementations of COO sparse tensor 
|               |   operations.
|               └── mkl/ mkldnn/ miopen/ cudnn/ - Implementations of operators 
|                   which simply bind to some backend library.
├── torch/ - The actual PyTorch library.
|   └── csrc/ - C++ files composing the PyTorch library.
|       ├── jit/ - Compiler and frontend for TorchScript JIT frontend.
|       ├── autograd/ - Implementation of reverse-mode automatic differentiation.
|       ├── api/ - The PyTorch C++ frontend.
|       └── distributed/ - Distributed training support for PyTorch.
├── tools/ - Code generation scripts for the PyTorch library.
├── test/ - Python unit tests for PyTorch Python frontend.
|   ├── test_torch.py - Basic tests for PyTorch functionality.
|   ├── test_autograd.py - Tests for non-NN automatic differentiation support.
|   ├── test_nn.py - Tests for NN operators and their automatic differentiation.
|   ├── test_jit.py - Tests for the JIT compiler and TorchScript.
|   ├── cpp/ - C++ unit tests for PyTorch C++ frontend.
|   ├── expect/ - Automatically generated "expect" files which are used to compare
|   |   against expected output.
|   └── onnx/ - Tests for ONNX export functionality, using both PyTorch and Caffe2.
└── caffe2/ - The Caffe2 library.
    ├── core - Core files of Caffe2, e.g., tensor, workspace, blobs, etc.
    ├── operators - Operators of Caffe2.
    └── python - Python bindings to Caffe2.
```

# <a id="td">Technical debt</a>
Technical debt reflects the implicit the cost of choosing a simple solution instead of using a better method that takes longer time. Identifying and repaiding the technical debt is important as it can accumulate interest the difficulies on implementing changes later on [[16]](#tb).  It may also cause problem later on if it is not fixed.
# rewrite thisparagraph
We anlyzed the Keywords/tags(TODO, FIXME and XXX), the files that are often changed（hotspots), Complexity trends, which files are frequently changed together(temporal coupling), and duplications and bugs. Following are tools used to identify and anylsis the technical debts:

| Name of tool | Purpose |
| -------- | -------- | 
| CodeScene | Identifying hotspot, Complexity trends and code smells|
| Github-grep| Finding keywords like `TODO`, `FIXME` and `XXX`|
| SonarQube     | Identifying duplications and finding bugs    |

The reasons for these decisions are:
`CodeScene` is a easily operated tool to analyze a project. It also has several intuitively functions to show a software's evolution or defects. For example, the hotspot map can be shown by using CodeScene and it can intuitively show the code structure and point out some parts that really matter. In that case, we choose three functions to analyze technical debt by identifying hotspot, complexity trends and code smells. We will explain why we choose these functions to analyze.
`Github-grep` is a tool of Git to search everything you want in the code. For example, we often need to find some functions where it defined or called. In this case, we used this tool to find all three keywords: `TODO`, `FIXME` and `XXX`.
`SonarQube`is a powerful code quality management tool. It can detect the code quality in the following aspects: bugs, vulnerabilities, code smells, code coverage and duplications. Since some of those aspects we have used `codescene` to analyze, we mainly explained duplications and bugs by using SonarQube. 


### Keywords/tags
We used `Github-grep` to analyze the Keywords/tags  "TODO"s, "FIXME"s and "XXX"s. "TODO" denotes the task which should be done in future work, "FIXME" tag means that there is the potential problem in the code should be fixed, while "XXX" indicates a bad structure of the code[20]. All these keywords can indicate that there are technical debts which the developer noticed but did not fix. By using`git grep`, 1098 "TODO"s, 118 "FIXME"s, and 119 "XXX"s are identified. Some of the comments mention who will fix it, none of them indicates when they will be fixed. Following are the examples:

* ```caffe2/python/caffe_translator.py:# TODO(jiayq): find a protobuf that uses this and verify.```
* ```caffe2/operators/lp_pool_op.cc:// TODO: reduce the apparent redundancy of all the code below. ```
* ```tools/autograd/gen_variable_type.py:    # FIXME: figure out a better way when we support sparse tensors in jit```
* ```caffe2/operators/conv_op_cudnn.cc:// TODO(Yangqing): a lot of the function contents are very similar. Consider```
* ```torch/csrc/autograd/functions/accumulate_grad.cpp:  // XXX: this method is not thread-safe!```

"TODO"s are almost everywhere in pytorch code. There are 432 "TODO"s in caffe2. 72 out of 119 "XXX" tag are in the main pytoch component-torch. "FIXME" are mainly in aten, caffe2, test, torch. Along with the accumulation of tags, some of them may be forgotten, or become bugs[17]. These keywords may clutter the code and have negative effects on code comprehension[17]. This large amount of technical debt may make developers understand the code incorrectly. 

To solve this. It is necessary to track the progress of fixing the issues which can be done by appointing an issue tracker and ask developers to add more details to the keywords/tags such as the name who is responsible for the issue, deadline and bug ID.


### Hotspot 
# explain the hotspot
We used `Codescene` to analyze the hotspot.

![](https://i.imgur.com/2jnlywS.png)
**Figure 4.** Hotspot

The result is in Figure 4 which indicates the pytorch directory structure. Each blue circle represents a package in the code. Inside those packages, there is a code file. The red color indicates re-designs priority. The darker red hotspots mean the higher re-designs priority. Fix those red hotspots will lead to a large improvement of the whole project.

### Complexity trends[23]
This section describes the complexity trends the hotspots of pytorch. The reason why we should care about complexity trends is that it coresponds the technical debt interest rate. Complexity trends denote how do the hotspots get more complicated over time. The hotspot with fast-growing complexity should be priortised for refactoring to reduces the risk of technical debt. Otherwise as soon as the complexity become unmanageable, the code modifications become really difficult.

By fetching the history of the hotspots and calculating the complexity of those historic versions, the complexity trends can be obtained. We used sofeware`Codescene` to get the Complexity trends of the hotspot. 

Figure 5 indicates a high technical debt interest rate. It shows that the complexity trend of hotspot `pytorch/test/test_jit.py` grows rapidly since June 2018. This accumulating complexity is a sign which means that the hotspot needs refactoring[21] to reduce the risk of technical debt. 

![](https://i.imgur.com/6mIn9eq.png)
**Figure 5.** Complexity trends of `pytorch/test/test_jit.py`

A good example of the reducing the complexity trend is shown in Figure 6. The complexity trend of the hotspot `pytorch/test/test_nn.py` increases continuously before December 2018. There is a downward slope in this complexity trend which occurs at the end of November. This is due the commit [#814b571](https://github.com/pytorch/pytorch/commit/814b5715ba42449f2231a40cdd93273ec6f7e76c#diff-d89baec73022f5f511c5beb5ce6498df) which moves the `new_module_tests` from `test_nn.py` to `common_nn.py`. The refactoring of this hotspot leads to a decrease complexity.

![](https://i.imgur.com/V578i5d.png)
**Figure 6.** Complexity trends of `pytorch/test/test_nn.py`

To avoid the technical debt, the complexity regularly should be tracked , and the hotspot should be appropriatly redesigned and refactored.

### Temporal coupling

This section describes the temporal coupling  of PyTorch. The reason why we should care about temporal coupling is that it temporal coupling leads to technical debts. The large codebases with multiple developers may leads technical debt. Even though at the beging of development, people seperate the modules on purpose, the couplings may still forms with the development process. The couplings among different modules leads to rigid of system design and the difficulty of extending the features[24].

Temporal coupling indicates how two or more modules change together over time. It can be obtained by calculating how often a module changes together with other modules. We used `Codescene` to visulize the temporal coupling.

In Figure 7, the lines that the modules which are modified together. The thicker lines, the stronger of temporal coupling. 

![](https://i.imgur.com/PvngF9g.png)
**Figure 7.** Temporal coupling

The coupling degrees of the pairs shown in the table in the right of the Figure are quite strong. For example, the coupling degrees of the third pairs are 95%. It means that 95% chance of changing one file results in a change in another file. The temporal coupling sometimes suggests a refactoring. As you can see the coupling degrees of the first pairs are 100%. Figure 8 shows the code of `add_cpu.cc` and `mul_cpu.cc`. These two files are more like copy-paste work.

![](https://i.imgur.com/aHnzbOE.png)
**Figure 7.** Code of `add_cpu.cc` and `mul_cpu.cc`


To avoid this technique debt. The temporal coupling should be broken by extracting the common part into a module. In this way, the code will be easier to maintain.


### Duplications and existing bugs
To identify duplications and existing bugs, we utilized `SonarQube` as our data quality tool. 

* Duplicates: In the detection of `SonarQube`, there are overall 3.9% duplicates, which include 9986 duplicated lines, 601 duplicated blocks and 141 duplicated files. Among all duplicated files, the test documents account more proportions. However, this phenomena cannot be avoided for testing frameworks.

![](https://i.imgur.com/AdXQpqI.png)
**Figure 7**. duplicates in documents

* Bugs: `SonarQube` detected 49 bugs in current release. According to the **Figure 8**, there are several bugs were caused by coding style. Those incorrect usages including expression of operator, some useless self-assignment may cause some effects in future development and need to be fixed.
 
* test_jit.py: As shown in **Figure 8**, the first document contains 10 bugs which is test_jit.py. This document also has the most duplicates. According to SOLID violations, this document violates the **The Single Responsibility Principle**. The test_jit.py has 13414 code lines and contains lots of methods in big classes which may satisfy the same responsibilities.

![](https://i.imgur.com/QN5AYZO.png)
**Figure 8** Bugs in documents


## The evolution of Technical Debt
Besides the technical debt that is currently present in Pytorch, we also looked at the evolution of this software. Pytorch was released in 2016 and developed by Facebook's artificial-intelligence research group. Originally considered Pytorch as a Python package for GPU-accelerated deep neural network programming, it could complement or partially replace existing Python packages and statistics packages such as NumPy. 

There are two variants of PyTorch. Originally, Pytorch was developed as a Python wrapper for the LuaJIT-based Torch framework[[19]](#1). Then, Pytorch became a completely new development. Unlike the old variant, PyTorch no longer uses the Lua language and LuaJIT. Instead, it's a native Python package.

PyTorch redesigned and implemented Torch in Python while sharing the same core C library in the backend code. Torch was originally implemented in C, with a wrapper for the Lua scripting language, but PyTorch wraps the core Torch binaries in Python and provides GPU acceleration for many functions[[18]](#infoworld). PyTorch developers have tweaked their backend code to run Python efficiently. They also retained GPU-based hardware acceleration and the extensibility that made Lua-based Torch popular among researchers.

![](https://i.imgur.com/Zp6DKp8.png)

**Figure 9.** Contributions to master, excluding merge commits 

Since Pytorch was the next generation products of torch, it was developed officially since 2016. Pytorch was mainly developed on Github to control its different releases. As shown in **Figure 9**, the contributions increased dramatically since 2016. According to the releases showing on the Github, Pytorch has released several versions which from v0.1.1 to v1.0.1. Among those releases, many versions contained the contribution of fixing previous works. Those works can be seen as technical debts. 



| Version | Main works | Time |
| -------- | -------- | -------- |
| v0.1.1   | bumping to alpha-1     |  Aug 24, 2016     |
| v0.1.2   | bumping to alpha-2     |  Sep 1, 2016     |
| v0.1.3   |docstrings for container and batchnorm | Sep 16, 2016|
| v0.1.4   | python 2.7 fixes | Oct 3, 2016|
| v0.1.5   | tensor docs      |Nov 8, 2016|
| v0.1.6   | fix docs for torch.nn.functional.conv1d (#536)|Jan 21, 2017
| v0.1.7   | add cc 3.0 to nccl (#594)|Jan 26, 2017|
| v0.1.8   | std::move fixes |Feb 3, 2017|
| v0.1.9   | Fix Engine::compute_dependencies|Feb 17, 2017|
| v0.1.10  |import numpy before setting dlopen flags (#928) |Mar 5, 2017|
| v0.1.11  |check for nvidia driver's sufficiency before checking for number of CUDA devices (#1156)|Mar 31, 2017|
| v0.1.12 |version bump |May 1, 2017|
| v0.2.0  |fix static linkage and make THD statically linked| Aug 28, 2017|
| v0.3.0  |Backport transposes optimization to v0.3.0 (#3994) | Dec 4, 2017|
| v0.3.1  |Scopes 0.3.1 backport (#5153)| Feb 9, 2018|
| v0.4.0  |move to eigenteam github for eigen submodule| May 30, 2018 |
| v0.4.1  |fix lint | Jul 26, 2018|
| v1.0rc0 |Back out "Revert D10123245: Back out "codemod cuda_gpu_id to device_id"" (#12232)|Oct 2, 2018|
| v1.0rc1 | Same as v1.0rc1, add more notes|Oct 2, 2018|
| v1.0.0  | add fix for CUDA 10 |Dec 7, 2018|
| v1.0.1  | Remove unnecessary typing dependency. (#16776)| Feb 7, 2019|

# <a id="deploy_view">Deployment view</a>

Deployment view determines the related environment to run the system, such as the hardware support or hosting environment. Pytorch has clearly introduced its deployment for all interested stakeholders. It belongs to the situation that the system can be deployed into a different environment and the characteristic need to be clearly illustrated. We will discuss the deployment view in the following passages and Figure 10 is the overall figure deployment view of Pytorch.

![](https://i.imgur.com/jYN0Bj3.png)

**Figure 10**. Figure deployment view

## Third Party Library
Pytorch used different libraries to develop its system. Those third-party libraries have been specifically introduced in section development view. Those third parties includes `Numpy`, `Sphinx`, `pyyaml`, etc, form a third-party system requirements for running Pytorch and support the daily operating of Pytorch.

## Runtime platforms
Runtime platforms are the essential parts in deployment view. In Pytorch, C++ and Python are thought to be the two runtime platforms for running Pytorch. Most base codes of Pytorch are written in C++ and the basic functions of Pytorch are written in Python.

## Operating systems
PyTorch can be installed and used in many types of operating systems. 
1. **Linux**: PyTorch 1.0 supports various Linux distributions that use glibc >= v2.17:
    - Arch Linux, minimum version 2012-07-15
    - CentOS, minimum version 7.3-1611Debian, minimum version 8.0
    - Fedora, minimum version 24
    - Mint, minimum version 14
    - OpenSUSE, minimum version 42.1
    - PCLinuxOS, minimum version 2014.7
    - Slackware, minimum version 14.2
    - Ubuntu, minimum version 13.04
2. **MacOS**: PyTorch is supported on macOS 10.10 (Yosemite) or above. As MacOS does not has CUDA support building from binaries, it will not benefit from GPU accelaration. Installation from source is required if users want to harness the full power of PyTorch’s CUDA support.
3. **Windows**: PyTorch is supported on the following Windows distributions:
    - Windows 7 and greater; Windows 10 or greater recommended.
    - Windows Server 2008 r2 and greater.
    

## PyTorch on cloud
Cloud platforms provide powerful hardware and infrastructure for training and testing the PyTorch models. PyTorch offers following cloud installation options:
    


| Could Platform | Details | 
| -------- | -------- | 
| Microsoft Azure    | On the Azure Data Science Virtual Machine for CPU only or accessing at most four GPUs     |
| Google Cloud Platform    | On machine instances with access to one, four, or eight  GPU in specific regions or on containerized GPU-backed Jupyter notebooks   |
| Amazon Cloud Platform    |  AWS Deep Learning Amazon Machine Image with optional NVIDIA GPU support   |
| IBM Cloud Kubernetes cluster     | On kubernetes clusters on IBM Cloud    |
| IBM Cloud data science and data management     | Python environment with Jupyter and Spark   |

# <a id="conclusion">Conclusion</a>
Here's the conclusion of the report. // TODO

# <a id="references">References</a>
1. <a name="pytorch_github">PyTorch github website,</a> https://github.com/pytorch/pytorch.
2. <a name="book">William Del Ra, III. 2012. Software systems architecture: second edition by Nick Rozanski and Eoin Woods. SIGSOFT Softw. Eng. Notes 37, 2 (April 2012), 36-36. DOI: </a>https://doi.org/10.1145/2108144.2108171
3. <a name="learning_pytorch_book">Deep Learning with PyTorch by Eli Stevens Luca Antiga. MEAP Publication.</a> https://livebook.manning.com/#!/book/deep-learning-with-pytorch/welcome/v-7/
4. <a name="tensorflow">Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from </a>[tensorflow.org](https://www.tensorflow.org)
5. <a name="caffe">Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick,
S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for
fast feature embedding. <i>arXiv:1408.5093>arXiv:1408.5093</i>, 2014</a>
6. <a name="cntk">Seide, Frank & Agarwal, Amit. (2016). CNTK: Microsoft's Open-Source Deep-Learning Toolkit. 2135-2135. 10.1145/2939672.2945397. </a>
7. <a name="dlpack">DLPack github website, </a>https://github.com/dmlc/dlpack
8. <a name="python-future">Python-future github website. </a>https://github.com/PythonCharmers/python-future
9. <a name="numpy">Numpy github website. </a>https://github.com/numpy/numpy
10. <a name="pyyaml">Pyyaml github website. </a>https://github.com/yaml/pyyaml
11. <a name="setuptools">Setuptools documentation. </a>https://setuptools.readthedocs.io/en/latest/
12. <a name="six">Six github website. </a>https://github.com/benjaminp/six
13. <a name="matplotlib">Matplotlib github website. </a>https://github.com/matplotlib/matplotlib
14. <a name="sphinx">Sphinx github website. </a>https://github.com/sphinx-doc/sphinx
15. <a name="breathe">Breathe github website. </a>https://github.com/michaeljones/breathe
16. "Definition of the term "Technical Debt" (plus, some background information and an "explanation")". Techopedia. Retrieved August 11, 2016.
</a>
17. Margaret-Anne Storey, Jody Ryall, R. Ian Bull, Del Myers, Janice Singer, “TODO or To Bug:Exploring How Task Annotations Play a Role in the Work Practices of Software Developers”, InProceedings of the 30th Int’l. Conf. Software Engineering (ICSE ’08), pp. 251-260, 2008.
18. <a name="inforworld">Facebook brings GPU-powered machine learning to Python </a>https://www.infoworld.com/article/3159120/facebook-brings-gpu-powered-machine-learning-to-python.html
19. <a name="1">Getting started with Pytorch. </a>https://www.ibm.com/developerworks/cn/cognitive/library/cc-get-started-pytorch/index.html 
20. </a> XUAN, Jifeng; HU, Yan; JIANG, He. Debt-prone bugs: technical debt in software maintenance. arXiv preprint arXiv:1704.04766, 2017.
21. </a> Codescene Guides-COMPLEXITY TRENDS https://codescene.io/docs/guides/technical/complexity-trends.html
22. <a name="blog">PyTorch blog.</a> https://pytorch.org/blog/a-tour-of-pytorch-internals-1/
23. <a name="1">A Crystal Ball to Prioritise Technical Debt in Monoliths or Microservices: Adam Tornhill's Thoughts. </a>https://www.infoq.com/news/2017/04/tornhill-prioritise-tech-debt 
24. <a name="blog">Coupling analysis
</a> https://github.com/smontanari/code-forensics/wiki/Coupling-analysis






# <a id="appendix">Appendix</a>
## PR Analysis
### Pull requests which are accepted：
| Pull request | Lifetime| Components it touches | PR content and Related issues| Deprecate another pull request or not|
| -------- | -------- | -------- | --------| --------|
| Implement adaptive softmax  awaiting response [#5287](https://github.com/pytorch/pytorch/pull/5287) by @elanmart  | After v0.4.0 |`torch.nn`(neural networks library which is integrated with autograd designed)| This pr implements "Efficient softmax approximation for GPUs" which discussed in another pr #4659| No| Implement adaptive softmax awaiting response [#5287](https://github.com/pytorch/pytorch/pull/5287) by @elanmart  | After v0.4.0 |`torch.nn`(a neural networks library deeply integrated with autograd designed for maximum flexibility)| This pr implements "Efficient softmax approximation for GPUs" which discussed in another pr #4659| No|
| Parallelize elementwise operation with openmp [#2764](https://github.com/pytorch/pytorch/pull/2764) by @MlWoo | After v0.4.0 | Low-level tensor libraries for PyTorch(TorcH) | This pr parallelizes elementwise operation of discontiguous THTensor with openmp| No|
| Implement sum over multiple dimensions [#6152](https://github.com/pytorch/pytorch/pull/6152) by@t-vi | After v0.3.1 | ATen C++ bindings | This pr implements summing over multiple dimensions as an ATen native function and fixed #2006| No|
| Fixed non-determinate preprocessing on DataLoader [#4640](https://github.com/pytorch/pytorch/pull/4640) by @AlexanderRadionov | After v0.3.1  | `torch.utils`	DataLoader(Trainer and other utility functions for convenience) | This pr adds ind_worker_queue parameter in DataLoader to solve the non-deterministic issue which happens when DataLoader is in multiprocessing mode| No|
| Introduce scopes during tracing [#3016](https://github.com/pytorch/pytorch/pull/3016) by @lantiga | After v0.2.0 | `Scope`, `IR` (intermediate representation) and `Tracing` | This pr introduced the scopes for group operations in the tracing IR| No|

### Pull requests which are rejected：
| Pull request | Lifetime| Components it touches| PR content and Related issues| Deprecate another pull request or not|
| -------- | -------- | -------- | --------| --------|
|Fixes errors [#10765](https://github.com/pytorch/pytorch/pull/10765) by @peterjc123|After v0.4.0|ATen C++ bindings|Fixes errors when linking caffe2 statically related to issues [#10746](https://github.com/pytorch/pytorch/issues/10746) and [#10902](https://github.com/pytorch/pytorch/issues/10902|No
|Low rank multivariate normal [#8635](https://github.com/pytorch/pytorch/pull/8635)|After v0.4.0|`torch.distributions`, `torch.trtrs` |Implements low rank multivariate normal distribution where the covariance matrix has the from `W @ W.T + D`.|No
|[caffe2]seperate mkl, mklml, and mkldnn [#12170](https://github.com/pytorch/pytorch/pull/12170)|After v0.4.1|Caffe2 component of PyTorch and Docs.|1. Remove avx2 support in mkldnn. 2. Seperate mkl, mklml, and mkldnn. 3. Fix convfusion test case|No|
|Checkpointing [#4594](https://github.com/pytorch/pytorch/pull/4594)|After v0.3.0|`torch.autograd`, `torch.utils` and Docs.|Autograd container for trading compute for memory.|Be deprecated by pull request [#6467](https://github.com/pytorch/pytorch/pull/6467).
|Adding katex rendering of equations [#8848](https://github.com/pytorch/pytorch/pull/8848)|After v0.4.0|Docs, `torch/functional` and `torch.nn`|This fixes issue #8529. 1. Adds Katex extension to `conf.py` and requirements.txt. 2. Fixes syntax differences in docs. 3. Should allow documentation pages to render faster|No|


### Codify pull-request "Introduce scopes during tracing" [#3016](https://github.com/pytorch/pytorch/pull/3016)
**From Ziyu Bao:**
|  Comments Index | Code |
| :--------: | -------- |
|1|Introduce scopes, not accepted|
|2|Try-finally suggested|
|3|Change from 1 to multiple variables|
|4|Choices to make strings interned kept|
|5|Capture module name
|6|Context manager for scopes/try-finally
|7|Args to `__call__` in module are flattened (instead of `_first_var`); introduced a `torch.jit.scope` context manager
|8|Tracing_state in try finally for later proper cleaning-up
|9|Setstate, child name
|10|Remove if scope.empty in pushscope
|11|Remove if the value is not none
|12|Context manager activated only when tracing to reduce cost
|13|Module name wasn't gotten right, fixed
|14|Printing of scope printed only if defined
|15|Fixed scopes for the ONNX pass
|16|Trying to merge
|17|Cannot share common subtries between tries and not needed
|18|Appeal to merge, discussion on scope stability
|19|Add underscores, others are exposed on .torch namespace
|20|Use unique_ptr safely
|21|Raise an error if it can't pop
|22|Return scope name in scope class
|23|Fix numerous small issues
|24|Define jit in the module and attach the module to tracing
|25|Activate dynamic attributes on TracingStat; manage a stack per trace
|27|Build finished
|28|Double check: Scope ownership scheme
|29|Merged

**From Zhao Yin:**
<!-- 54321233333 -->
| Comments Index | Code |
| :------------: | ---- |
|1|Usage showed
|2|Small advice to use try-finally.
|3|Technical advice about function usage.
|4|Add name as the attribute to the module.
|5|Add context managet to handle the scope.
|6|Integration of advice above.
|7|Cope with field missing after adding the name.
|8|Remove passing an empty string to the decorator.
|9|Remove redundant condition check.
|10|Remove context when nothing is tracing.
|11|Fix $O(n^2)$ to $O(n)$ of loading module tree.
|12|Fix scopes for the ONNX pass.
|13|Wait for `autograd` PR because it changes JIT infra.
|14|This PR get rebased on the new master.
|15|Make notes that different `Graph` can't share portions of the trie.
|16|Good enough, need to merge.
|17|Two concerns: backward tracing with the scope not working and a "scope" is morally a property of the tracer.
|18|Clarifications for the two concerns: making scope inexpensive and a `Graph` node needs to hold information.
|19|State that the scopes are stable if name registered in `Module` is stable.
|20|Add an underscore and leave others exposed to `torch.`.
|21|Change pointer type for the root, scope, and children.
|22|Raise error when can't pop.
|23|Question (with the answer of not using the flat list of scope) about reversing.
|24|Put string representation inside `scope` class.
|25|Many small issues fixed.
|26|Create the field in tracing state to store extra info.
|27|Activat dynamic attributes on TracingState and manage a stack per trace.
|28|Build finished.
|29|Change the scope ownership scheme.
|30|Use vector as children container in scope trie.
|31|Avoid segfault in case of memory allocation error.
|32|Changes approved. Finally merged.
