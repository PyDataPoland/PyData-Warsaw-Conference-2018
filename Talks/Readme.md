# Talks

##  Monday Nov. 19, 2018

* "PyTorch 1.0: now and in the future" - Adam Paszke
* "Deep Learning for 3D World: Point Clouds" - Marcin Mosiołek
* "Where visual meets textual. Luna - overview" - Sylwia Brodacka
* "Can you trust neural networks?" - Mateusz Opala
* "From Data to Deliverable" - Steph Samson
* "Overview of imbalanced data prediction methods" - Robert Kostrzewski
* "Recognizing products from raw text descriptions using “shallow” and “deep” machine learning" - Tymoteusz Wołodźko, Tomasz Płomiński
* "How I learnt computer vision by playing pool" - Łukasz Kopeć
* "Distributed deep learning and why you may not need it" - Jakub Sanojca, Mikuláš Zelinka
* "AI meets Art" - Agata Chęcińska
* "Hand in hand with weak supervision using snorkel" - Szymon Wojciechowski
* "3d visualisation in a Jupyter notebook" - Marcin Kostur, Artur Trzęsiok
* "Deep Learning Semantic Segmentation for Nucleus Detection" - Dawid Rymarczyk
* "Bit to Qubit: Data in the age of quantum computers" - Shahnawaz Ahmed
* ["Transfer Learning for Neural Networks"](https://github.com/Lingaro-DataScience/transfer_learning) - Dominik Lewy
* "Spot the difference: train your image analytics model to recognize fine grained similarity" - Katarina Milosevic, Ioana Gherman,
* [In Browser AI - neural networks for everyone](https://www.dropbox.com/s/djht30ujl39w5sj/20181119_inbrowserai%20pydata%20warsaw.pdf?dl=0) - Kamila Stepniowska, Piotr Migdał
* "Using convolutional neural networks to analyze bacteriophages DNA" - Michał Jadczuk
* "Comixify: Turning videos into comics" - Adam Svystun, Maciej Pęśko,
* "High Performance Data Processing in Python" - Donald Whyte
* "What ad is this?" - Adam Witkowski
* [Spammers vs. Data: My everyday fight](https://drive.google.com/file/d/16XXvLtucyqQpMqQ7OPc0T3XHMMXSbpvN/view) - Juan De Dios Santos
* "Pragmatic application of Machine Learning in commercial products" - Łukasz Słabiński


## Tuesday Nov. 20, 2018

* "Towards Data Pipeline Hyperparameter Optimization" - Alex Quemy
* "Similarity learning using deep neural networks" - Jacek Komorowski
* "Application of Recurrent Neural Networks to innovative drug design" - Rafał A. Bachorz
* "Computer vision challenges in drug discovery" - Dr Maciej Hermanowicz
* "Learning to rank @ allegro.pl" - Tomasz Bartczak, Ireneusz Gawlik
* "The smart shopping basket: A Case Study with deep learning, Intel Movidius and AWS" - Marcin Stachowiak, Michal Dura, Piotr Szajowski
* "It is never too much: training deep learning models with more than one modality" - Adam Słucki
* ["Visualize, Explore and Explain Predictive ML Models"](https://github.com/pbiecek/Talks/blob/master/2018/DALEX_PyDataWarsaw2018.pdf) - [Przemyslaw Biecek](http://pbiecek.github.io/)
* "The Dawn of Mind Reading in Python" - Krzysztof Kotowski
* "Uncertainty estimation and Bayesian Neural Networks" - Marcin Możejko
* "A deep revolution in speech processing and analysis" - Pawel Cyrta15:30
* "Predicting preterm birth with convolutional neural nets" - Tomasz Włodarczyk
* "Can you enhance that? Single Image Super Resolution" - Katarzyna Kańska
* "Burger Quest: finding the best hamburger in town!" - Roel Bertens
* "Hitting the gym: controlling traffic with Reinforcement Learning" - Steven Nooijen
* "Step by step face swap" - Sylwester Brzęczkowski
* [Optimizing Deep Neural Network Layer Topology with Delve](https://docs.google.com/presentation/d/143p3B9V-JJKtHpKse91KboN0Z03NRK3Ze2l0bSLcom8/view) - Justin Shenk

## Lecture descriptions

#### [PyTorch 1.0: now and in the future](https://pydata.org/warsaw2018/schedule/presentation/11/) - Adam Paszke
##### Description
PyTorch is one of the main tools used for machine learning research these days. It’s been developed in beta mode for over 2 years, but this October, a release candidate for 1.0 version has been finally released! In this talk, I’ll briefly introduce the library, and then move on to showcase the cutting edge features we introduced recently.
##### Abstract
The talk will be divided into multiple sections. First, an extremely quick introduction to what PyTorch is, and what can it be used for (including use cases outside of machine learning!). Then, I will cover a number of topics that are interesting in the current context of the library, including: - Hybrid frontend (JIT compiler) - Path from research to production - C++ API and inference - Caffe2 merger - New distributed backend

#### [Deep Learning for 3D World: Point Clouds](https://pydata.org/warsaw2018/schedule/presentation/32/) - Marcin Mosiołek
##### Description
The talk is about to provide a gentle introduction into the world of 3D deep learning techniques, considering basic aspects such as input representation, typical problems and most popular models. After the talk you should be able understand common challenges occurring when working with point clouds, and more importantly, you should be able to tackle them properly.
##### Abstract
Points clouds are a common representation in applications, such as autonomous navigation, housekeeping robots and augmented/virtual reality. They provide reliable depth information that can be used to accurately localize objects and characterise their shapes. Unlike images, point clouds are sparse, have highly variable points density and require properly designed deep learning approaches.  

#### [Where visual meets textual. Luna - overview.](https://pydata.org/warsaw2018/schedule/presentation/51/) - Sylwia Brodacka
##### Description
Luna is a general-purpose programming language with an intuitive visual interface that is dedicated to data science applications. In this presentation we will show how to improve interpretation of results and communication, including with non-technical business experts, by making data visualization the driving component of data science workflow.
##### Abstract
The talk is a presentation of Luna, a visual-textual programming language and a data processing environment. It showcases a novel paradigm for data processing and explains how purely functional programming can be combined with visual representation to help people create pipelines that are more intuitive, easier to comprehend and less error-prone. We demonstrate interactive examples and discuss the possibilities of such paradigm to change the way the data is being processed across industries.

#### [Can you trust neural networks?](https://pydata.org/warsaw2018/schedule/presentation/3/) - Mateusz Opala
##### Description
Recently neural networks have become superior in many machine learning tasks. However, they are more difficult to interpret than simpler models such as decision trees. Such a condition is not acceptable in industries like healthcare or law. In this talk, I will talk on unified approach to explain the output of any machine learning model, especially neural networks.
##### Abstract
Talk is going to be build around 3 main points: 
1) Why interpretability is important? 
2) Introducing Shapley Additive Explanations 
3) SHAP framework in Python  

In 1) I will elaborate on need for interpretability of machine learning models. Latter, I will introduce SHAP framework from theoretical standpoint and provide intuitions behind it. SHAP has been introduced year ago at NIPS conference and tends to perform better than LIME or original DeepLift with respect to consistency with human intuition. In last part I will show how to use SHAP in Python on several examples like image classification and text classification among the others.

#### [From Data to Deliverable](https://pydata.org/warsaw2018/schedule/presentation/52/) - Steph Samson
##### Description
The data we need is sourced from different places. The data could even come in different formats. I will talk about how I used data from different APIs, cleaned and preprocessed it, and wrapped it up under a new API for a restaurant discovery service. This talk is for attendees looking to jump into the data science industry for the first time.
##### Abstract
Many practitioners in our industry often work with unclean data. As a result, this data is not fit to work immediately with off-the-shelf libraries, regardless of whether these libraries are for machine learning or for creating a RESTful API. In this talk, I will describe my end-to-end process to get a data set to a state that can be used for a deliverable.  The pipeline begins with extracting data. The comes the cleaning and preprocessing. Cleaning and preprocessing data is rarely, if ever, painless. One will often encounter formatting and encoding errors, and sometimes different representations altogether. I will demonstrate some techniques I use when examining dirty data sets. From preparing the data, I will then demonstrate how I resolve the differences in the data sources by creating a new data structure. This data structure is subsequently exposed through a RESTful API.  By the end of this talk, attendees can hopefully gain some insight into how much data cleaning and preprocessing is involved in a work day.

#### [Overview of imbalanced data prediction methods](https://pydata.org/warsaw2018/schedule/presentation/37/) - Robert Kostrzewski
##### Description
Imbalance ratio is a definition describing relation of frequency of data classified to following classification classes. Assuming binary classification as datasets' domain, higher the ratio is, more disproportion on feature existence distribution is observed. The talk’s goal is to compare, in both theoretical and practical ways, various fresh methods of dealing with the problem.
##### Abstract
Intro  Imbalance ratio is a definition applicable to Machine Learning classification problem. It describes relation of frequency of data classified to following classes. Assuming binary classification as datasets domain, higher the ratio is, more disproportion on feature existence distribution is observed. The presentation describes various fresh methods of dealing with imbalance problem. With the support of theory explanation, definition’s papers references and experiments performed on real datasets there is a compare of the mentioned techniques performed. The following algorithms are introduced during the talk:    
* Splitbal - the technique based on splitting majority class subset, sub-classifiers creation and ensembling process. Described in _A novel ensemble method for classifying imbalanced data_   
* SMOTE - algorithm fully called _Synthetic Minority Oversampling TEchnique_. There is a method of oversampling minority class in a specific way. Described in _SMOTE: Synthetic minority over-sampling technique_   
* EDBC - the dissimilarity-based imbalance data classification method. Described in _A dissimilarity-based imbalance data classification algorithm_  ## Experiments  As a result of working on paper called _Imbalanced data classification using MapReduce and relief_ , already mentioned algorithms have been compared in a way of experiments applicated to 11 datasets with various size and imbalance ratio.

#### [Recognizing products from raw text descriptions using “shallow” and “deep” machine learning](https://pydata.org/warsaw2018/schedule/presentation/17/) - Tymoteusz Wołodźko
##### Description
We will compare “shallow” and “deep” machine learning approaches to solving a natural language processing problem. Pros, cons and consequences of both choices will be discussed.
##### Abstract
Working with raw text data is usually hard, because of their noisy nature. During this talk, we will show two proof-of-concept solutions for the practical problem of recognizing products given raw text descriptions on the online e-commerce platform. We took two different approaches: “shallow” (decision tree ensembles in sklearn/xgboost) and “deep” (recurrent neural network in Keras) machine learning. “Shallow” methods need more feature engineering, but in certain cases it can provide high accuracy at lower computational cost. On the other hand, deep learning can be fed with (almost) a raw text and does the feature engineering semi-automatically, but at the cost of tuning the architecture and hyperparameters of the models.

#### [How I learnt computer vision by playing pool](https://pydata.org/warsaw2018/schedule/presentation/33/) - Łukasz Kopeć
##### Description
Is there a way to automatically say how warped a pool table is, just by looking at a video of people playing? In this talk I will explain how I did that using OpenCV for pool balls tracking, and how more heuristic approaches may improve this model’s performance. I will also present a simple analysis of the effect of different levels of warp on players’ scores.
##### Abstract
Imagine your workplace has a pool table and runs a pool tournament that you’re very excited about. You track the games, but it turns out there’s an extra factor at play: the table is pretty uneven. How would you quantify its ‘wonkiness’? Does it even matter for the scores?  In this talk I will describe how I used a combination of off-the-shelf computer vision tools like OpenCV’s multiple object trackers, trained models for object detection, and simple geometry to achieve a rating for a table, based on a video of gameplay. I will then explain why the standard models may not perform as well as may be expected, but heuristic approaches that incorporate some of the domain may help improve performance. Finally, I will talk about insights gained from comparing these ratings with scores from real games, and how we can (and why we probably shouldn’t) use it in a scoring model.

#### [Distributed deep learning and why you may not need it](https://pydata.org/warsaw2018/schedule/presentation/38/) - Jakub Sanojca
##### Description
Deep learning thrives with always bigger networks and always growing datasets but single machine can only handle so much. When to scale to multiple machines and how do do it efficiently? What pros and cons available options have and what is theory behind their approach to distributed training? In this talk we will answer those questions and show what problems we are trying to solve at Avast.
##### Abstract
With growing accuracy of deep learning, the needs of computation grow too. Cutting-edge models, hyperparameter tuning or architecture search require either big amounts of GPU memory or take long time to finish. It is not always feasible to invest in high-end server-grade hardware and the development of Big Data shows that scaling horizontally is viable alternative. Why even bother about distributing training to multiple machines and how it differs from single-machine scenario? What are the options available and which one is best?  This talk aims to introduce audience to few available distributed deep learning systems like Distributed Tensorflow, Tensorflow on Spark or Horovod and compare them including both theory and our benchmarks. We will explain how this need arose in Avast, how we solved it and share experiences from our journey.

#### [AI meets Art](https://pydata.org/warsaw2018/schedule/presentation/9/) - Agata Chęcińska
##### Description
Intersection of artificial intelligence and art world is an intriguing concept. I will show several examples of this intersection, with the focus on the AI based "Museum Treasures" game, a winning solution from the HackArt: a hackathon organised by the National Museum in Warsaw. I will also touch on the topic of analogies between creative processes and machine learning processes.
##### Abstract
Recent developments in the field of deep learning brought to our attention questions on how artificial intelligence may impact art world, how it can be used by the artists, or whether it can be considered creative.  Firstly, I would like to share some examples of this intriguing intersection, with the special focus on a recent project that I was involved in: [Museum Treasures](https://deepsense.ai/museum-treasures-ai-at-the-national-museum-in- warsaw/), a concept of a game for kids & parents, a scavenger hunt, combining art history and AI. This is a winning solution from this year’s [HackArt](https://hackathon.mnw.art.pl) hackathon organised by the National Museum of Warsaw. The game uses object detection applied to digitalised paintings, and in more advanced version it can also apply models generating contexts. I would like to discuss the concept of the game from the point of use of the machine learning models.  Secondly, I would like to touch on the subject of analogies between creative processes and machine learning processes. By showing some intriguing analogies between these two and potential venues to develop, I would like to ask the question on how AI can help us better understand the nature of our creativity.

#### [Hand in hand with weak supervision using snorkel](https://pydata.org/warsaw2018/schedule/presentation/34/) - Szymon Wojciechowski
##### Description
In Natural Language Processing, where we want to use a supervised model the frequent problem is the availability of labels. One way of circumventing tedious and cost-intensive manual annotation of (tens) of thousands of samples may be weak supervision, where heuristics, fragmentary datasets with annotations and crowd-sourced indications (on a fraction of samples) can be unified to generate labels.
##### Abstract
Let's assume the following scenario: you've scrapped half of the web in order to build a model supporting your research. You've collected and organized thousands if not millions of data pieces that you want to subsequently reason about. Everything is fine as long as you won't be needing any labels associated with this horrifying amount of data...  What can you do? - Give up. - Cry in the depth of your soul, traversing samples one by one, manually annotating them, and finally give up around the 200th sample. - Crowdsource for the annotations, where you may be subjected to all shades of mankind like malice or incompetence.  Or you can find some loose patterns in the data which characterize samples that you are interested in for whatever reason, sum them up and let computers reason for you. This is where weak supervision steps in.  Weak supervision is a paradigm where you can gather under one coherent interface as many types of data annotations as you have at your disposal. They do not need to be 100% perfect, they do not need to cover each and every sentence, they do not need to be very strong - if they are meaningful in the context of your problem, then you are good to go. That's why you can mix up all the annotations that you tediously developed before you dropped off, partially faulty results from crowdsourcing and all the remarks from domain experts around you.  To make all the vague description above more practical, one of the solutions to deal with such a problem will be discussed during the presentation: snorkel. It is a tool designed under **data programming** paradigm, where your expertise is passed to the model as a bundle of functions expressing all your hitherto knowledge on the domain.

#### [3d visualisation in a Jupyter notebook](https://pydata.org/warsaw2018/schedule/presentation/36/) - Marcin Kostur
##### Description
K3D-jupyter is a notebook visualization package which we have been developing since 2015. It offers a powerful and efficient tool with a simple interface to appealing techniques like volumetric rendering, meshes, lines, and points. Its performance allows for smooth visualization of datasets like 100 million points and updates with their position from python interface.
##### Abstract
The Jupyter Notebook is a web application that enables the creation and sharing of executable documents containing live code, equations, visualisations and explanatory text. In particular, Jupyter is used actively for interactive and exploratory computations, often involving visualisation of data. Two-dimensional visualisation in Jupyter is an area with popular and well-established tools, but there have been fewer mature solutions for interactive three-dimensional visualisation. K3D-jupyter is a package which provides a fast and simple 3d plotting tool in the Jupyter notebook. The primary aim of K3D-jupyter is to be easy for use as stand alone package like matplotlib, but also to allow interoperation with existing libraries as VTK. The power of ipywidgets makes it also a fast and performant visualisation tool for HPC computing e.g. fluid dynamics.  This software is part of OpenDreamKit project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 676541.

#### [Deep Learning Semantic Segmentation for Nucleus Detection](https://pydata.org/warsaw2018/schedule/presentation/16/) - Dawid Rymarczyk
##### Description
Semantic segmentation is the process which aims to classify individual pixels of an image. Recently, Kaggle hosted the 2018 Data Science Bowl competition dedicated to nucleus detection and segmentation based on microscopic images. In this talk, I will present two approaches to this problem, based on U-Net and Mask R-CNN.
##### Abstract
U-Net is a deep learning approach to image segmentation task, that works with a relatively small number of training images and produces precise segmentation. On the other hand, Mask R-CNN requires much more training data, however, it detects many object instances and simultaneously generates high- quality segmentation map for each object. While Mask R-CNN outperforms all existing, single-model entries on every task (including the COCO 2016 challenge winners), it has lost 2018 Data Science Bowl competition with much simpler U-Net with heavy pre and postprocessing. In this talk, I would like to concentrate on the reasons for this failure. Moreover, I will present the possible extensions of both methods and show the practical guidelines of how to use them.

#### [Bit to Qubit: Data in the age of quantum computers.](https://pydata.org/warsaw2018/schedule/presentation/22/) - Shahnawaz Ahmed
##### Description
We will discuss the qubit, a quantum bit, and what data processing and machine learning in the quantum computing era might look like using python based open- source tools.
##### Abstract
The fundamental unit of information, a bit, will be replaced by a qubit, or a quantum bit in the age of quantum computing. The use of a quantum state to store, manipulate and process information can have far reaching effects for large-scale data analysis. An N qubit quantum computer can simultaneously exist in a superposition of all 2^N states. Google's latest 72 qubit chip already begins to challenge the limits of classical supercomputers for quantum simulations. On the other hand, efforts towards developing such technologies which are capable of handling the large amount of information in a quantum system has led to a new possibility. The possibility of big data analysis with quantum computers. Rigetti, IBM, Google, Microsoft and many other organizations have already released open-source tools for writing quantum algorithms. pyQuil, a tool developed by Rigetti makes it easy to write unsupervised Machine Learning algorithms to be run on their cloud based quantum computer. Similarly, many of these tools are python-based and in this talk I will give a brief idea about some of these projects starting from QuTiP: the Quantum Toolbox in Python. The talk will focus on familiarizing the audience with the concept of a qubit, quantum algorithms and how open source tools are laying the foundation of future data processing methods with quantum computers.  The outline of the talk will be as follows:    
1. Bit vs Qubit: A very short description of quantum weirdness.    
2. Quantum circuits and algorithms: Manipulating quantum data.    
3. QuTiP, ProjectQ, pyQuil, QISKIT: Python based tools for quantum simulation and algorithms.    
4. Conclusion.

#### [Transfer Learning for Neural Networks](https://pydata.org/warsaw2018/schedule/presentation/40/) - Dominik Lewy
##### Description
During the session I will explain the notion of transfer learning both in context of single task learning and recently very popular multitask models. I will give a brief overview of the state of the art approaches to machine translation (zero shot translation) and image recognition with a focus on transfer learning.
##### Abstract
During the session I will explain the notion of transfer learning both in context of single task learning and recently very popular multitask models. I will give a brief overview of the state of the art approaches to machine translation (zero shot translation) and image recognition with a focus on transfer learning.  Next I would like to deep dive into the topic of Deep Convolutional Neural Networks (ConvNet) to visualize the information extracted by the network and how it differs depending on how deep into the network we are. After that I would present two approaches on how we could benefit from transfer learning when solving image recognition tasks:  
* ConvNet as feature extractor – this approach uses already trained ConvNet and simply changes to top feed-foreward fully connected layers  
* ConvNet as start for further learning – this approach also tries to fine tune the weights in the convolutional part of the Neural Net  

During the session I would also like to mention the very important idea of Common Representation Learning which is critical when it comes to applying transfer learning to problems that are similar in the idea but use different representation space. Languages are a good example of this, although we say the same thing the input vocabulary sets are usually disjoint. The approaches that I would cover in the session are:  
* Canonical Correlation Analysis – an approach that focuses on minimizing the similarity of two different inputs when transformed to common representation  
* Autoencoder – an approach that focuses on minimizing the reconstruction error 
* A combination of both – a hybrid approach joining advantages of the previous two

#### [Spot the difference: train your image analytics model to recognize fine grained similarity](https://pydata.org/warsaw2018/schedule/presentation/20/) - Katarina Milosevic, Ioana Gherman
##### Description
Imagine two images of the same car model, same color and with small scratches on bumpers. How would you make a machine look at the scratches and decide if they are the same? This is a story about the implementation of a new structure of neural network trained on “triplets” of images which recognizes fine grained images similarity based on Deep Ranking algorithm.
##### Abstract
Imagine two images of the same model of car, the same color and with small scratches on the bumpers. How would you make a machine look at the scratches and decide if they are the same? This talk will be a story about the implementation of a new structure of neural network trained on “triplets” of images. We will guide you through the implementation of the Deep Ranking algorithm for recognizing fine grained similarity of images, based on a real use case of car damages, with the purpose of fraud fighting. Presentation will follow the outline: 
* Business problem 
* Data description (images and metadata) 
* Triplets creation
* Deep Ranking model 
* Results

The goal is to understand how is it possible to train a deep ranking model such that it focuses on damage differences. The training set contains positive pairs represented by pictures of a particular damage on the same car, and their corresponding negatives represented by very similar damages on the same car model. This network is able to learn similarity better and faster than methods based on vector comparison. The talk targets all Data Scientists interested in the Image Analytics topic, who are familiar up to a certain degree with the techniques already used in this field.

#### [In Browser AI - neural networks for everyone](https://pydata.org/warsaw2018/schedule/presentation/26/) - Kamila Stepniowska
##### Description
Let's talk about In Browser AI - the open source educational project brings together Python & JavaScript.  Bring deep learning demos of your research papers, commercial or side projects to the browsers. Create interactive explanations and tutorials, using interactive environment! Train in Python, show in JavaScript.
##### Abstract
Train in Python, show in JavaScrip!  InBrowserAI is a group of four data science practitioners and enthusiasts, who decided to cooperate to make ML more available for scientists and business. We are building an Open Source educational materials that we hope to be broadly used and help to support this goal.  We will talk about: Project and research demos - How to use TenserFlow.js to Show your last research project. Convince your client with an interactive demo of your service. Install-free apps - Make it work on a desktop (Win, Linux, macOX) and mobile. Train offline in Python, show in JavaScript! Real-time interaction - Use a microphone and video inputs. Interact with provided content is a smooth way. Privacy - Deliver AI solutions with privacy in mind. Data stays on the client side.

#### [Using convolutional neural networks to analyze bacteriophages DNA](https://pydata.org/warsaw2018/schedule/presentation/55/) - Michał Jadczuk
##### Description
With the overuse of antibiotics causing the emergence of superbugs, people all over the world are looking into their safer alternatives. Bacteriophages - or viruses that feed on the bacteria - might just be one of them. All phage strains must be carefully analyzed before commercial use, though. This presentation will explain how machine learning can speed up the evaluation of bacteriophages DNA.
##### Abstract
The bacteriophages are viruses feed on bacteria that are viewed as a potential alternative to the antibiotics. One could say that antibiotics work like “bombs” – they kill both good and bad bacteria in your system, which isn’t ideal. On the other hand, the so-called _phage cocktails_ can target only the specific strains of bacteria that cause the actual treated disease, which potentially makes them much safer. Also, the overuse of antibiotics in the agriculture industry causes the emergence of the “superbugs”, which are drug- resistant bacteria – not to mention that antibiotics can cause other side effects upon meat consumption. Commercialized bacteriophage-based alternatives could potentially prevent that.  However, before commercializing a certain bacteriophage strain, one has to perform a series of tests to make sure it meets all the necessary criteria. The tests might take weeks or even months of tedious laboratory work. One of the most important criteria that the phages must meet, right next to the bacteria type they target, is the way they can reproduce. Based on the reproduction cycle, one can detect two distinct types of phages: lytic and lysogenic. Only the lytic bacteriophages can be used in phage cocktails, as lysogenic strains might cause bacteria mutations.  Detecting whether the strain is lytic or lysogenic used to require manual work, but we managed to speed up this process. All thanks to developing an automated classifier that can detect the reproduction cycle type using bacteriophage genome as input processed by the convolutional neural networks. This lecture will describe the origin of the project, difficulties with processing DNA, and – more importantly – it will try to prove that convolutional neural networks are not only effective for images, but can also handle sequential data (such as the DNA) reasonably well.

#### [Comixify: Turning videos into comics](https://pydata.org/warsaw2018/schedule/presentation/7/) - Adam Svystun
##### Description
In our talk, we will present a Web-based working solution for video comixification - a task of converting a video into a comics. We will disclose technical details of how our comixification engine works and, finally, we will give a first public presentation of a working video comixification demo available at comixify.ii.pw.edu.pl.
##### Abstract
In our talk, we will present a Web-based working solution for video comixification - a task of converting a video into a comics. We split this task into two separate problems: (a) frame extraction and (b) style transfer. To extract meaningful, representative frames from the video we employ a keyframe extraction algorithm based on Reinforcement Learning, while for transferring the style into comics we implement a generative adversarial network (GANs) model. Since there have been many works published on the so- called neural style transfer, we evaluate them all on the very same task, namely frame comixification and select the most appropriate method. We examined different combinations of Adaptive Instance Normalization, Universal Style Transfer and GAN models and confront them to find their advantages and disadvantages in terms of qualitative and quantitative analysis. In the talk we will disclose technical details of how our comixification engine works and, finally, we will give a first public presentation of a working video comixification demo available at [comixify.ii.pw.edu.pl](http://comixify.ii.pw.edu.pl/).

#### [High Performance Data Processing in Python](https://pydata.org/warsaw2018/schedule/presentation/15/) - Donald Whyte
##### Description
numpy and numba are popular Python libraries for processing large quantities of data. This talk explains how numpy/numba work under the hood and how they use vectorisation to process large amounts of data extremely quickly. We use these tools to reduce the processing time of a large, real 600GB dataset from one month to 40 minutes, even when the code is run on a single Macbook Pro.
##### Abstract
The Internet age generates vast amounts of data. Most of this data is unstructured and needs to post processed in some way. Python has become the standard tool for transforming this data into more useable forms.  numpy and numba are popular Python libraries for processing large quantities of data. When running complex transformations on large datasets, many developers fall into common pitfalls that kill the performance of these libraries.  This talk explains how numpy/numba work under the hood and how they use vectorisation to process large amounts of data extremely quickly. We use these tools to reduce the processing time of a large, real 600GB dataset from one month to 40 minutes, even when the code is run on a single Macbook Pro.

#### [What ad is this?](https://pydata.org/warsaw2018/schedule/presentation/31/) - Adam Witkowski
##### Description
Ads on the web often consist of a picture with some text. Based on this information, can you tell what exactly is advertised? In this talk I will describe a system that automatically categorizes ads seen on the web. I will talk about potential approaches to this problem and describe in detail the chosen solution. This system was created as a joint project between Gemius and MIM Solutions.
##### Abstract
I will talk about a solution to the following problem: we are given an advertisement from the web that consists of a picture and some text. We also know the website the ad links to. We need to determine what brand is advertised (from a list of known brands). A brand can be, for example:    
* a specific product (a model of a car),   
* a type of product (drugs for weight loss),   
* a website (Allegro.pl) or   
* a charity (pajacyk.pl).  

Such task is quite easy for humans but hiring humans is quite expensive. Therefore, we looked for a way to automate this task. From machine learning point of view, this is a multiclass classification problem. It is complicated by the fact that there are a lot of classes and the input is a combination of text and image.  There are many possible approaches to such problem, among them:    
* One vs Rest binary classification   
* neural net with many outputs   
* hierarchical classification  I will shortly discuss the pros and cons of those approaches and tell you why none of them were good enough for us. Then I will describe the chosen solution.

#### [Spammers vs. Data: My everyday fight](https://pydata.org/warsaw2018/schedule/presentation/4/) - Juan De Dios Santos
##### Description
LOVOO, a dating app, is attractive to spam. These spammers disguise themselves as real users, using believable images and accurate descriptions on their profiles to lure our community into chatting with them. This presentation is on everything Antispam. In it, I will talk about the architecture of the system, data, and the algorithms and methods we employ to fight the spammers.
##### Abstract
Spam sucks, right? You don't like it, I don't like it, users don't like it, and my boss doesn't want them on the app.  LOVOO, the dating platform I work at, is an attractive target for spammers. These fake users (or bots) disguise themselves as real ones, using believable images and accurate descriptions on their profiles to lure our community into believing they are real people.  To fight them, we have a dedicated group - the Antispam team - whose purpose is to research and develop techniques that rely heavily on machine learning and massive real-time data processing to avoid the proliferation of spammers in the platform.  This presentation is on everything Antispam. In it, I will talk about the Kafka and Google Cloud-based architecture of the system, data, our algorithms and methods we employ to kick spammers, and lastly, I will show some insight and numbers about our performance.

#### [Analysing Russian Troll Tweets data with Python](https://pydata.org/warsaw2018/schedule/presentation/27/) - Mia Polovina
##### Description
This talk focuses on the insights gathered from analysis of Russian Troll Tweets, a dataset created by researchers at Clemson University and released on Github by FiveThirtyEight.
##### Abstract
Social media sites are increasingly used for propagation of misinformation. Recent efforts include sophisticated campaigns run on Facebook and Twitter which aimed to interfere in the 2016 US elections and politics. This talk will focus on a campaign that saw trolls engage in such efforts on Twitter. Insights obtained with Exploratory Data Analysis (EDA) and text analysis of Russian Troll Tweets dataset will be presented. The talk will also highlight the importance of combating misinformation and computational propaganda.

#### [Pragmatic application of Machine Learning in commercial products.](https://pydata.org/warsaw2018/schedule/presentation/53/) - Łukasz Słabiński
##### Description
I will show a few interesting examples of application the state of the art Machine Learning techniques in products used by millions of users. In this context, I will discuss also general but pragmatic need of R&D in the area of Artificial Intelligence from the perspective of Samsung Electronics - the world's largest information technology company, consumer electronics and chips maker.
##### Abstract
During this talk I will show a few examples of application of the state of the art Machine Learning techniques in world-wide products used by millions of users. The examples of ML applications will be selected from various areas like:    
* Natural Language Processing   
* Audio, Image and Video Processing   
* Data Analytics  

In the context of real-life examples, I will discuss also the general but pragmatic need of R&D in the area of Artificial Intelligence from the perspective of Samsung Electronics - the world's largest information technology company, consumer electronics and chips maker.

#### [The Neural Aesthetic](https://pydata.org/warsaw2018/schedule/presentation/65/) - Gene Kogan
##### Description
Over the past several years, two trends in machine learning have converged to pique the curiosity of artists working with code: the proliferation of powerful open source deep learning frameworks like TensorFlow and Torch, and the emergence of data-intensive generative models for hallucinating images, sounds, and text as though they came from the oeuvre of Shakespeare, Picasso, or just a gigantic database of digitized cats.
##### Abstract
This talk will review these developments through the lens of creative, exploratory research. Artistic metaphor helps clarify that which is otherwise shrouded by layers of academic jargon, making these highly specialized subjects more accessible. A selection of experimental projects at the intersection of AI and new media art will be shown, including several real- time interactive demos. It will conclude with a survey of interdisciplinary tools and learning resources for artists and data scientists alike, offering an accessible introduction to this field.

#### [Similarity learning using deep neural networks](https://pydata.org/warsaw2018/schedule/presentation/14/) - Jacek Komorowski
##### Description
Deep neural network give very good results in visual object recognition tasks, but they require large number of training examples from each category. I'll present a class of neural network architectures, that can be used when only few training examples from each class are available. They are based on 'similarity learning' concept and can be used to solve various practical problems.
##### Abstract
Deep neural networks give excellent results in many visual object recognition and image classification tasks. However, to achieve good performance, the network must be trained using very large number of examples from each category. In many practical situations, this is not always feasible. For example, for face recognition applications, we have only few or even one training example per each person.  In this talk, I’ll present a class of neural network architectures, such as Siamese or triplet networks, based on ‘metric learning’ or ‘similarity learning’ concept. These methods can be effectively used to learn from limited number of examples from each category. I’ll demonstrate how these methods can be used to solve various practical problems, such as face recognition, pedestrian re-identification or visual search.

#### [Application of Recurrent Neural Networks to innovative drug design](https://pydata.org/warsaw2018/schedule/presentation/30/) - Rafał A. Bachorz
##### Description
The presentation shows the application of Recurrent Neural Network to the problem of innovative drug design. The audience will also have an opportunity to get familiar with modern cheminformtics libraries available in Python ecosystem. The final element of the preeentation is live demonstration of the pretrained, generative predictove model applied to generation of new molecules.
##### Abstract
Recurrent Neural Networks are surprisingly effective. They are capable of solving many problems which usually involve some sort of sequence. This might be the sequence of characters, sequence of words, the notes, the elements of energy time series describing the electrical energy consumption and many other examples. The common denominator of all of them is the sequence nature of the quantity. The chemical compound, being in principle the three-dimensional object, can be successfully and uniquely encoded as a sequence of characters. Relatively simple syntax of SMILES code allows on fast and comprehensive translation of complicated structure of the molecule into convenient short- hand notation. Within this presentation a set of biologically active species have been turned into the SMILES character representation. Such a representation have been passed to particular form of the Recurrent Neural Network. Carefully converged neural Network has been later on "stimulated" to produce other molecules. It turns out that these molecules are, to large extent, syntactically correct. It is also postulated that together with the knowledge related to the proper syntax of the molecule, also the semantics - i.e. the biological activeness - has also been learned. This would mean that new molecules are expected to pose similar biological properties. There are strong indications that it is indeed the case. Entire workflow of this application has been developed within the Python Ecosystem. The audience will have an opportunity to get familiar with theoretical background of Recurrent Neural Network and the cheminformatice tools that have been used to create relevant data sets. Within the final part of the presentation the pretrained model will be used to generate new molecules.

#### [Computer vision challenges in drug discovery](https://pydata.org/warsaw2018/schedule/presentation/10/) - Dr Maciej Hermanowicz
##### Description
I will present a high-level overview of how automated image analysis approaches can be incorporated into pharmaceutical discovery pipelines. By taking a look at two GSK case studies I will demonstrate how to apply computer vision techniques to featurize imaging data, enabling the use of standard machine learning algorithms. I will highlight how these techniques benefit the drug discovery process.
##### Abstract
I will present a high-level overview of how automated image analysis approaches can be incorporated into pharmaceutical discovery pipelines. I will explore the nature of imaging features such as Zernike moments, Haralick coefficients and parameter-free TAS.  I will then demonstrate how to use computer vision libraries (OpenCV & mahotas) to extract these features from microscopy images and how to use them as input to machine learning models implemented in sklearn. I will highlight how these techniques benefit the drug discovery process.  Prior knowledge of pandas and sklearn required.

#### [Learning to rank @ allegro.pl](https://pydata.org/warsaw2018/schedule/presentation/35/) - Tomasz Bartczak
##### Description
We will share our experience gained during development of ‘learning to rank’ system in an e-commerce setting. Our talk will cover both theoretical and practical aspect of the project - general approach in learning to rank, suitable tools, data and modelling pipeline and production deployment considerations. Finally, we will share some tips and tricks that come from actual, practical insights.
##### Abstract
With almost 100 million offers coming from a diversified set of categories and 40 million searches daily - it is a challenge to address the need of our customers - identify the most relevant offer and present it on the top of the results list. Being able to do it with both high quality and low latency makes it even harder. Apart from looking at the technical side of developing a ‘learning to rank’ system - we will additionally show the evolution of our experimentation and modelling processes.

#### [The smart shopping basket: A Case Study with deep learning, Intel Movidius and AWS](https://pydata.org/warsaw2018/schedule/presentation/25/) - Marcin Stachowiak
##### Description
Our objective was to build a connected intelligent shopping cart/basket, which will detect, which products have been placed in it and will generate shopping recommendations for the current cart user. We have used the state-of-the-art, real-time object detection system - YOLOv2, which deep architecture has been reduced to accelerate the evaluation on the Raspberry PI device and the Intel® Movidius.
##### Abstract
During the talk we will show our project live and we will also talk more about innovative technical aspects such as the Intel® Movidius™ Neural Compute Stick, AWS IoT and AWS Lambda.   Our solution has been also presented at the O'Reilly Artificial Intelligence Conference 2018 in London.  We have used several innovations in our project:  Intel® Movidius™ Neural Compute Stick, AWS IoT Core, AWS Lambda, Docker, Tensorflow, OpenCV, Intel DevCloud, Raspberry Pi, AWSIoTPythonSDK library. Our algorithm recognised the product even when only the back of the package was visible. It means that the logo doesn't have to be visible. Only a texture and a shape will be enough. We've observed nearly 100% accuracy. It was enough to put the product in the basket, without paying attention to its arrangement in space.      Traditional (off-line) retailers in the self-service stores do not have many opportunities to offer additional products (basicaly only by the cashier). Our solution gives the off-line retailer additional opportunieties to advertise their products in the very simmilar fashion, like on-line sellers (i.e. based on the products already placed in a shopping basket/cart).

#### [It is never too much: training deep learning models with more than one modality.](https://pydata.org/warsaw2018/schedule/presentation/21/) - Adam Słucki
##### Description
Using visual features alone is not enough to fully exploit content of social media videos. We propose a whole pipeline for extracting textual data embedded in videos and fusing them with the original visual data to predict drops in viewer’s retention.
##### Abstract
With all the hype that deep learning is causing, research in this area is advancing very quickly. Using images or text as input for successful prediction is a day-to-day task for most machine learning researchers. However, we think that architectures that use several modalities are still not that popular. We claim that combining different, often complementary, signals is a step forward in designing accurate and effective algorithms.  In this spirit, during our talk we want to outline how we approach a real-life challenge of predicting retention of social media videos.  At the beginning, we will talk about methods that we used to compare corresponding parts of videos of different lengths to calculate relative retention drops. Then, we will talk about challenging preprocessing step of extracting textual overlays from video. Finally, we will show how we approached the task of fusing visual and textual data. In order to solve our problem we used a range of deep learning techniques: LSTM, CNNs and C-RNN (convolutional recurrent neural networks). Models were developed in Keras and Tensorflow.  Hopefully, at the end of the talk you will be familiar with techniques of processing both videos and text in order to train multimodal models.

#### [Visualize, Explore and Explain Predictive ML Models](https://pydata.org/warsaw2018/schedule/presentation/19/) - Przemyslaw Biecek
##### Description
Why you need tools for exploration, visualisation and explanation for predictive models? During the talk I will present use cases in which model interpretability is crucial and overview tools that support model interpretability (Ceteris Paribus, Break Down, LIME, live, auditor). See more at DALEXverse: https://github.com/pbiecek/DALEX
##### Abstract
The talk will be related to Explainable Artificial Intelligence (XAI). I will introduce key concepts and tools, as in the book https://github.com/pbiecek/DALEX_docs

#### [The Dawn of Mind Reading in Python](https://pydata.org/warsaw2018/schedule/presentation/18/) - Krzysztof Kotowski
##### Description
An introduction to EEG signal analysis from low-cost devices using the MNE Python package. A review of advances and the future of "mind reading".
##### Abstract
Today, in the machine learning and artificial intelligence era we are closer to the mind-boggling reality from the Black Mirror TV series than many of us think. This is somehow frightening but certainly exciting vision. The presentation will be a gentle introduction to the analysis of brain signals using the Emotiv EPOC+ low-cost EEG and the MNE Python package. It will cover both theoretical basics, practical examples, and state-of-the-art applications. Including brain activity measurement methods, analysis and visualization of the signal in Python and machine learning methods of "mind reading". The talk will be addressed to audience at all levels.

#### [Uncertainty estimation and Bayesian Neural Networks](https://pydata.org/warsaw2018/schedule/presentation/57/) - Marcin Możejko
##### Description
We will show how to assess the uncertainty of deep neural networks. We will cover Bayesian Deep Learning and other out-of-distribution detection methods. The talk will include examples that will show how to implement the methods in Pytorch.
##### Abstract
Overconfidence is nowadays one of the most hot topics in discussion about safety of Machine Learning applications. This problem strikes both, models which very often suffer from providing confident scores for problematic cases, and researchers who are often likely to believe that good results on validation sets prove that model will generalize well to the new examples. Because of this we would like to dedicate our talk to the following topics:  Analysis of different kinds of problems that may occur due to overconfidence and their corresponding uncertainty types,  Introduction of a variety of methods on how to estimate the level of uncertainty for a given model. We will mostly concentrate on techniques from bayesian deep learning family.  We will show how to implement them with Pytorch.

#### [A deep revolution in speech processing and analysis](https://pydata.org/warsaw2018/schedule/presentation/54/) - Pawel Cyrta
##### Description
In the past two years, we’ve seen the industry discovery of speech as a critical interface protocol between humans and machines, especially for cloud- based information queries driving by speech recognition. These create significant new opportunities for every application that touches audio or video - opening up new potential for improved intelligibility, personalisation and customer “stickiness”.
##### Abstract
In the past two years, we’ve seen the industry discovery of speech as a critical interface protocol between humans and machines, especially for cloud- based information queries driving by speech recognition. However, speech recognition is just the tip of the iceberg for cloud-based speech. A whole new set of basic functions - speech enhancement, speaker identification and authentication, background noise classification - are becoming available. These create significant new opportunities for every application that touches audio or video - opening up new potential for improved intelligibility, personalisation and customer “stickiness”. We use BabbleLabs Clear Cloud as an example of breakthrough deep learning technology applied to widely-applicable speech APIs, give a sense of the future roadmap of speech-centric applications.

#### [Predicting preterm birth with convolutional neural nets](https://pydata.org/warsaw2018/schedule/presentation/13/) - Tomasz Włodarczyk
##### Description
The desirable date of the birth of a child follows the full duration of pregnancy. According to WHO data, 15 million children are born prematurely every year, of which 1.1 million dies, unfortunately. In this talk, we will present how to improve prediction rate of the spontaneous preterm delivery using deep learning and computer vision methods.
##### Abstract
Preterm delivery (preterm birth) is the most common cause of neonatal death. Despite great advancement in pregnancy care, frequency of preterm delivery does not decrease. Therefore, it is important to develop prediction methods that will assess risk of the preterm delivery and thus enable gynecologists to start appropriate prevention treatment. Current diagnostic methods that assess risk of the spontaneous preterm delivery involve collection of maternal characteristics (via interview) and transvaginal ultrasound (US) conducted in the first and second trimester of pregnancy. Analysis of the US data is based on gynecologist's expertise, that is visual inspection of images, which is sometimes supported by hand-designed image features such as cervical length. Such approach is naturally prone to errors, thus approximately 30% of spontaneous preterm deliveries are not predicted. Moreover, 10% of predicted preterm deliveries are false-positives. 

During the talk, we will present results of our project focused on improving prediction rate of spontaneous preterm delivery. To that end, we use a deep neural network architecture trained for segmenting prenatal ultrasound images and another network to classify the resulting cropped image. We validate our results on real-life cases as our research is done in a close collaboration with gynecologists from Warsaw University of Medicine - they provide us with constant feedback and assess the usability of our solution in a day-to-day gynecologists' routine.

Our talk is meant for intermediate machine learning researchers, engineers and students who are interested in the practical aspects of using deep learning for medical imaging.

#### [Can you enhance that? Single Image Super Resolution](https://pydata.org/warsaw2018/schedule/presentation/28/) - Katarzyna Kańska
##### Description
The talk will introduce a problem of upscaling a picture with as small loss of quality as possible using deep learning techniques. What metric to use when evaluating the solution? Upsample the image in the beginning or near the end of your neural network? Which upscaling layers to use? Answers to these and more questions on this topic will be discussed.
##### Abstract
Single Image Super Resolution is a problem where, having a single low- resolution image, one wants to produce an estimate of the corresponding high‑resolution image. Several neural network architectures that tackle this problem will be described and compared. Special attention will be paid to different upscaling layers in neural networks and their properties. Also, the concept of perceptual loss and its application to Super Resolution will be presented. All of these topics will be supplemented with relevant experience from developing research code in PyTorch for this problem. Finally, results of the research will be presented.

#### [Burger Quest: finding the best hamburger in town!](https://pydata.org/warsaw2018/schedule/presentation/49/) - Roel Bertens
##### Description
At our company we like to eat burgers. Also, we like to analyse data. So on one Friday we decided to leverage our expertise and use online reviews, ratings and images to find us the best hamburger nearby our office in Amsterdam. This talk will be about that quest, and about the (overkill of) tools that we used for this purpose: Scrapy, ElasticSearch, Google's Vision API and BigQuery.
##### Abstract
With the goal of making a data-driven choice for our next hamburger lunch, we gathered information about restaurants in Amsterdam from a review website. The data contains reviews, ratings, and pictures. In this talk we will examine and combine these three different pieces of information to make our decision.  In the talk, I will demonstrate how to collect data from the internet using web scraping package Scrapy. To give it a bit more of an engineering feel, we will run the scraper from a Docker container on Google cloud and push the collected data into BigQuery. I will then show how you can easily retrieve data from BigQuery with Pandas, and we will analyse the ratings. Sentiment in review texts is quite easily extracted using ElasticSearch. And the collected images are send to the Google Vision API in order to classify if there's a nice and juicy burger on it. Although all this might sound like an overkill of tools, the point here is to experiment and learn. There are just too many toys out there to play with!

#### [Hitting the gym: controlling traffic with Reinforcement Learning](https://pydata.org/warsaw2018/schedule/presentation/8/) - Steven Nooijen
##### Description
Finally a good real-life use case for Reinforcement Learning (RL): traffic control! In this talk I will show you how we hooked up traffic simulation software to Python and how we built our own custom `gym` environment to run RL experiments with `keras-rl` for a simple 4-way intersection.
##### Abstract
Traffic congestion causes unnecessary delay, pollution and increased fuel consumption. Learning-based traffic control algorithms have recently been explored as an alternative to existing traffic control logics, which are often manually configured and therefore not optimal. In this talk, I will demo how we trained a Reinforcement Learning (RL) algorithm for traffic control and share with you some of our learnings and best practices in doing so.  The session will start with a conceptual understanding of Reinforcement Learning and how algorithms like (Deep) Q-learning work. I will then explain why this is relevant for traffic control, after which I will zoom in on OpenAI and how to build your own custom `gym` environment. With such an environment you can easily tap into existing `keras-rl` algorithms, which will speed up your RL project significantly.  In our case, connecting the `gym` to the traffic simulation software wasn't trivial. Therefore, there is also a short note on the use of `multiprocessing` and blocking queues to enable the reinforcement learning agent to gain control of the simulation software.

#### [Step by step face swap.](https://pydata.org/warsaw2018/schedule/presentation/24/) - Sylwester Brzęczkowski
##### Description
I will present how to implement a face swap. We will go step by step from simple “copy & paste” face on another image to fully functioning and nice looking face swap.
##### Abstract
Some people think Computer Vision is magical... well it's not. It's just math, sweat and software engineering. In this talk, I will present how to implement a face swap. We will go step by step from simple “copy & paste” face on another image to fully functioning and nice looking face swap. Everything is backed with live examples right there on the stage. The talk will give you a general understanding of what kind of operations are performed under the hood of face swapping applications. You will know how these operations work and how exactly they improve the solution. The talk will resolve the following problems:    
1. Face detection.   
2. Facial landmarks detection.   
3. Simulation of nonlinear image transformation using triangles warps.   
4. Finding such triangles.   
5. Blending 2 images together.   
6. Video stabilization.

#### [Optimizing Deep Neural Network Layer Topology with Delve](https://pydata.org/warsaw2018/schedule/presentation/23/) - Justin Shenk
##### Description
Identifying the number of hidden units in a fully connected layer is considered a heuristically-guided craft. A PyTorch library, Delve, was developed that allows identifying the degree of over-parameterization of a layer, thus guiding architecture selection. The library compares the intrinsic dimensionality of the layer over training, providing the user with live feedback during training.
##### Abstract
* Why is identifying the number of units in a hidden layer hard?   
* How is layer saturation calculated?   
* Eigendecomposition of the latent representations   
* What is the advantage of this approach over alternatives?   
* Live - during training   
* Lightweight PyTorch extension   
* Intuitive - visualized as horizontal bar graph in the terminal with tqdm   
* How do I use [Delve](https://github.com/justinshenk/delve)?   
* Live demo
