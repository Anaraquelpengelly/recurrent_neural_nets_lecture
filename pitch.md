
---

## Recurrent Neural Networks
14/03/2022

[Ana Pengelly, PhD](http://anapengelly.info/)



---

#### Course structure

<ol style="color:black;font-size:40px;line-height:1.7;">
<li>What are recurrent Neural Networks?</li> 

<li>Different types of RNNs</li>

<li>What is Back-propagation Through Time?</li>

<li>Long Short Term Memory (LSTM) networks</li>

<li>Applications of RNNs</li>
</ol>



---

### Before we start, a survey...

<img title="mentimeter QR start" src=images/mentimeter_qr_code_0.png  width="300" height="300">



---

### Survey results

<div style='position: relative; padding-bottom: 56.25%; padding-top: 35px; height: 0; overflow: hidden;'><iframe sandbox='allow-scripts allow-same-origin allow-presentation' allowfullscreen='true' allowtransparency='true' frameborder='0' height='315' src='https://www.mentimeter.com/embed/62653c44e847b4f89e84316fa6de53d8/80f0fc3888d9' style='position: absolute; top: 0; left: 0; width: 100%; height: 100%;' width='420'></iframe></div>



---

#### Reminder: what are Neural Networks?

<img style="width:auto; height:250px;" title="perceptron" src=images/perceptron.png>

Basic structure of a perceptron. 



---

#### Reminder: what are Neural Networks?

<img style="width:auto; height:500px;" title="feedforward_neural_net" src=images/feed_forward_neural_net.jpeg>
Basic structure of a neural network. 



---

### What are Recurrent Neural Networks?



---

#### What are Recurrent Neural Networks?
##### Sequence data

<img style="width:auto; height:250px;" title="DNA" src= images/genomic_data.png> <!-- .element: class="fragment" data-fragment-index="1" -->
<img style="width:auto; height:250px;" title="ECG" src= images/ECG.png><!-- .element: class="fragment" data-fragment-index="2" -->
<img style="width:auto; height:35px;" title="text" src= images/text.png><!-- .element: class="fragment" data-fragment-index="3" -->
<img style="width:auto; height:250px;" title="STOCKS" src= images/stocks.png><!-- .element: class="fragment" data-fragment-index="4" -->



---
#### Introduction to recurrent Neural Networks

* <p style="color:black;font-size:24px;line-height:1.25;">RNNs are types of neural networks specialized in processing <b>sequential or time series data.</b></p>

* <p style="color:black;font-size:24px;line-height:1.25;">RNNs are commonly used for <b>ordinal or temporal problems</b> such as language translation, natural language processing (NLP), speech recognition, they are incorporated into popular applications such as Siri, voice search and Google translate.</p>

* <p style="color:black;font-size:24px;line-height:1.25;"><b>Sequence data is characterized by long and variable sequence size</b>, e.g. a word or a sentence have different length.</p>
* <p style="color:black;font-size:24px;line-height:1.25;">Traditional deep networks with separate parameters for each neuron could not generalize to unseen data in a longer sequence.</p>

<img style="width:auto; height:200px;" title="neural_net" src= images/simple_rnn.png>



---

#### Differences between feed-forward neural networks and recurrent Neural Networks

<img style="width:auto; height:300px;" title="neural_net" src= images/rnn_vs_nn.png>

<ul>
<p style="color:black;font-size:28px;line-height:1.25;"> rNNs are different from feed-forward NN by their <b>'memory'</b> as they take <b>information from prior inputs to influence the current input and output</b>. While <b>traditional deep neural networks assume that inputs and outputs are independent of each other</b>, the output of recurrent neural networks depend on the prior elements within the sequence.</p>
</ul>



---

#### RNN architecture

##### Folded RNN

<img style="width:auto; height:300px;" title="neural_net" src= images/unrolled_rNN_0.png>
<ul>
<p style="color:black;font-size:28px;line-height:1.25;text-align=left;">The <b>recurrent arrow</b> is modelling the <b>influence that the current value of the input layer outputs can have on its future values</b>. In this graph recurrent connections are between hidden layers. We can unroll this graph to visualise each state in time.
</p>
</ul>



---

#### RNN architecture

##### Unfolded RNN

<img style="width:auto; height:400px;" title="neural_net" src= images/unrolled_rNN.png>

<p style="color:black;font-size:18px;line-height:1.0;"> Note: The sequence is not necessarily temporal, it could represent a position in a sentence.
</p>



---

#### RNN equations
##### Reminder - equation
<img style="width:auto; height:400px;" title="n_n_eq" src= images/nn_eq.png>

<img style="width:auto; height:50px;" title="n_n_eq" src= images/eq_0.svg>



---

#### RNN equations
##### recurrent network equations 
###### Single layer example
<img style="width:auto; height:230px;" title="n_n_eq" src= images/unrolled_rNN_alone.png>
<p><img style="width:auto; height:25px;" title="rnn_eq" src= images/eq_1.svg></p>
<ul style="color:black;font-size:18px;line-height:1.25;"><p> Recurrent networks share parameters across each layer of the network. While feed-forward networks have different weights across each node, recurrent neural networks share the same weight parameter within each layer of the network.</p>
<p> Many-to-many recurrent network: the number of outputs is the same as the number of inputs (example: translation).</p>
</ul>



---

#### RNNs from Scratch
```Python
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        #initialise weight matrices:
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        #Initialise hidden state to zeros:
        self.h = tf.zeros([rnn_units, 1])
    def call(self, x):
        #update the hidden state
        self.h = tf.math.tanh(self.W_hh * self.W_xh + self.W_xh * x)
        #compute the output
        output = self.W_hy * self.h
        #return the current output and hidden state
        return output, self.h
```



---

#### RNNs with Tensorflow 

``` bash
$pip install tensorflow
```

``` Python
import tensorflow as tf
tf.keras.layers.SimpleRNN(rnn_units)
```



---

### Different types of RNNs



---

#### One-to-one

<img style="width:auto; height:400px;" title="rnn_eq" src= images/one_to_one.png>

<p style="color:black;font-size:24px;line-height:1.0;"><em>Binary Classification.</em></p>



---

#### One-to-many

<img style="width:auto; height:400px;" title="rnn_eq" src= images/one_to_many.png>

<p style="color:black;font-size:24px;line-height:1.0;"><em>Text generation.</em></p>



---

#### Many-to-one

<img style="width:auto; height:400px;" title="rnn_eq" src= images/many_to_one.png>

<p style="color:black;font-size:24px;line-height:1.0;"><em>Sentiment classification.</em></p>



---

#### Many-to-many

<img style="width:auto; height:400px;" title="rnn_eq" src= images/many_to_many.png>

<p style="color:black;font-size:24px;line-height:1.0;"><em>Translation.</em></p>
<p style="color:black;font-size:24px;line-height:1.0;"><em>Forecasting.</em></p>



---

### What is Back Propagation Through Time (BPTT)?



---

#### Reminder: Back Propagation in Feed-forward models

<img style="width:auto; height:300px;" title="backprop_nn" src= images/backprop_feedforward.png>

<p style="color:black;font-size:28px;line-height:1.25;"> 
Back propagation algorithm:</p>
<ol style="color:black;font-size:28px;line-height:1.25;">
<li>Get the derivative (gradient) of the loss with respect to each parameter,</li>
<li>Shift parameters in order to minimise the loss.</li>
</ol>



---

#### RNNs: Back Propagation Through Time (BPTT)

<img style="width:auto; height:400px;" title="backprop_nn" src= images/BPTT_diagram.png>



---

#### Back Propagation Through Time (BPTT)

<img style="width:auto; height:150px;" title="rnn_eq" src= images/BPTT_simple.png>

<ul>

<li><p style="color:black;font-size:24px;line-height:1.25;"> 
We can compute the gradient and use the common optimizers.</p></li>

<li><p style="color:black;font-size:24px;line-height:1.25;"> 
Recurrent neural networks leverage back propagation through time (BPTT) algorithm to determine the gradients, which is slightly different from traditional back propagation. </p></li>

<li><p style="color:black;font-size:24px;line-height:1.25;"> 
The principles of BPTT are the same as traditional back propagation, where the model trains itself by calculating errors from its output layer to its input layer. These calculations allow us to adjust and fit the parameters of the model appropriately. </p></li>

<li><p style="color:black;font-size:24px;line-height:1.25;"> 
BPTT differs from the traditional approach in that BPTT sums errors at each time step whereas feed-forward networks do not need to sum errors as they do not share parameters across each layer. This makes RNNs very computationally expensive.</p>
</ul></li>



---

#### Optimizing RNNs
##### Problems with BPTT
<img style="width:auto; height:130px;" title="rnn_eq" src= images/BPTT_simple.png>
<p style="color:black;font-size:20px;line-height:1.25;">Here, computing the gradient with regards to <em>h<sub>0</sub></em> involves many factors of <em>W<sub>hh</sub></em> and repeated gradient computations!</p>
<ul style="color:black;font-size:24px;line-height:1.25;">
<li>
One problem with RNNs is the long time-dependency: because every output depends on all the previous inputs, long inputs will generate long chains of dependencies (function compositions).</li>
<li>This leads to two main problems: exploding gradients and vanishing gradients. These issues are defined by the size of the gradient, which is the slope of the loss function along the error curve.</li>
<li><b>Vanishing gradients</b> happen when the <b>gradient is too small</b>, and it continues to become smaller, updating the weight parameters until they become insignificant (0). When that occurs, the algorithm is no longer learning.</li>
<li><b>Exploding gradients</b> occur when the <b>gradient is too large</b>, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. </li>
</ul>



---

#### Optimizing RNNs
##### Some solutions to BPTT exploding gradients

<img style="width:auto; height:200px;" title="rnn_eq" src= images/echo_state_network.jpeg>
<ul>
<li><p style="color:black;font-size:28px;line-height:1.25;"> 
For exploding gradients, gradient clipping (i.e. re-scaling: if the gradient gets too large, we rescale it to keep it small). </p></li>
<li><p style="color:black;font-size:28px;line-height:1.25;">Reduce the number of hidden layers within the neural network, eliminating some of the complexity in the RNN model.</p></li>
<li><p style="color:black;font-size:28px;line-height:1.25;">Using parallel networks with different time scales (coarse time scales are achieved by skipping connections).</p></li>
<li><p style="color:black;font-size:28px;line-height:1.25;">Fix recurrent weights and only learn the output weights (echo state networks).</p></li>
</ul>



---

#### Optimizing RNNs
##### Some solutions to BPTT vanishing gradients

<ul>
<li><p style="color:black;font-size:30px;line-height:1.25;"> 
Change the activation function to functions like ReLU. </p></li>
<li><p style="color:black;font-size:30px;line-height:1.25;">Weight initialisation to the identity matrix and thereby initialise the biases to zero.</p></li>
<li><p style="color:black;font-size:30px;line-height:1.25;">Use gated cells to control what information is passed through, i.e. Long Short Term Memory (LSTM) networks and Gated Recurrent Unit (GRU) networks.</p></li>
</ul>



---

### Long Short Term Memory (LSTM) Networks



---

#### Variant RNN architectures
##### Long short-term memory (LSTM)

<img style="width:auto; height:300px;" title="rnn_layer" src= images/simple_rnn_layer.png>

<img style="width:auto; height:300px;" title="lstm_layer" src= images/lstm_layer.png>

<p style="color:black;font-size:32px;line-height:1.25;">LSTM modules contain blocks that control information flow. They are able to track information through many time-steps.</p>



---

#### Long short-term memory (LSTM)

<img style="width:auto; height:400px;" title="lstm_layer" src= images/lstm_layer.png>

<p style="color:black;font-size:32px;line-height:1.25;"> Information can be added or removed through gates, which let information through via for example sigmoid neural net layers and pointwise multiplication.</p>



---

#### Long short-term memory (LSTM)
##### How do they work?
<p style="color:black;font-size:32px;line-height:1.0;"><b>1) Forget  2) Store  3) Update  4) Output </b> </p>

<img style="width:auto; height:400px;" title="lstm_main" src= images/LSTM_main.png>



---

#### Long short-term memory (LSTM)
<p style="color:black;font-size:32px;line-height:1.0;"><b>1) Forget </b>  2) Store  3) Update  4) Output  </p>

<img style="width:auto; height:400px;" title="lstm_main" src= images/LSTM_forget.png>

<p style="color:black;font-size:32px;line-height:1.0;">Irrelevant parts of information from the previous state are forgotten.</p>



---

#### Long short-term memory (LSTM)
<p style="color:black;font-size:32px;line-height:1.0;">1) Forget <b>2) Store</b>   3) Update  4) Output  </p>

<img style="width:auto; height:400px;" title="lstm_main" src= images/LSTM_store.png>

<p style="color:black;font-size:32px;line-height:1.0;">Relevant new information from the 'current' state is saved.</p>



---

#### Long short-term memory (LSTM)
<p style="color:black;font-size:32px;line-height:1.0;">1) Forget 2) Store  <b>3) Update</b>   4) Output  </p>

<img style="width:auto; height:400px;" title="lstm_main" src= images/LSTM_update.png>

<p style="color:black;font-size:32px;line-height:1.0;">Cell state values are selectively updated.</p>



---

#### Long short-term memory (LSTM)
<p style="color:black;font-size:32px;line-height:1.0;">1) Forget 2) Store  3) Update   <b>4) Output </b></p>

<img style="width:auto; height:400px;" title="lstm_main" src= images/LSTM_output.png>

<p style="color:black;font-size:32px;line-height:1.25;">The output gate selects what information is sent to the next time step.</p>



---

#### LSTM Gradient flow

<img style="width:auto; height:400px;" title="lstm_main" src= images/backprop_LSTM.png>



---

#### Long short-term memory (LSTM) with Tensorflow

```Python
tf.keras.layers.LSTM(num_units)
```



---

#### LSTM take home points
<ul>
<li><p style="color:black;font-size:32px;line-height:1.1;">Maintain a separate cell state through time, which is different from the output.</p></li>

<li><p style="color:black;font-size:32px;line-height:1.1;">Use gates to control the flow of information: forget gates removes irrelevant information, important information from the current input is stored, the cell state is selectively updated, and the output gate returns a filtered version fo the cell state.</p></li>
<li><p style="color:black;font-size:32px;line-height:1.1;">Back-propagation through time with uninterrupted gradient flow.</p></li>
</ul>



---

#### Another Variant RNN architecture
##### Gated Recurrent Units (GRU) RNN

<p style="color:black;font-size:32px;line-height:1.0;">RNN architecture using gates, like LSTM but simpler.</p> 

<p style="color:black;font-size:32px;line-height:1.0;">There are two types of gates:
<ul style="color:black;font-size:32px;line-height:1.0;"> 
<li>reset (r)</li>
<li>update (z)</li>
</ul></p>

<img style="width:auto; height:400px;" title="rnn_eq" src= images/GRU.png>



---

### Applications of RNNs



---

#### Sentiment classification

<p style="color:black;font-size:32px;line-height:1.25;"><b>Input:</b> sequence of words (i.e. a Tweet)</p>

<p style="color:black;font-size:32px;line-height:1.25;"><b>Output:</b> probability of having a positive (or negative) sentiment</p>

```Python
loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)
```

<img style="width:auto; height:400px;" title="rnn_eq" src= images/sentiment.png>



---

#### Sentiment classification
<p style="color:black;font-size:32px;line-height:1.25;"><b>Input:</b> sequence of words (i.e. a Tweet)</p>

<p style="color:black;font-size:32px;line-height:1.25;"><b>Output:</b> probability of having a positive (or negative) sentiment</p>

<img style="width:auto; height:300px;" title="rnn_eq" src= images/sentiment.png> <img style="width:auto; height:300px;" title="rnn_eq" src= images/positive_sentiment.png>



---

#### Machine translation (i.e. Google translate)

<img style="width:auto; height:300px;" title="rnn_eq" src= images/translation.png>



---

#### Machine translation (i.e. Google translate)

<p style="color:black;font-size:32px;line-height:1.25;"><b>Problems:</b> encoding bottleneck (words may have different orders in a sentence in different languages), not long memory and no parallelisation, so slow!</p>
  
<img style="width:auto; height:300px;" title="rnn_eq" src= images/translation.png>



---

#### Machine translation (i.e. Google translate)

<p style="color:black;font-size:32px;line-height:1.25;">Attention mechanisms in neural networks provide <b>learnable memory access</b></p>

<img style="width:auto; height:300px;" title="rnn_eq" src= images/attention_translation_new.png>



---

#### Machine translation (i.e. Google translate)

<p style="color:black;font-size:32px;line-height:1.25;"> The current generation of Transformer models are based on self-attention mechanisms. If you are interested have a look at the reference <a href="https://arxiv.org/abs/1706.03762">here.</a></p>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).

###### Data: 

<ul>
<li><p style="color:black;font-size:32px;line-height:1.0;">Covid-19 hospitalisations (ONS)</p><img style="width:auto; height:50px;" title="rnn_eq" src= images/ons_logo.svg> </li>
<li><p style="color:black;font-size:32px;line-height:1.0;">Visa card transaction volume (Visa)</p><img style="width:auto; height:50px;" title="rnn_eq" src= images/Visa_logo.png></li>
<li><p style="color:black;font-size:32px;line-height:1.0;">Total international arrivals in the UK (Home Office).</p></li>
</ul>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).

###### Data: 
<object data="images/hosp_transactions_london.html" style = "width:1000px; height:500px;"></object>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).

###### Data:
<object data="images/hosp_arrivals_london.html" style = "width:1000px; height:500px;"></object>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).

###### Data preprocessing: 
<ul>
<li><p style="color:black;font-size:32px;line-height:1.0;"> Scaling: Min-max scaling between 0 and 1.
</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;">Transformation of time series to supervised data (differencing: generating a shift of one day before as the predictors).</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;">Prediction lag = 1, 7, 14, 28 days. </p></li>
</ul>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).
###### Model: 
<img style="width:auto; height:250px;" title="rnn_eq" src= images/hosp_model.png>
<p style="color:black;font-size:20px;line-height:1.0;">Dense layer implements:
</p>
<ul style="color:black;font-size:20px;line-height:1.7;">
<img src="http://latex.codecogs.com/png.latex?\huge&space;\dpi{110}&space;\mathit{y}&space;=&space;\mathit{activation}(\mathit{WX}&plus;\mathit{b})" title="http://latex.codecogs.com/png.latex?\huge \dpi{110} \mathit{y} = \mathit{activation}(\mathit{WX}+\mathit{b})"/>
<li><em>y</em> is the output</li>
<li><em>W</em> is the weight matrix</li>
<li><em>X</em> the input matrix</li>



---

#### Time series prediction (i.e. Predicting Covid-19 hospitalisations with LSTM networks).

###### Results:

<img style="width:auto; height:500px;" title="rnn_eq" src= images/covid_19_hosp.png>



---

#### Self Driving car trajectory

<iframe width="560" height="315" src="https://www.youtube.com/embed/NG_O4RyQqGE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



---

#### Test your knowledge

<img title="mentimeter QR end" src= images/mentimeter_qr_code_end.png width='300' height='300'>



---

#### Test your knowledge : results

<div style='position: relative; padding-bottom: 56.25%; padding-top: 35px; height: 0; overflow: hidden;'><iframe sandbox='allow-scripts allow-same-origin allow-presentation' allowfullscreen='true' allowtransparency='true' frameborder='0' height='315' src='https://www.mentimeter.com/embed/33a9bb28cf223cf62de00f7cc0a5737e/58b102e848dc' style='position: absolute; top: 0; left: 0; width: 100%; height: 100%;' width='420'></iframe></div>



---

#### RNNs summary

<ul>
<li><p style="color:black;font-size:32px;line-height:1.0;text-align:left;">RNNs are mainly used to model sequence data.</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;text-align:left;">They model sequences via a recurrence relation.</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;text-align:left;">RNNs are trained using Back Propagation Through Time (BPTT).</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;text-align:left;">LSTMs are a type of RNN that uses gated cells to effectively model long-term dependencies.</p></li>
<li><p style="color:black;font-size:32px;line-height:1.0;text-align:left;">RNNs are used for music generation, classification, machine translation to name a few.</p>
</ul></li>



---

#### Please check the references & further reading</a> section for more! 
<p style="font-size:100px">&#129321 &#128170</p>	



---

#### References & further reading

<ul style="color:black;font-size:32px;line-height:1.5;text-align:left;">
<li><a href=http://introtodeeplearning.com/2021/slides/6S191_MIT_DeepLearning_L2.pdf> MIT Introduction to Deep Learning: Deep Sequence Modeling</a></li>
<li>Machine Learning Recurrent Neural Network lecture by Dragana Vuckovic.</li>
<li><a href=https://colah.github.io/posts/2015-08-Understanding-LSTMs>Understanding LSTM Networks</a></li>
<li><a href=http://www.bioinf.jku.at/publications/older/2604.pdf>Original publication of the Long Short-Term Memory (LSTM)</a></li>
<li><a href=https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf>Recursive deep models for Semantic Compositionality Over a Sentiment Treebank</a></li>
<li><a href=https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf>Sequence to Sequence Learning with Neural Networks (LSTM)</a></li>
<li><a href=https://arxiv.org/pdf/1409.0473.pdf>Neural Machine translation and the introduction of attention</a></li>
<li><a href=https://arxiv.org/abs/1706.03762>Attention is all you need (original publication of Transformer models)</a></li>



---