<link rel="stylesheet" type="text/css" href="assets/common.css">
<script src="https://d3js.org/d3.v4.min.js"></script>

# Understanding Augmented Recurrent Neural Networks

<div class="byline">
  <div class="authors">
    <div class="author">
      <div class="name">Chris Olah</div>
      <div class="affiliation">Google Brain</div>
    </div>
    <div class="author">
      <div class="name">Shan Carter</div>
      <div class="affiliation">Google Brain</div>
    </div>
  </div>
  <div class="date">
    <div class="month">Jan. 1</div>
    <div class="year">2016</div>
  </div>
  <div class="citation">
    <div>Olah & Carter, 2016</div>
    <div>BibTeX</div> <!-- This should be a link. -->
  </div>
</div>

<script type="text/javascript" async
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

Recurrent neural networks are one of the staples of deep learning, allowing neural networks to work with sequences of data like text, audio and video. They can be used to boil a sequence down into a high-level understanding, to annotate sequences, and even to generate new sequences from scratch!

{{> assets/rnn_basic_rnn.html}}

The basic RNN design struggles with longer sequences, but if you use [LSTM networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), a special kind of RNN, they can even work with these. Such models have been found to be very powerful, achieving remarkable results in many tasks including translation, voice recognition, and image captioning. As a result, recurrent neural networks have become very widespread in the last few years.

As this happened, we’ve seen a growing number of attempts to augment RNNs with new properties. Four directions stand out as particularly exciting:

* *Neural Turing Machines* have external memory that they can read and write to.
* *Attentional Interfaces* allow RNNs to focus on parts of their input.
* *Adaptive Computation Time* allows for varying amounts of computation per step.
* *Neural Programmers* can call functions, building programs as they run.

Individually, these techniques are all potent extensions of RNNs, but the really striking thing is that they can be combined together. My guess is that these "augmented RNNs" will radically extend what deep learning is capable of in the coming years.

### Neural Turing Machines

Neural Turing Machines ([Graves, *et al.*, 2014](https://arxiv.org/pdf/1410.5401v2.pdf)) combine a RNN with an external memory bank. Since vectors are the natural language of neural networks, the memory is arranged as an array of vectors:

<figure class="w-page">
  <!--
  <figcaption>Memory is an array of vectors.</figcaption>
  <figcaption style="top: 100px;">At every time step, the RNN controller can read from and write to this external memory.</figcaption>
  -->
  <div id="rnn-memory">
  {{> assets/rnn_memory.svg}}
  </div>
</figure>

But how does reading and writing work? The challenge is that we want to make them differentiable. In particular, we want to make them differentiable with respect to the location we read from or write to, so that we can learn where to read and write. This is tricky, because memory addresses seem to be fundamentally discrete.

NTMs take a very clever solution to this: every step, they read and write everywhere, just to different extents. As an example, let’s focus on reading. Instead of specifying a single location, the RNN gives “attention distribution” which describe how we spread out the amount we care about different memory positions. As such, the result of the read operation is a weighted sum.

{{> assets/rnn_read.html}}

Similarly, we write everywhere at once to different extents. Again, an attention distribution describes how much we write at every location. We do this by having the new value of a position in memory be a convex combination of the old memory content and the write value, with the position between the two decided by the attention weight.

<figure class="w-page">
  <!--
  <figcaption style="top: 130px;"><b>When writing</b> the RNN reads from everywhere, just to different extents</figcaption>
  -->
  {{> assets/rnn_write.svg}}
</figure>

But how do NTMs distribute their attention over positions in memory? They actually combine together two different attention mechanisms: content-based attention and location-based attention. Content-based attention allows NTMs to search through their memory and move to places that match what they’re looking for, while location-based attention allows relative movement in memory, enabling the NTM to loop.

The addressing process starts with the generating the content-based focus. First, the controller gives a “query” vector, describing what we should focus on. Each memory entry is scored for similarity with the query, using either a dot product or cosine similarity. The scores are then converted into an attention distribution using softmax.

<figure class="w-page">
  <!--
  <figcaption style="top: 50px;">First, the controller gives a query vector, describing what we should focus on. Each memory entry is scored for similarity with the query.</figcaption>
  <figcaption style="top: 250px;">The scores are then converted into an attention distribution using softmax.</figcaption>
  <figcaption style="top: 410px;">Next, we interpolate the attention from the previous time step. </figcaption>
  <figcaption style="top: 530px;">We convolve the attention with a shift filter — this allows the controller to move relative to the position it is anchored to.</figcaption>
  <figcaption style="top: 700px;">Finally, we sharpen the attention distribution to concentrate our focus. This final attention distribution is fed to the read or write operation.</figcaption>
  -->
  {{> assets/rnn_write_detail.svg}}
</figure>

This capability to read and write allows NTMs to perform many simple algorithms, previously beyond neural networks. For example, they can learn to store a sequence in memory, and then loop over it, repeating it back. As they do this, we can watch where they read and write, to better understand what they're doing:

<figure class="side-saddle-right">
  <figcaption style="top: 75px;">See more experiments in [Graves, *et al.*, 2014](https://arxiv.org/pdf/1410.5401v2.pdf). This figure is based on the Repeat Copy experiment.</figcaption>
  <img src="assets/NTM-Copy-ReadWrite.svg" style="width:70%; margin-left:17%; padding-top:20px; padding-bottom:17px;"></img>
</figure>


They can also learn to mimic a lookup table, or even learn to sort numbers (although they kind of cheat)! On the other hand, they still can’t do many basic things, like add or multiply numbers.

Since the original NTM paper, there's been a number of exciting papers exploring similar directions. The Neural GPU ([Kaiser & Sutskever, 2015](http://arxiv.org/pdf/1511.08228v3.pdf)) overcomes the NTM's inability to add and multiply numbers.  [Zaremba & Sutskever, 2016](http://arxiv.org/pdf/1505.00521.pdf) train NTMs using reinforcement learning instead of the differentiable read/writes used by the original. Neural Random Access Machines ([Kurach *et al.*, 2015]( http://arxiv.org/pdf/1511.06392.pdf)) work based on pointers. Some papers have explored differntiable data structures, like stacks and queues ([Grefenstette *et al*. 2015](http://papers.nips.cc/paper/5648-learning-to-transduce-with-unbounded-memory.pdf); [Joulin & Mikolov, 2015](https://arxiv.org/pdf/1503.01007v4.pdf)). And memory networks ([Weston *et al.*, 2014](http://arxiv.org/abs/1410.3916); [Kumar *et al.*, 2015](http://arxiv.org/abs/1506.07285)) are another approach to attacking similar problems.

*(TODO: make above more readable)*

### Attentional Interfaces

When I’m translating a sentence, I pay special attention to the word I’m presently translating. When I’m transcribing an audio recording, I listen carefully to the segment I’m actively writing down. And if you ask me to describe the room I’m sitting in, I’ll glance around at the objects I’m describing as I do so.

Neural networks can achieve this same behavior using *attention*, focusing on part of a subset of the information they're given. For example, an RNN can attend over the output of another RNN. At every time step, it focuses on different positions in the other RNN.

We'd like attention to be differentiable, so that we can learn where to focus. To do this, we use the same trick Neural Turing Machiens use: we focus everywhere, just to different extents.

<figure class="side-saddle-right w-page">
  <figcaption>We use the same trick Neural Turing Machines use: we focus everywhere, just to different extents.</figcaption>
  {{> assets/rnn_attentional_01.svg}}
</figure>

The attention distribution is usually generated with content-based attention. The attending RNN generates a query describing what it wants to focus on. Each item is dot producted with the query to produce a score, describing how welll it matches the query. The scores are fed into a softmax to create the attention distribution.

<figure class="w-page">
  {{> assets/rnn_attentional_02.svg}}
</figure>

Attention between two RNNs can be used in translation. A traditional sequence-to-sequence model has to boil the entire input down into a single vector and then expands it back out. Attention avoids this by allowing the RNN processing the input to pass along information about each word it sees, and then for the RNN generating the output to focus on words as they become relevant.

<figure class="side-saddle-right">
  <figcaption style="top: 170px;">Figure from<br> [Bahdanau, *et al.* 2014](https://arxiv.org/pdf/1409.0473.pdf)</figcaption>
  <img src="assets/old-rnn-attention-vis2.png" style="width:50%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>
</figure>

This kind of attention between RNNs can also be used in voice recognition. This allows one RNN to process the audio, and then another to skim through it, focusing on the relevant parts to generate a transcript.

<figure class="side-saddle-right">
<figcaption style="top: 150px;">Figure from<br> [Chan, *et al.* 2015](https://arxiv.org/pdf/1508.01211.pdf)</figcaption>
<img src="assets/old-rnn-attention-vis1.png" style="width:60%; margin-left:18%; padding-top:20px; padding-bottom:17px;"></img>
</figure>

Attention can also be used on the interface between a convolutional neural network and an RNN. This allows the RNN to look at different position of an image every step.

(TODO: last sentence awkward)

<img src="assets/old-rnn-attention-conv.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

One popular use of this kind of attention is for image captioning. First, a conv net processes the image, extracting high-level features. Then an RNN runs, generating a description of the image. As it generates each word in the description, the RNN focuses on the conv nets interpretation of the relevant parts of the image. We can explicitly visualize this:

<figure class="side-saddle-right">
  <figcaption style="top: 60px;">Figure from<br> [Xu, *et al.*, 2015](https://arxiv.org/pdf/1502.03044.pdf)</figcaption>
  <img src="assets/ShowAttendTell.png" style="width:90%; margin-left:5%; padding-top:20px; padding-bottom:17px;">
</figure>

More broadly, attentional interfaces can be used whenever


... **TODO**

### Adaptive Computation Time

Standard RNNs do the same amount of computation each time step. This seems unintuitive -- surely, one should think more when things are hard? -- and limits RNNs to doing $O(n)$ operations. Adaptive Computation Time ([Graves, 2016](https://arxiv.org/pdf/1603.08983v4.pdf)), or ACT, is a way for RNNs to do variable amounts of computation each step.

The big picture idea is simple: allow the RNN to do multiple steps of computation for each time step.

<figure class="side-saddle-right w-page">
  {{> assets/rnn_adaptive_01.svg}}
</figure>

In order for the network to learn how many steps to do, we want the number of steps to be differentiable. We achieve this with the same trick we used before: we consider an attention distribution over computation steps, and have the output be a weigthed combination of the states at each step. We also want the RNN to know when it has moved on to a new step, so we set a special bit on the input for the first computation step of each time step.

There are a few more details, which were left out in the previous diagram. Here's a complete diagram of a time step with three computation steps.

<figure class="side-saddle-right w-page">
  {{> assets/rnn_adaptive_02.svg}}
</figure>

That's a bit complicated, so let's work through it step by step. At a high-level, we're still running the RNN and outputting a weighted combination of the states:

<img src="assets/old-act-step1.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

The weight for each step is determined by a "halting neuron". It's a sigmoid neuron that looks at the RNN state and gives an halting weight, which we can think of as the probability that we should stop at that step.

<img src="assets/old-act-step2.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

We have a total budget for the halting weights of 1, so we track that budget along the top. When it gets to less than epsilon, we stop.

<img src="assets/old-act-step3.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

When we stop, might have some left over halting budget because we stop when it gets to less than epsilon. What should we do with it? Technically, it's being given to future steps but we don't want to compute those, so we attribute it to the last step.

<img src="assets/old-act-step4.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

### Neural Programmer

Neural nets are excellent at many tasks, but they also struggle to do some basic things like arithmetic, which are trivial in normal approaches to computing. It would be really nice to have a way to fuse neural nets with normal programming, and get the best of both worlds.

The neural programmer ([Neelakantan, *et al.*, 2015](http://arxiv.org/abs/1511.04834)) is one approach to this. It learns to create programs in order to solve a task. In fact, it learns to generate such programs *without needing examples of correct programs*. It discovers how to produce programs as a means to the end of accomplishing some task.

The actual model in the paper answers questions about tables by generating SQL-like programs to query the table. However, there are a number of details here that make it a bit complicated, so let's start by imagining a slightly simpler program, which is given an arithmetic expression and generates a program to evaluate it.

The program is a sequence of operations. Each operation is defined to operate on the output of past operations. So an operation might be something like "add the output of the operation 2 steps ago and the output of the operation 1 step ago." It's more like a unix pipe than a program with variables being assigned and read from.

<img src="assets/old-np?.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

The program is generated one operation at a time by a controller RNN. At each step, the controller RNN outputs a probability distribution for what the next operation should be. For example, we might be pretty sure we want to perform addition at the first time step, then have a hard time deciding whether we should multiply or divide at the second step, and so on...

<img src="assets/old-np1.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

The resulting distribution over operations  can now be evaluated.

Instead of running a single operation at each step, we use our usual trick of running all of them, and then average the outputs together, weighted by the probability we ran that operation.

<img src="assets/old-np2.png" style="width:60%; margin-left:20%; padding-top:20px; padding-bottom:17px;"></img>

As long as we can define derivatives through the operations, the program's output is differentiable with respect to the probabilities. We can then define a loss, and train the neural net to produce programs that give the correct answer. In this way, the Neural Programmer learns to produce programs without examples of good programs. The only supervision is the answer the program should produce.

That's the core idea of Neural Programmer, but the version in the paper answers questions about tables, rather than arithmetic expressions. There's a few additional neat tricks:

* **Multiple Types:** Many of the operations in the Neural Programmer deal with types other than scalar numbers. Some operations output selections of table columns or selections of cells. (To allow us to backprop through the selecting things and average selections, we allow things to be selected to different extents, with 0 as unselected and 1 as fully selected.) Only outputs of the same type get merged together. This is part way to having a type system.

* **Referencing Inputs:** The neural programmer needs to answer questions like "How many cities have a population greater than 1,000,000?" given a table of cities with a population column. To facilitate this, some operations allow the network to reference constants in the question they're answering, or the names of columns. This referencing happens by attention, in the style of pointer networks (discussed above). For example, in order to use the *Greater* operation, the controller must select a value that table entries are greater than; instead of using a previous scalar value it's computed, it has the controller select a value in the question using attention. The exciting thing about this is that, while it's being used in a very limited way, it shows that pointer network-style attention can be used enable a kind of variable.


### Conclusion

(TOD: add heirachal memory paper)


Acknowledgments:

Maithra, Dario, Natalie, Ian
