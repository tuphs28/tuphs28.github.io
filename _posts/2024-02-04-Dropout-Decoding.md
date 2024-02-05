---
title: "BadTorch Part 3: Language Model Decoding and Dropout"
tags:
    - BadTorch
    - Deep Learning
---

In my spare moments for the past few weeks I have been finishing off my "BadTorch" project of re-creating Pytorch's module-level API for recurrent language models. In this post I will outline two specific pieces of relevant functionality that I have implemented - dropout and decoding. All code can be found [here](https://github.com/tuphs28/BadTorch). 

### Dropout

As I mentioned in a prior post, my original motivation for BadTorch was realising that I wasn't sure how to implement variational dropout. Thus, something I wanted to accomplish as part of this project was doing exactly this.

I therefore implemented both standard dropout and variational dropout for reccurent models. Without going into too much detail here (read [this](https://tuphs28.github.io/Dropout-In-Recurrent-Models/) post if interested), at training time these two forms of dropout work as follows:
- In standard dropout, at each new time step of a recurrent model, we randomly mask out (i.e. set to 0) with probability \\(p\\) different weights of the model
- Varaiational dropout is similar, except that rather than randomly masking different weights of the model at each step we mask the same weights of the model at each step for each sequence

The idea here is that by randomly masking out model weights, we encourage the model to learn more robust features that generalise better outside of the training set. With the task of language modelling, this means we encourage the model to learn representations that allow it to better model a wide array of text instead of just learning representations that it allow it to effectively memorise its training corpus. At test time, we then multiply network weights by \\(1-p\\) to keep outputs the same in expectation.

Implementing these forms of dropout requires using a dropout mask. This is just the mask that randomly zeros out elements of tensors. However, we can choose whether to implement dropout by applying this mask to the hidden states of a reccurent model or by applying it to the model weights directly. Both choices are valid in the sense that zeroing out a hidden unit is equivalent to zeroing out the corresponding rows of all weight matrices that act on it. 

Now, for basic RNNs either choice ends up implementing exactly the same idea (e.g. dropout as applied to whole rows of weight matrices). However, it should be noted that for more complex models like GRUs and LSTMS whereby multiple weight tensors operate over the same hidden units, applying the mask to the hidden unit is effectively applying the same dropout mask to all weight matrices acting on the same hidden unit. This parameter-tying can be seen as theoretically undesirable (e.g. since we are no longer zeroing out rows of weight tensors in an iid fashion), though it typically doesn't massively impact performance. For a minor boost in efficiency (e.g. to avoid creating additional masks), I chose to apply the mask to the hidden units.

So, how did I actually implement this? The first step was adding dropout functionality to the forward method of wrapper RNN module (i.e. the module that repatedly calls the recurrent cell over an input sequence). A summary of the code for this is shown below. The key things to notice are that (1) if we use "standard" dropout, a new mask is created for each step, (2) if we use variational dropout, the same mask is used for all steps and (3) we have to introduce an attribute to keep track of if we are in train or test mode such that we rescale by \\(1-p\\) when performing inference.

```python
class RNN:
  ....
  def __call__(self, x: torch.Tensor, h_prev: Optionl[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

    ... 

    inputs = x
    for n_layer in range(self.n_layers):

      cell = self.cells[n_layer]

      ...

      if self.dropout == "":
        dropout_masks = []
      elif not training:
        if self.cell_type == "LSTM":
          dropout_masks = [1 - self.dropout_strength for _ in range(3)]
        else:
          dropout_masks = [1 - self.dropout_strength for _ in range(2)]
      elif self.dropout == "variational" and training:
        if self.cell_type == "LSTM":
          dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim, self.hidden_dim]]
        else:
          dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim]]

      for seq_idx in range(seq_length):
        if self.dropout == "standard" and training:
          if self.cell_type == "LSTM":
            dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim, self.hidden_dim]]
          else:
            dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim]]
        x_t = inputs[:, seq_idx, :]
        h_t = cell(x_t, h_t, dropout_masks)
        
        ...

    return h_outputs, h_final
```

The next step is then to use the generated dropout masks in the forward pass at each call of the recurrent cell. I provide the example of the forward method of the GRU cell below, with the idea being very similar for the RNN and LSTM cell (albeit applying a different mask to the hidden and cell state for the LSTM).

```python
class GRU:
  ....
  def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor, dropout_masks: Optional[list] = None) -> torch.Tensor:

    if dropout_masks:
      x_t = x_t * dropout_masks[0] # drop out inputs to the cell
      h_t = h_t * dropout_masks[1] # drop out hidden units

    i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
    i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
    h_prop = torch.tanh(x_t @ self.weight_xh + (h_t * i_f) @ self.weight_hh + (self.bias_h if self.bias_h is not None else 0))
    h_t = i_o * h_t + (1 - i_o) * h_prop

    return  h_t
  ....
```

The final step is then to apply the dropout masks to the final feedforward connection. This was the most annoying part since "standard" dropout means that we end up having to de-parallelise the final linear layer (e.g. since we can no longer just pass all hidden states through concurently and must instead pass them through in the order in which they appear in the input sequence so that we can generate different masks for each timestep). I did this de-parallelisation at the level of thee recurrent language model itself as shown in the overview code below:

```python
class RecurrentLanguageModel:

    def __call__(self, x, h = None):

        embs = self.embedding(x)
        hs, h_final = self.rnn(embs, h, self.training)
        
        ...

        if self.training and self.dropout == "standard": # if using standard dropout, need to generate a new mask for each timestep and so have to sequentially pass timestep logits through linear layer
            logits = torch.zeros(size=(hs.shape[0]*hs.shape[1], self.vocab_size), device=self.device)
            for seq_idx in range(hs.shape[1]):
                seq_hs = hs[:, [seq_idx], :].view(-1, self.hidden_dim).to(self.device)
                seq_logits = self.linear(seq_hs, self.training)
                logits[torch.arange(seq_idx, hs.shape[0]*hs.shape[1], hs.shape[1]), :] = seq_logits
        else:
            hs = hs.contiguous().view(-1, self.hidden_dim).to(self.device)
            logits = self.linear(hs, self.training)

        return logits, h_final
  ....
```

So, did this actually end up having much of an impact in the performance of recurrent language models? Well, I was pretty limited by the amount of compute that I had access to but in my (admitedly incredibly limited experiments) the answer seemed to be "not really". For instance, I trained 3 1-layer LSTMs with model dimensionalities of 1024 for 10,000 iterations on 128-character strings from The Complete Works of Willian Shakespeare using a batch size of 16 (taking ~ 1 hour to train each time). One of these models was trained with variational dropout, one with standard dropout and one with no dropout. The two dropout models were trained using a dropout strength of 0.2. The training and validation loss curves for these models are shown below:

![loss curves](/assets/images/2024/02-04-badtorch/drpt_loss.png)

We can see from these figures that all three models achieve roughly equivalent validation losses meaning that neither form of dropout seems to improve generalisation performance or be better than the other in this set up. There are a few explanations that we could give for these patterns:
- First, it might just be that if I trained the models for longer the dropout models would perform better. This is a plausible story given that dropout tends to make models learn slower (this makes sense - we are adding noise to the training process such that learning the desired signal is somewhat more tricky). Indeed, that this is happening might be consistent with the fact that the training loss for the two dropout models hasn't fallen at the same rate as for the non-dropout model.
- Second, it might be that I haven't played about with the hyperparameters enough and we are just getting stuck at a generically bad optima.
- Finally, it might be that a 1-layer LSTM with a model dimensionality of 1024 simply isn't large enough to overfit to the corpus (for reference a 1-layer LSTM with dimensionality 1024 corresponds to these models have ~8.5M parameters). Relatedly, it might be that using a dropout strength of 0.2 corresponds to too aggresive an application of dropout.

### Greedy Decoding

Another interesting aspect of language modelling is decoding. Decoding refers to how a neural lanuage model generates new text (optionally given some prompt to continue). Decoding, therefore, relates to how language models use the distribution over output symbols (i.e. characters in character-level language modelling) they learn during training to produce plausible strings of text.

The simplest form of decoding is greedy decoding. In greedy decoding, the language model merely outputs the most likely symbol at each step. I implemented the decoding methods as a method on the Recurrent Langauge Model class, and the following snipet overviews how greedy decoding was implemented:
```python
class RecurrentLanguageModel:
  ...
  def sample(self, ...,  prompt: str = "", sample_length: int = 250, decoding_method: str = "sample", ... ) -> str:
        ...
        elif decoding_method == "greedy": 
            for _ in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                current_idx = prob.view(-1).argmax().view(1,1)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)
        ...
        return sampled_text
```
Now, while greedy decoding does have its place in wider machine learning (e.g. it works relatively well with CTC in basic speech recognition) it ends up performing incredibly poorly in language modelling. This because generating the most likely token at each step will not give rise to a generally likely string of tokens. Instead, generating the most likely token will, in practice, end up just repeating common words and phrases in a globally non-sensical fashion. This is because greedy decoding, by focusing on merely producing the most probable token at each step, ignores the extent to which these tokens are globally likely. This is a problem since likely strings of text involve deviations that are locally unprobable. 

To illustrate the flaws of greedy decoding, I generated an output string (using a random character as the inital prompt) using the 1-layer, 1024-wide LSTM trained with no dropout described above. This output is shown below, and is clearly non-sensical due to the fact it has maximised the extent to which it takes locally probable steps by repeating itself.

> That hath sent for the streets of the streets of the streets  
  That shall be so many a shadow of the streets  
    That shall be so many a shadow of the streets  
      That shall be so many a shadow of the streets  
        That shall be so many a shadow of the streets  
          That shall be so many a shadow of the streets  
            That shall be so many a shadow of the streets  
              That shall be so many a shadow of the streets

### Beam Decoding

So, what went wrong here? Well, as already mentioned, greedy decoding fails since it ignores the fact that taking locally improbable steps can lead to strings that are globally more plausible. This then motivates beam search as an alternative decoding procedure.

The idea of beam search is that we keep track of multiple plausible strings, and iteratively extend these strings and only keep the most plausible extensions. By keeping track of multiple such "beams" we then hope we can judge proposed strings by their global plausibilities, avoiding the problems of beam search. More precisely, decoding with beam search  works as follows:
- Choose a "beam width" k
- Create a set B of beams and insert our prompt into it
- Then, iterate for a set number of steps the following procedure:
  - Create a set of new proposal beams C
  - For each beam in B:
    - Extend the beam with each of the k most plausible tokens
    - For each of these k new proposal beams:
      - If C has less than k proposal beams, add the proposal beam to C
      - Else if C has k proposed beams, add the new beam to C if it is more probable than the least probable beam, and then remove that least probable beam from C
      - Else, discard the proposed beam
  - Set B = C
- Return the most probable beam in B.

The following snippet shows how I implemented beam decoding for recurrent language models. Notice that we have to store the current state of the recurrent hidden state for each proposed beam (unlike beam search in, say, a transformer LM).
```python
class RecurrentLanguageModel:
  ...
  def sample(self, ...,  prompt: str = "", sample_length: int = 250, decoding_method: str = "sample", beam_width: int = 5, ... ) -> str:
        ...
        elif decoding_method == "beam": 
            beams = []
            current_str = "".join([idx_to_char_dict[i] for i in history])
            beams = [
                [current_str, 1, hidden]
            ]
            k = beam_width
            for _ in range(sample_length):
                new_beams = []
                for beam in beams:
                    current_str, prob, hidden = beam
                    current_idx = torch.Tensor([char_to_idx_dict[current_str[-1]]]).int().view(1,1).to(self.device)
                    logits, hidden = self(current_idx, hidden)
                    probs = F.softmax(logits.view(-1), dim=0)
                    for prop_idx, prop_prob in zip(torch.topk(probs, k).indices.view(-1), torch.topk(probs, k).values.view(-1)):
                        new_prob = prob * prop_prob.item()
                        new_str = current_str + idx_to_char_dict[prop_idx.item()]
                        if new_str[-3:].count(" ") > 2: # penalise excessive outputing of spaces (needed due to how text files are formatted on laptop)
                            new_prob *= 0.1**(new_str[-3:].count(" ")-1)
                        if len(new_beams) < k:
                            new_str = current_str + idx_to_char_dict[prop_idx.item()]
                            new_beams.append([new_str, new_prob, hidden])
                        else:
                            sorted_probs = [new_beam[1] for new_beam in new_beams]
                            sorted_probs.sort(reverse=True)
                        if new_prob > sorted_probs[k-1]:
                            new_str = current_str + idx_to_char_dict[prop_idx.item()]
                            for i in range(len(new_beams)):
                                if sorted_probs[k-1] == new_beams[i][1]:
                                    new_beams = new_beams[:i] + [[new_str, new_prob, hidden]] + new_beams[i+1:]
                beams = new_beams
            max_prob = -1
            for beam in beams:
                if beam[1] > max_prob:
                    max_prob = beam[1]
                    sampled_text = beam[0]
        return sampled_text
```

To test whether decoding with beam search actually suceeded in producing more plausible text, I used beam decoding with a beam width of 5 to generate a sample of text using the same character-level LSTM language model as for greedy decoding. The output is shown below:

>   CORIOLANUS. What says the matter?  
      CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?  
          CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?  
          CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?  
          CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?  
          CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?  
          CASSIUS. What says the matter?  
          CLEOPATRA. What says the matter?

So, again, we clearly seem to be facing the same problem as greedy decoding since the model has decided that thre most probable strings are still mere repetitions. Why is this still happening even though we are now tracking multiple plausible hypotheses rather than being purely greedy with respect to model outputs?

The answer to this question is that beam decoding ends up behaving very similarly to greedy decoding. This is because the singuarly most probable output at most steps is signficantly more probable than other outputs (i.e. the output distribution is sharp). Thus, the beams corresponding to these most probable outputs end up dominating all other beams. Hence, beam decoding in practice ends up collapsing into a more computationally expensive form of greedy decoding.

### Sampling and Top-K Decoding

This then leaves open the question of what method we should use to decode LMs. It turns out that a sensible decoding method for producing plausible strings of text is to merely sample from the multinoulli distribution whose probabilities are set equal to the softmax outputs of the model. In other words, we can just sample from the output distribution of the model.

Why is this a better idea than greedy or beam decoding? Well, this is because (as mentioned) what matters is global plausibility and this requires taking locally improbable steps during the decoding process. Crucially, greedy decoding cannot take such steps (by definition), while beam decoding usually ends up discarding any beams that take such steps since the beam probabilities will no longer be sufficient to continue to be tracked. However, when we decode by sampling from the output distribution, we can easily take such steps. The following snippet illustrates my implementation of this idea:

```python
class RecurrentLanguageModel:
  ...
  def sample(self, ...,  prompt: str = "", sample_length: int = 250, decoding_method: str = "sample", ... ) -> str:
        ...
        if decoding_method == "sample": 
            for i in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                current_idx = torch.multinomial(prob, 1).view(1,1).to(self.device)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)
        ...
        return sampled_text
```

However, we can actually improve on this. This is because merely decoding by naive sampling will occasionally lead to the model outputting tokens that are too improbable, and to doing so too frequently. Basically, while we want some exploration of locally unlikely steps, we don't want to take this too far. Thus, a better decoding method id top-k decoding.

Top-k decoding is very similar to the basic sampling approach. In top-k decoding we only sample from the k most probable output tokens at each step such that the distribution we sample from is the (re-normalised) multinoulli with parameters proportional to the k highest output probabilities. This leads to us being able to sample improbable-but-plausible tokens whilst avoiding sampling silly ones. My implementation for this is shown below:

```python
class RecurrentLanguageModel:
  ...
  def sample(self, ...,  prompt: str = "", sample_length: int = 250, decoding_method: str = "sample", k: int = 10, ... ) -> str:
        ...
       elif decoding_method == "top-k":
            for _ in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                top_prob = torch.topk(prob, k)
                current_idx = top_prob.indices[torch.multinomial(top_prob.values, 1)].view(1,1)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)
        ...
        return sampled_text
```

To investigate whether this yielded more plausible strings of text, I used top-k decoding (with k set to 10) to sample from the same LSTM language model as above. The output is shown below:

> FORD. Tell me, for this will speak of fools;  
And make it and steal'd up the rock.  
Thy face and so, in honour, your love war to him.  
KING RICHARD. I'll serve my mind that no sport-mell,  
Be advantage; sickly with your wife and force a horse.  
I am saw a strange to the potent.  

> Enter PERCY,  
Re-enter CLOWN

> CORIOLANUS. Marry's becomes in the bell.  
ROSALIND. This save, I am glad to married all hunter might be true day.

Clearly, this output is far superior and much more Shakespeare-esque!