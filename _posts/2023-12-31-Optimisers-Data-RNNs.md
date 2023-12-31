---
title: "BadTorch Part 2: Data, Optimisers and RNNs"
tags:
    - BadTorch
    - Deep Learning
---

Having outlined the purpose of [BadTorch](https://github.com/tuphs28/BadTorch) in the prior post, I will not outline the first parts of BadTorch I completed. Namely, I will explain how I handled data processing and optimisers. I will also outline the construction and character-level language modelling performance of some simple RNNs built with BadTorch,

### Handling Data

To perform character-level language modelling, we need to convert strings of text into a format that can be fed into a model. The simplest way to do this is to associate each unique character with a unique index, and convert strings of characters into tensors of the associated ordered indices. This was done using a simple dictionary mapping, where `text` is just a string containing our whole text dataset (e.g. The Complete Works of William Shakespeare).

```python
vocab = list(set(text))
vocab = sorted(vocab)
vocab_size = len(vocab)
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}
```

We can then create a simple train-val-test split of the data to produce the necessary input and target tensors (where the target for each input element of a sequence is merely the index corresponding to the following character).

```python
seq_length = 128
num_obs = len(text) // seq_length
X, Y = torch.zeros((num_obs, seq_length), dtype=torch.int), torch.zeros((num_obs, seq_length), dtype=torch.long)
for i in range(num_obs):
  chars = text[seq_length*i: seq_length*(i+1)+1]
  xs, ys = chars[:-1], chars[1:]
  xs, ys = [stoi[s] for s in xs], [stoi[s] for s in ys]
  X[i], Y[i] = torch.tensor(xs), torch.tensor(ys)

Xtr, Ytr = X[:(num_obs*80)//100, :], Y[:(num_obs*80)//100, :]
Xval, Yval = X[(num_obs*80)//100:(num_obs*90)//100, :], Y[(num_obs*80)//100:(num_obs*90)//100, :]
Xte, Yte = X[(num_obs*90)//100:, :], Y[(num_obs*90)//100:, :]
```

A necessary step we must complete before training any model is, of course, implementing a principled way to feed in training examples and the associated outputs we wish the model to produce from them. PyTorch does this with its DataLoader objects, which are iterables that allow a user to feed a model batches of (shuffled) training inputs and outputs. This is a relatively simple idea to re-implement, especially if we don't include all of PyTorch's bells and whistles.

I hence chose to implement my own version of the DataLoader class in the simple way shown in the code block below. The Dataset object serves to couple the input tensors and target output tensors for the whole training data together, while the DataLoader class allows us to iterate through batches of these examples in a random order. 

```python
class Dataset:

  def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:

    assert inputs.shape[0] == targets.shape[0], "Different number of inputs and targets"
    self.n = inputs.shape[0]

    self.inputs = inputs
    self.targets = targets


class DataLoader:

  def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
    self.inputs = dataset.inputs
    self.targets = dataset.targets
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.n = dataset.n
    self.iter_count = 0

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.iter_count == 0:
      self.idxs = list(range(self.n))
      if self.shuffle:
        random.shuffle(self.idxs)
    if (self.iter_count+1)*self.batch_size > self.n:
      self.iter_count = 0
      raise StopIteration
    else:
      batch_idxs = self.idxs[self.iter_count*self.batch_size: (self.iter_count+1)*self.batch_size]
      current_inputs = self.inputs[batch_idxs]
      current_targets = self.targets[batch_idxs]
      self.iter_count += 1
      return current_inputs, current_targets

  def __len__(self) -> int:
    return self.n
```

We can then create train, validation and test dataloaders to use in the model development process.

```python
train_data = Dataset(Xtr, Ytr)
train_loader = DataLoader(train_data, 16, True)
val_data = Dataset(Xval, Yval)
val_loader = DataLoader(val_data, 16, True)
test_data = Dataset(Xte, Yte)
test_loader = DataLoader(test_data, 16, True)
```

### A Basic RNN

Having implemented functionality for handling data, the obvious next step was to build a simple language model. The first architecture I built was a basic multi-layer RNN. I implemented this by implementing an RNNCell class that could be called on each element of an input sequence by an RNN module as [is done in PyTorch](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNN). I also implemented an embedding layer and a linear layer. Note that weights are mostly initialised with Xavier initialisation - this is very important for ensuring the model learns when we optimise it with SGD and without normalising layers like BatchNorm or LayerNorm. This is because otherwise gradients can easily explode or vanish.

```python
class Embedding:
    def __init__(self, num_emb: int, emb_dim: int, device: torch.device = torch.device("cpu")) -> None:
        self.embeddings = torch.randn((num_emb, emb_dim)).to(device)

    def __call__(self, idx: int) -> torch.Tensor:
        x = self.embeddings[idx]
        return x

    def parameters(self) -> list:
        return [self.embeddings]


class Linear:

    def __init__(self, in_dim: int, out_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        self.weight = torch.rand((in_dim, out_dim)).to(device) / (in_dim ** 0.5)
        self.bias = torch.zeros((out_dim)).to(device) if bias else None

    def __call__(self, x: torch.tensor) -> torch.tensor:
        h = x @ self.weight
        if self.bias is not None:
            h += self.bias
        return h

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class RNNCell:

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        self.weight_xh = torch.randn((in_dim, hidden_dim)).to(device)
        self.weight_hh = torch.randn((hidden_dim, hidden_dim)).to(device)
        self.bias = torch.zeros((hidden_dim)).to(device) if bias else None

    def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        z_t = x_t @ self.weight_xh + h_t @ self.weight_hh + (self.bias if self.bias is not None else 0)
        h_t = torch.tanh(z_t)
        return h_t

    def parameters(self) -> list:
        return [self.weight_xh, self.weight_hh] + ([] if self.bias is None else [self.bias])


class RNN:

    def __init__(self, in_dim: int, hidden_dim: int, n_layers:int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cells = [RNNCell(in_dim, hidden_dim, device, bias) if i==0 else RNNCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]

    def __call__(self, x: torch.Tensor, h_prev: Optional[None]) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_length = x.shape[:2]

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)

        inputs = x
        h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)
        for n_layer in range(self.n_layers):
            cell = self.cells[n_layer]
            h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim)).to(self.device)
            h_t = h_prev[:, :, n_layer]
            for seq_idx in range(seq_length):
                x_t = inputs[:, seq_idx, :]
                h_t = cell(x_t, h_t)
                h_outputs[:, seq_idx, :] = h_t
            inputs = h_outputs
            h_final[:, :, n_layer] = h_t

        return h_outputs, h_final

    def parameters(self) -> list:
        parameter_list = []
        for cell in self.cells:
            parameter_list += cell.parameters()
        return parameter_list
```

This little snippet does, I think, a fairly good job at illustating the simple API I created for BadTorch: modules can be called on inputs and we can access module parameters with the epynonymous method. Putting this all together, we can build a simple RNN language model from which we can sample text.

```python

class RecurrentModel:

  def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_layers: int, device: torch.device, bias: bool = True) -> None:
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.hidden_dim = hidden_dim
    self.device = device

    self.embedding = Embedding(vocab_size, emb_dim, device)
    self.rnn = RNN(emb_dim, hidden_dim, n_layers, device, bias)
    self.linear = Linear(hidden_dim, vocab_size, device, bias)

    for parameter in self.parameters():
      parameter.requires_grad = True

  def __call__(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    embs = self.embedding(x)
    hs, h_final = self.rnn(embs, h)
    hs = hs.contiguous().view(-1, self.hidden_dim).to(self.device)
    logits = self.linear(hs)

    return logits, h_final

  def parameters(self) -> list:
    parameters = []
    parameters += self.embedding.parameters()
    parameters += self.rnn.parameters()
    parameters += self.linear.parameters()
    return parameters

  def sample(self, sample_length: int = 250) -> str:

    sampled_idxs = []
    current_idx = torch.randint(high=vocab_size, size=(1,1)).to(self.device)
    sampled_idxs.append(current_idx.item())
    hidden = None
    for i in range(sample_length):
      out, hidden = rnn_model(current_idx, hidden)
      with torch.no_grad():
        prob = nn.functional.softmax(out[-1], dim=0).data
      current_idx = torch.multinomial(prob, 1).view(1,1).to(self.device)
      sampled_idxs.append(current_idx.item())

    sampled_text = []
    for idx in sampled_idxs:
      sampled_text.append(itos[idx])
    sampled_text = "".join(sampled_text)
    return sampled_text
```

### Optimisation

The next step was then to implement optimisation algorithms. Given the huge variety of different algorithms, I decided to create an Optimiser parent class to handle some commonalities (e.g. zeroing out gradients, performing clipping) from which individual optimisers could inherit from. The first algorithms I implemented were SGD and Adam as shown below. As is clear, I copy the optimsation API found in PyTorch so that optimisers have a `.zero_grad()` method to zero out gradients after a pass of optimisation and a `.step()` method to perform a step of optimisation.

```python
class Optimiser:

  def __init__(self, parameters: list, lr: float, grad_clip: int = 0) -> None:
    self.parameters = parameters
    self.lr = lr
    self.grad_clip = grad_clip

  def zero_grad(self) -> None:
    for parameter in self.parameters:
      parameter.grad = None

  def clip_gradient(self, x: torch.Tensor) -> None:
    clipped_grad = torch.clip(x.grad, -self.grad_clip, self.grad_clip)
    x.grad = clipped_grad

class SGD(Optimiser):

  def step(self) -> None:
    for parameter in self.parameters:

      if self.grad_clip:
        self.clip_gradient(parameter)

      parameter.data -= self.lr * parameter.grad

class Adam(Optimiser):

  def __init__(self, parameters: list, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, grad_clip: int = 0) -> None:
    super().__init__(parameters, lr, grad_clip)
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps

    self.t = 0

    self.m1s = []
    self.m2s = []
    for parameter in self.parameters:
      self.m1s.append(torch.zeros_like(parameter))
      self.m2s.append(torch.zeros_like(parameter))

  def step(self) -> None:
    self.t += 1
    for parameter_idx, parameter in enumerate(self.parameters):

      if self.grad_clip:
        self.clip_gradient(parameter)

      self.m1s[parameter_idx] = self.beta1 * self.m1s[parameter_idx] + (1-self.beta1) * parameter.grad
      self.m2s[parameter_idx] = self.beta2 * self.m2s[parameter_idx] + (1-self.beta2) * (parameter.grad ** 2)

      m1_hat = self.m1s[parameter_idx] / (1 - self.beta1 ** self.t)
      m2_hat = self.m2s[parameter_idx] / (1 - self.beta2 ** self.t)
      parameter.data -= self.lr * m1_hat / (torch.sqrt(m2_hat) + self.eps)
```

### Training A Simple Language Model

Having implemented all of this, I could finally train a simple model. Given all of the above, this is pretty easy as shown in the case of training a single-layer RNN with a hidden dimensionality of 512 with SGD below. This model was trained on a training set consisting of 80% of the Complete Works of William Shakespeare (a training set of 4358191 characters), and evaluated with a validation and test set consisting of 2 the remaining 10%s. This data is comprised of a vocabulary of 84 unique chars.

```python
torch.manual_seed(seed=42)
emb_dim = 512
hidden_dim = 512
n_layers = 1
lr = 0.001
num_epochs = 20
batch_size = 16
report_interval = 250
t = 0
total_loss = 0

rnn_model = RecurrentModel(vocab_size, emb_dim, hidden_dim, n_layers, device, True)
optimiser = SGD(rnn_model.parameters(), lr)


for epoch in range(1, num_epochs+1):

  for Xb, Yb in train_loader:
    t += 1
    optimiser.zero_grad()
    Xb, Yb = Xb.to(device), Yb.to(device)
    logits, hs = rnn_model(Xb)
    loss = F.cross_entropy(logits, Yb.view(-1))
    total_loss += loss.item()

    if t % report_interval == 0:
      val_loss = calculate_val_loss(rnn_model)
      train_loss = total_loss / report_interval
      total_loss = 0
      print(f"{t=}, {train_loss=}, {val_loss=}\n")
      print(rnn_model.sample(), "\n")

    loss.backward()
    optimiser.step()
```

Now, this model does pretty poorly as would be expected. For one, if we try to train it on sequences of longer than ~32 chars, the model refuses to train as the sequence length leads to gradients vanishing and/or exploding. Additionally, SGD is a pretty poor choice of optimisation algorithm for an RNN even if gradients do not fully explode/vanish, since the momentum and adaptive learning rates are better suited (at least in the case of not conducting thorough hyperparameter initialisation investigations) to recurrent gradients. As such, this model gets stuck on a validation loss of ~1.8 after around 150,000 iterations of gradient descent. For reference, since our vocabulary of chars has 84 unique characters, the loss associated with the model guessing characters at random would be ln(1/84) ~= 4.43 Thus, there is much work to be done to improve this. To give a sense of what this model has learnt, the following is a 250-character sample from the model when prompted with "ROMEO":

>ROMEO. Fibe, From yet sood  
CARTONE  
YNELSTAVDIs castro's druit's sher good farery you deart knownin,  
    Wheluss onked out bloor by ibloon o' thk  
    Fye depiris;  
    Corry in mighss us hold he will not, sare,  
    Hew her dotull you bess licm.
RoIR. If nuth lith to m  
    tent on 'trave  
    Coldow 'twould de'ble; Enter dif,  
    And whidies strithes homarn,  
    Foan qumes gony-dain,  
    Popecle.  Dom. 'Swath thme thee prow  
    Do gives his hupsto the for as you duts, he theer lick estela talk?


Clearly, the above leaves much to be desired. However, the model does seem to have learned the borad structure of Shakespeare's writing, such that, at a glance, it at least "looks" like a Shakespeare play (i.e. short lines, character names in capital letters...). It also seems, sometimes, to produce English words, which is promising.

An obvious next step is to use Adam as our optimisation algorithm. This leads to an improvement in performance, with the model seeming to bottom out at a validation loss of ~1.56. To give a sense of the improvement, the following is is a 250-character sample from the new model when prompted with "ROMEO". This sample has noticably more actual English words.

>ROMEON. Very grief pitile he take a captestance as thy cateling apice  
Enter [Wills, and I am sounds long loy deathing  
    And I know  
! This here down from thy hairs to thy box to ARLEALIND. Grould you, like if this happing'd see the heir, and strang'd again,  
    There's so.  
      BUCKINGHAM. Not love  
          Was sin! Coming. Re-apose thee.  
            MENNER. No to dark you this jewel and suit-not up  
                vey come, that so king in horself, gyly becomings  
                Within a enry, and when angry,  
                    We on,  
                        But soul,

Additionally, Adam allows this model to converge much faster since this no further training seems to occur after about 25,000 iterations of optimisation. However, so long as we use a basic RNN cell as out building block we cannot train our language model on longer sequence, limiting the extent to which it can learn the long-rage dependencies that are intuitively key to human language.

### More Powerful RNNs

There are many extensions to the basic idea of RNN cells that can better learn long-range dependencies across sequences without suffering from exploding gradients. Two of the most popular such extensions are the Gated Recurrent Unit (GRU) and the Long-Short-Term-Memory (LSTM). The ideas behind both extentions are remarkably similar. With GRUs, we add learnable model parameters to control how much, at each time step, to (1) reset the hidden state by and (2) update the hidden state by. In LSTMs, we have a cell state that acts as a memory through-line in addition to a hidden state. We then have learnable weights controlling how much, at each time step, to (1) forget the cell state by, (2) update the cell state by and (3) write from the cell state to the hidden state. By being able to adaptively alter its "long-term memory" based on parameter inputs, both GRUs and LSTMs are able to propogate gradients further back through time, allowing the models to learn longer-range dependencies.

Therefore, I added implementations of both GRU cells and LSTM cells to BadTorch. The GRU cell implementation slotted perfectly into the RNN layer modue, while the LSTM cell implementation required substantial changes to the forward pass of the RNN layer modules to allow the model to keep track of both the cell and hidden states since the other two cell types did not require a cell state. The altered implementations are shown below.

```python
class GRUCell:

  def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
    self.hidden_dim = hidden_dim
    self.weight_xf = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_hf = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_f = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
    self.weight_xo = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_ho = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_o = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
    self.weight_xh = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_hh = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_h = torch.zeros(size=(hidden_dim,)).to(device) if bias else None

  def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
    i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
    h_prop = torch.tanh(x_t @ self.weight_xh + (h_t * i_f) @ self.weight_hh + (self.bias_h if self.bias_h is not None else 0))
    h_t = i_o * h_t + (1 - i_o) * h_prop
    return  h_t

  def parameters(self) -> list:
    weights = [self.weight_xh, self.weight_hh, self.weight_xo, self.weight_ho, self.weight_xf, self.weight_hf]
    biases = [] if self.bias_h is None else [self.bias_h, self.bias_o, self.bias_f]
    return weights + biases

class LSTMCell:

  def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
    self.hidden_dim = hidden_dim
    self.weight_xf = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_hf = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_f = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
    self.weight_xi = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_hi = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_i = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
    self.weight_xo = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_ho = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.weight_xc = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
    self.weight_hc = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
    self.bias_c = torch.zeros(size=(hidden_dim,)).to(device) if bias else None

  def __call__(self, x_t: torch.Tensor, hc_t: torch.Tensor) -> torch.Tensor:
    h_t, c_t = hc_t[:, :, 0], hc_t[:, :, 1]
    hc_new = torch.zeros_like(hc_t)
    i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
    i_i = torch.sigmoid(x_t @ self.weight_xi + h_t @ self.weight_hi + (self.bias_i if self.bias_i is not None else 0))
    i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
    c_prop = torch.tanh(x_t @ self.weight_xc + h_t @ self.weight_hc + (self.bias_c if self.bias_c is not None else 0))
    c_t = i_f * c_t + i_i * c_prop
    h_t = i_o * torch.tanh(c_t)
    hc_new[:, :, 0], hc_new[:, :, 1] = h_t, c_t
    return hc_new

  def parameters(self) -> list:
    weights = [self.weight_xc, self.weight_hc, self.weight_xi, self.weight_hi, self.weight_xo, self.weight_ho, self.weight_xf, self.weight_hf]
    biases = [] if self.bias_c is None else [self.bias_c, self.bias_i, self.bias_o, self.bias_f]
    return weights + biases


class RNN:

  def __init__(self, in_dim: int, hidden_dim: int, n_layers:int, device: torch.device = torch.device("cpu"), bias: bool = True, cell_type: str = "RNN") -> None:
    self.device = device
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    assert cell_type in ["RNN", "GRU", "LSTM"], "Please enter a valid recurrent cell type from: RNN, GRU, LSTM"
    self.cell_type = cell_type
    if self.cell_type == "RNN":
      self.cells = [RNNCell(in_dim, hidden_dim, device, bias) if i==0 else RNNCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
    elif self.cell_type == "GRU":
      self.cells = [GRUCell(in_dim, hidden_dim, device, bias) if i==0 else GRUCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
    elif self.cell_type == "LSTM":
      self.cells = [LSTMCell(in_dim, hidden_dim, device, bias) if i==0 else LSTMCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]

  def __call__(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, seq_length = x.shape[:2]

    if h_prev is None:
      if self.cell_type == "LSTM":
        h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers, 2).to(self.device)
      else:
        h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)

    if self.cell_type == "LSTM":
      h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers, 2).to(self.device)
    else:
      h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)

    inputs = x
    for n_layer in range(self.n_layers):

      cell = self.cells[n_layer]

      if self.cell_type == "LSTM":
        h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim, 2)).to(self.device)
        h_t = h_prev[:, :, n_layer, :]
      else:
        h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim)).to(self.device)
        h_t = h_prev[:, :, n_layer]

      for seq_idx in range(seq_length):
        x_t = inputs[:, seq_idx, :]
        h_t = cell(x_t, h_t)
        if self.cell_type == "LSTM":
          h_outputs[:, seq_idx, :, :] = h_t
        else:
          h_outputs[:, seq_idx, :] = h_t
      if self.cell_type == "LSTM":
        inputs = h_outputs[:, :, :, 0]
      else:
        inputs = h_outputs
      if self.cell_type == "LSTM":
        h_final[:, :, n_layer, :] = h_t
      else:
        h_final[:, :, n_layer] = h_t

    return h_outputs, h_final

  def parameters(self) -> list:
    parameter_list = []
    for cell in self.cells:
      parameter_list += cell.parameters()
    return parameter_list
```

### Training Language Models to Produce Shakespeare

With LSTMs and GRUs implemented, we can finally turn to training language models that can learn longer dependencies. As a first example, I trained a single-layer model consisting of GRU units with hidden dimensionality of 512. Crucially, using GRUs allowed me to train on longer sequences. After training this model for 5000 iterations on 128-character sequences, I achieved a validation loss of ~1.39. When prompted with "ROMEO", this model produced the following sample, which seems an improvement over the prior two:

>ROMEOR. Thrine o'er widow less were mighty revelt,  
    Nor cry 'Thou say'st won.  
        Prick power that I live, whose show every man though his two princes-  
            That doth my very name with eatry that thou seas;  
                I thank not harm to force hard; and will be almost for;  
                    Yet, I know where we shall make myself,  
                        Backlive, to him, have Dead. I will look him.  
                          HOST. Though you bad mear more-forms are dread  
                                 Bothfared last sport. There's a thousand orchery.  
                                     That should have thine,  
                                         This

The results for using an LSTM are remarkably similar (albeit with the LSTM taking slightly longer to train due to its larger parameter count). We can achieve even better results if we use larger models, albeit at the cost of increased training time as all of the above models take <10 minutes to train with Adam. As such, I trained a 3-layer version of the above GRU for about an hour and a half. This model achieved a validation loss of ~1.32. More impressively, however, this model manages to produce text that looks at first glance like Shakespeare. The following is a sample when the model is prompted with "ROMEO":

>ROMEOR. I think I shall give your Grace do. I have nothing,  
  He refuses who writs no prince.  
    BARDOLK. Cousin Sharsh shall do us.  
      BUCKINGHAM. As ever lordship not in this instant,  
          But having as a kingdom. If I see his heart,  
              You shoak me throw no ill in any aunt  
                  Were fight into this transgression to seal  
                      You redeemed the mouth, and nones, of thee!  
                          This were a passage is force in lips.  
                            CARDINAL. To be made Times heal o'er the Qeworn from wenches.  
                              ARCHBISHOP. Out of fait

Now, of course this isn't a coherent, grammatic piece of Shakespearean literature. However, I would argue that it is nonetheless an incredibly Shakespeare-like piece of text for a model that was trained with minimal data for a short period of time. 
