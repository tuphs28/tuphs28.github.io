---
layout: distill
title: Model-Free Planning
description: Interpreting planning in model-free RL
img: assets/img/projects/planning/emgplanning_titlegif.gif
importance: 1
category: Paper-Summaries
related_publications: true
bibliography: interpplanning.bib

toc:
  - name: 0 - TL;DR
  - name: 1 - Introduction
  - name: 2 - Probing For Planning-Relevant Concepts
  - name: 3 - Does The Agent Plan?
  - name: 4 - Intervening on The Agent’s Plans
  - name: 5 - Discussion
  - name: 6 - Conclusion

---
In this post, I summarise my forthcoming paper in which we present evidence that a model-free reinforcement learning agent can learn to internally perform planning.  Our post is organised as follows. In Section 0, I provide a high-level TL;DR. After this, in Section 1, I provide a brief introduction to our work. Sections 2, 3 and 4 then detail the three primary steps of our analysis. Finally, Sections 5 and 6 conclude with some discussion and high-level takeaways.

## 0 - TL;DR
- Within modern AI, “planning” is typically associated with agents that have access to an explicit model of their environment. This naturally raises a question: can an agent learn to plan without such a world model?
- Guez et al. (2019) introduced Deep Repeated ConvLSTM (DRC) agents<d-cite key="guez2019investigation"></d-cite>. Past work has shown that, despite lacking an explicit world model, DRC agents *behave* in a manner that suggests they perform planning <d-cite key="garriga-alonso2024planning"></d-cite><d-cite key="chung2024predicting"></d-cite>. However, it was not previously known why DRC agents exhibit this behavior.
- In our paper, we seek to understand why DRC agents exhibit behavioural evidence of planning. Specifically, we provide evidence that strongly indicates that a Sokoban-playing DRC agent *internally* performs planning. 
- We do this by using linear probes to locate representations of planning-relevant concepts within the agent’s activations. These concepts correspond to predictions made by the agent regarding the impact of its future actions on the environment.
- The agent appears to use its internal representations of these concepts to implement a search-based planning algorithm.
- We demonstrate that these internal plans influence the agent's long-term behaviour in the way that would be expected. For example, we intervene on the agent’s activations to cause it to formulate and execute specific plans.


## 1 - Introduction
In the context of modern deep learning, “decision-time planning” – that is, the capacity of selecting immediate actions to perform by predicting and evaluating the consequences of different actions – is conventionally associated with model-based, AlphaZero-style agents. <d-footnote>
Note that “planning” refers to two different phenomena in RL. Sometimes, “planning” refers to decision-time planning, that is, the capacity of selecting actions by predicting and evaluating the consequences of future actions. However, “planning” in RL can also refer to background planning. Background planning refers to an agent learning a better policy and/or value function by interacting with a world model during training. A classic example of background planning is Dyna.
</d-footnote><d-footnote>In past work, decision-time planning has usually been defined as the process of interacting with an explicit world model to select actions associated with good long-term consequences. However, this definition presupposes that an agent has a world model. Thus, we pragmatically introduce the above characterization of decision-time planning for the purposes of studying planning in model-free agents. Note that this pragmatic characterisation mirrors model-based definitions of planning but relaxes the requirement for an explicit world model to the requirement that an agent predict consequences of future actions,
regardless of the method used. </d-footnote>  These agents predict and evaluate the consequences of different actions by interacting with an explicit model of their environment. <d-footnote> We understand an "explicit world model" as anything introduced for the purpose of approximating the dynamics of an environment. This covers both simulators that agents explicitly interact with to predict the consequences of their actions (e.g. AlphaZero), and inductive biases whereby network topologies are structured to reflect the application of some a planning algorimth to a world model (e.g. MCTSNets<d-cite key="guez2018learning"></d-cite>). </d-footnote> However, this naturally raises a question: can an agent learn to plan without relying on an explicit world model?

We investigate this question in the context of a Deep Repeated ConvLSTM (DRC) agent – a type of generic model-free agent introduced by Guez et al.
(2019)<d-cite key="guez2019investigation"></d-cite> – that is trained to play the game of Sokoban <d-footnote>
Specifically, we focus on DRC agents trained in an actor-critic setting with IMPALA <d-cite key="espeholt2018impala"></d-cite>
</d-footnote>. DRC agents are parameterized by a stack of 
ConvLSTM layers<d-cite key="shi2015convolutional"></d-cite> (i.e. LSTM layers with 3D recurrent states and convolutional connections) that perform 
multiple internal ticks of recurrent computation for each real time step. <d-footnote>At each time step $t$, a DRC agent passes the observed state 
$x_t$ through a convolutional encoder to produce an encoding $i_t \in \mathbb{R}^{H_0 \times W_0 \times G_0}$. 
This is then processed by $D$ ConvLSTM layers. At time $t$, the $d$-th ConvLSTM has a cell state 
$g_t^d \in \mathbb{R}^{H_d \times W_d \times G_d}$. Here, $H_d = W_d = 8$ and $G_d = 32$. Unlike standard recurrent networks, 
which perform a single tick of recurrent computation per time step, DRC agents perform $N$ ticks of recurrent computation per step. 
Here,  $N=3$.</d-footnote> <d-footnote> The DRC architecture actually makes a few minor modifications to the basic ConvLSTM. Specifically, the DRC architecture 
(1) has a bottom-up skip connection that passes the observation encoding as an additional input to all layers;
(2) has a top-down skip connection that passes the final-layer output at the prior tick as an additional input to the bottom layer at each tick;
(3) uses spatial mean- and max-pooling.</d-footnote> The DRC agent we focus on in our work has three ConvLSTM layers that each perform three recurrent ticks for each 
time step in the environment. The computation performed by this agent is illustrated in Figure 1 below.

<p align="center">
  <img src="/assets/img/projects/planning/drc_pic.png" style="width: 80%; height: auto;" >
  <p style="font-size: 0.75em; font-style: italic; text-align: center;">
  Figure 1: llustration of the DRC architecture. For each time step, the architecture encodes the input $x_t$ as a convolutional encoding $i_t$, passes it to a stack of 3 ConvLSTMs which perform three
ticks of recurrent computation. The output of the final ConvLSTM after the final internal tick is then flattened, passed through an MLP and projected to produce policy logits $\pi_t$ and a value estimate $v_t$.
  </p>
</p>

Sokoban is a deterministic, episodic environment in which an agent must navigate around an 8x8 grid to push four boxes onto four targets. The agent can move a box by stepping onto the square it inhabits. If, for example, the agent steps up onto a square containing a box, the box is pushed one square up. Sokoban allows agents to push boxes in such a way that levels become permanently unsolvable. It is hence a hard domain<d-footnote>Sokoban is PSPACE-complete.<d-cite key="Culberson1997SokobanIP"></d-cite></d-footnote> and is a common benchmark environment when studying planning<d-cite key="hamrick2020role"></d-cite> An illustration of an agent playing Sokoban can be seen in Figure 2 below.<d-footnote>We use a version of Sokoban in which the agent observes a symbolic representation of the environment. In this representation, each square of a Sokoban board is represented as a one-hot vector denoting which of 7 possible states that square is in. However, for ease of inspection, we present all figures using pixel representations of Sokoban. </d-footnote>

<p align="center">
  <img src="/assets/img/projects/planning/example_sokoban.gif" style="width: 30%; height: auto;" >
  <p style="font-size: 0.75em; font-style: italic; text-align: center;">
  Figure 2: Illustration of Sokoban
  </p>
</p>

Despite lacking an explicit world model, DRC agents have been shown to behave as though they are performing decision-time planning when playing Sokoban. For example, DRC agents have been shown to:
- rival the performance of model-based agents like MuZero in strategic environments <d-cite key="chung2024predicting"></d-cite> 
- perform better when given additional test-time compute <d-cite key="guez2019investigation"></d-cite>
- perform actions that serve no purpose other than giving the agent extra "thinking" time <d-cite key="garriga-alonso2024planning"></d-cite>.

Yet, it was not previously known why DRC agents behave in this way: is this behaviour merely the result of complex learned heuristics, or do DRC agents truly learn to internally plan?

In our work, we take a concept-based approach to interpreting a Sokoban-playing DRC agent and demonstrate that this agent is indeed internally performing planning. Specifically, we perform three steps of analysis:
1. First, we use linear probes to decode representations of planning-relevant concepts from the agent’s activations (Section 2).
2. Then, we investigate the manner in which these representations emerge at test-time. In doing so, we find qualitative evidence of the agent internally implementing a process that appears to be a form of search-based planning (Section 3).
3. Finally, we confirm that these representations influence the agent’s behaviour in the manner that would be expected if they were used for planning. Specifically, we show that we can intervene on the agent’s representations to steer it to formulate and execute sub-optimal plans (Section 4).

In performing our analysis, we provide the first non-behavioural evidence that it is possible for agents to learn to internally plan without relying on an explicit world model or planning algorithm. 

## 2 - Probing For Planning-Relevant Concepts
How might an agent learn to plan in the absence of an explicit world model? We hypothesised that a natural way for such an agent to learn to plan would be for it to internally represent a collection of planning-relevant concepts. Note that we understand a "concept" to simply be a minimal piece of task-relevant knowledge <d-cite key="schut2023bridging"></d-cite>. By planning-relevant concepts we mean concepts corresponding to potential future actions the agent could take, and concepts relating to the consequences of these future actions on the environment. 

Two aspects of Sokoban are dynamic: the location of the agent, and the locations of the boxes. As such, we hypothesise that planning-capable agent in Sokoban would learn the following two concepts relating to individual squares of the 8x8 Sokoban board<d-footnote>In this paper, we specifically consider `multi-class' concepts, which can formally be defined as mappings from input states (or parts of input states) to some fixed classes.</d-footnote><d-footnote>Both of the concepts we study map each grid square of the agent's observed Sokoban board to the classes $\{\texttt{UP}, \texttt{DOWN}, \texttt{LEFT}, \texttt{RIGHT}, \texttt{NEVER}\}$. The directional classes correspond to the agent's movement directions.  If the next time the agent steps onto a specific square, the agent steps onto that square from the left, the concept $C_A$ would map this square to the class $\texttt{LEFT}$. If the next time the agent pushes a box off of specific square, the box is pushed to the left, the concept $C_B$ would map this square to the class $\texttt{LEFT}$. Finally, the class $\texttt{NEVER}$ corresponds to the agent not stepping onto or pushing a box off of a square again for the remainder of the episode</d-footnote>:

- **Agent Approach Direction ($$C_A$$)**: A concept that captures (i) whether an agent will step onto a square at any point in the future; if the agent will step onto a square this concept also captures (ii) which direction the agent will step onto this square from. 
- **Box Push Direction ($$C_B$$):** A concept that captures (i) whether a box will be pushed off of this square at any point in the future; if a box will be pushed off of a square, this concept also captures (ii) which direction this box will be pushed. 

We use linear probes – that is, linear classifiers trained to predict these concepts using the agent’s internal activations – to determine whether the Sokoban-playing DRC agent we investigate indeed represents these concepts<d-footnote>Linear probes are just standard linear classifiers. So, when predicting  linear probe will compute a logit $l_k= w^T_kg$ for each class $k$ by projecting the associated activations $g \in \mathbb{R}^d$ along a class-specific vector $w_k \in \mathbb{R}^d$.</d-footnote>. Specifically, we hypothesise that the agent will learn a spatial correspondence between its 3D recurrent state and the Sokoban board. As such, we train linear probes that predict the concepts $$C_A$$ and $$C_B$$ for each square $$(x,y)$$ of a Sokoban board using the agent’s cell state activations at position $$(x,y)$$. We call these "1x1 probes". We also trained probes that recieve predict concepts for each square $$(x,y)$$ using a 3x3 patch of the agent’s cell state activations around position $$(x,y)$$. We call this latter type of probes "3x3 probes". Finally, as a baseline we trained 1x1 and 3x3 probes to predict square level concepts when recieving the agent's observation as input. 

We measure the extent to which linear probes can accurately decode these concepts from the agent's cell state using the Macro F1 score they achieve. <d-footnote>The Macro-F1 score is a multi-class generalisation of the F1 score. To calculate it, you calculate F1 scores when viewing each class as the positive class and then take the unweighted mean of these class-specific F1 scores. The F1 score for any binary classification task is the harmonic mean of the precision and recall. </d-footnote> The Macro F1 scores of 1x1 and 3x3 probes are shown below.

<p align="center">
  <img src="/assets/img/projects/planning/proberes.png" style="width: 90%; height: auto;">
  <p style="font-size: 0.75em; font-style: italic; text-align: center;">
    Figure 3: Macro F1s achieved by 1x1 and 3x3 probes when predicting (a) Agent Approach Direction and (b) Box Push Direction using the agent's cell state at each layer, or, for the baseline probes, using the observation. Error bars show ±1 standard deviation. 
  </p>
</p>

Our linear probes are able to accurately predict both (1) agent approach direction and (2) box push direction for squares of Sokoban boards. For instance, when probing the agent’s cell state at all layers for these concepts, all macro F1 scores are greater than 0.8 (for agent approach direction) and 0.86 (for box push direction). In contrast, probes trained to predict agent approach direction and box push direction based on the agent’s observation of the Sokoban board, only get macro F1s of 0.2 and 0.25. We take this as evidence that the agent indeed (linearly) represents the two hypothesised planning-relevant concepts. Similarly, we take the relatively minimal increase in performance when moving from 1x1 to 3x3 probes as evidence that the agent indeed represents these concepts at localised spatial positons of its cell state.

## 3 - Does The Agent Plan?
So, the Deep Repeated ConvLSTM (DRC) agent internally represents the aformentioned planning-relevant, square-level concepts. How, then, does the agent use these representations to engage in planning?

### 3.1 - Internal Plans

We find that the agent uses its internal representations of Agent Approach Direction and Box Push Direction for each individual Sokoban square to formulate coherent planned paths to take around the Sokoban board, and to predict the consequences of taking these paths on the locations of boxes. That is, when we apply 1x1 probes to the agent's cell state to  predict $$C_A$$ and $$C_B$$ for every square of observed Sokoban boards, we find what appear to be (1) paths the agent expects to navigate along and (2) routes the agent expects to push boxes along. Examples (a)-(c) in the below figure provide examples of the agent forming "internal plans" in this way in three example levels.

<p align="center">
  <img src="/assets/img/projects/planning/planning_ex.png" style="width: 100%; height: auto;">
  <p style="font-size: 0.75em; font-style: italic; text-align: center;">
    Figure 4: Examples of internal plans decoded from the agent’s cell state by a probe. Teal and purple arrows respectively indicate that a probe applied to the agent’s cell state decodes that the agent expects to next step on to, or push a box off, a square from the associated direction. 
  </p>
</p>

This, however, raises a question: how does the agent arrive at these internal plans? Is the agent merely remembering past sequences of actions that leads to good outcomes – that is, is it performing some form of heuristic-based lookup - or has the agent learned something more akin to a generalisable planning algorithm? 

### 3.2 - Iterative Plan Refinement

One way to go about answering this question is to look at what happens if we force the agent to pause and "think" prior to acting at the start of episodes. If the agent is merely performing something akin to heuristic-based lookup we would not expect the agent's internal plan to necessarily get any more accurate when given extra "thinking time". In contrast, if the agent were indeed performing some form of iterative planning, this is exactly what we would expect. 

To test this, we forced the agent to remain stationary and not perform any actions for the first 5 steps of 1000 episodes. Since the agent performs 3 internal ticks of computation for each real time step, this corresponds to giving the agent 15 additional computational ticks of "thinking time" before it has to act in these episodes. We then used 1x1 probes to decode the agent's internal plan (in terms of both $$C_A$$ and $$C_B$$) at each tick, and measured the correctness of the agent's plan at each tick by measuring the macro F1 when using that plan to predict the agent's actual future interactions with the environment. Figure 5 below shows that, as would be expected if the agent planned via an iterative search, the agent’s plans iteratively improve over the course of the 15 extra internal ticks of computation it performs when forced to remain still prior to acting. This is all to say that, when given extra time to think, the agent seems to iteratively refine its internal plan despite nothing in its training objective explicitly encouraging this!

<p align="center">
  <img src="/assets/img/projects/planning/planrefinement.png" style="width: 70%; height: auto;">
  <p style="font-size: 0.75em; font-style: italic; text-align: center;">
    Figure 5: Macro F1 when using 1x1 probes to decode $C_A$ and $C_B$ from the agent’s final layer cell state at each of the additional 15 internal ticks performed by the agent when the agent is given 5 ‘thinking steps’, averaged over 1000 episodes.
  </p>
</p>

### 3.3 - Visualising Plan Formation

We can also investigate the question of whether the agent forms plans via some internal planning mechanism by qualitatively inspecting the manner in which the agent's plans form. When visualising how the routes the agent plans to push boxes develop over the course of episodes, we observed a few recurring “plan-formation motifs”. These are described below. Note that, in the below examples, a blue arrow indicates that the agent expects to push a box off of a square in the associated direction. Additionally, note that we visualise the agent's internal plan at each of the 3 computational ticks it performs each time step.

- **Forward Planning** - The agent frequently formulates its internal plan by iteratively extending planned routes forward from boxes. An example can be seen in Figure 6.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_for.gif" style="width: 50%; height: auto;" alt="An example episode">
  <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
    Figure 6: An example of an episode in which the agent iteratively constructs a plan by extending its plan forward from the lower-most box.
  </p>
</div>


- **Backward Planning** - The agent likewise often constructs its internal plan by iteratively extending planned routes backward from boxes. An example can be seen in Figure 7.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_back.gif" style="width: 50%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 7: An example of an episode in which the agent iteratively constructs a plan by extending its plan backward from the lower-right target.
  </p>
</div>

- **Evaluative Planning** - The agent sometimes (1) plans to push a box along a naively-appealing route to a target, (2) appears to evaluate the naively-appealing route and realise that pushing a box along it would make the level unsolvable, and (3) form an alternate, longer route connecting the box and target. An example can be seen in Figure 8.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_eval.gif" style="width: 50%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 8: An example of the agent appearing to evaluate, and subsequently modify, part of its plan. In this episode, the shortest path between the center-right box and center-most target is to push the box rightwards along the corridor. However, it is not feasible for the agent to push the box along this path, since doing so requires the agent to "get under the box" to push it up to the target. The agent cannot do this as pushing the box in this way blocks the corridor. 
  </p>
</div>

- **Adaptive Planning** - the agent often initially plans to push multiple boxes to the same target before modifying its plan to push one of these boxes to an alternate target. An example can be seen in Figure 9.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_adapt.gif" style="width: 50%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 9: An example in which the agent initially plans to push two boxes to the same target (the left-most target) but modifies its plan such that one of the boxes is pushed to an alternate target. Note that, in this episode, the agent appears to realise that the upper-left box must be pushed to the left-most target as it, unlike the lower-left target, cannot be pushed to any other targets.
  </p>
</div>

We think these motifs provide qualitative evidence that the agent forms plans using a planning algorithm that (i) is capable of evaluating plans, and that (ii) utilises a form of bidirectional search. This is interesting as the apparent use of bidirectional search indicates that the agent has learned to plan in a manner that is notably different from the forward planning algorithms commonly used in model-based RL<d-footnote>Most AlphaZero-style agents that engage in model-based planning rely on either MCTS or some other form of forward-rollout based planning</d-footnote>. We conjecture that the agent has learned to rely on bi-directional planning as it allows for efficient plan-formation in an environment such as Sokoban in which there are clear states to plan forward and backward from<d-footnote>Interestingly, one of the more capable RL agents not utilising deep learning also relies on a procedure that is similar to bidirectional planning<d-cite key="Culberson1997SokobanIP"></d-cite></d-footnote>.

### 3.4 - Planning Under Distribution Shift
If the agent does indeed form plans by performing a bidirectional, evaluative search, we would expect the agent to be able to continue to form coherent plans in levels drawn from different distributions to the levels it saw during training. This is because the agent’s learned search procedure would presumably be largely unaffected by distribution shift. In contrast, if the agent solely relied on memorised heuristics, we would expect the agent to be unable to coherently form plans in levels drawn from different distributions, as the memorised heuristics may no longer apply. Interestingly, the agent can indeed form coherent plans in levels drawn from different distributions. We now provide two examples of this.

- **Blind Planning** - The agent can frequently form plans to solve levels in levels in which the agent cannot see its own locations. An example can be seen in Figure 10.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_noag.gif" style="width: 50%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 10: An example of an episode in which the agent iteratively constructs a plan to push boxes to targets despite not being able to see its own location.
    </p>
</div>

- **Generalised Planning** - Despite being trained entirely on boxes with four boxes and four targets, the agent can succesfully form plans in levels with additional boxes and targets. An example can be seen in Figure 11.
<div style="text-align: center;">
  <img src="/assets/img/projects/planning/motifs_gen.gif" style="width: 50%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 11: An example of an episode in which the agent iteratively constructs a plan to push 6 boxes to 6 targets despite never having seen such a level during training.
    </p>
</div>


## 4 - Intervening on The Agent’s Plans
However, the evidence presented thus far only provides evidence that the agent constructs internal plans. It does not confirm that the agent’s apparent internal plans influence its behaviour in the way that would be expected if the agent were truly engaged in planning..

As such, we performed  intervention experiments to determine whether the agent’s representations of Agent Approach Direction and Box Push Direction influence the agent’s behaviour in the way that would be expected if these representations were indeed used for planning. Specifically, we performed interventions in two types of handcrafted levels:
- We designed levels in which the agent could take either a short or a long path to a region containing boxes and target. In these levels, we intervened to encourage the agent to take the long path.
- We designed levels in which the agent could push a box a short or a long route to a target. In these levels, we intervened to encourage the agent to push the box the long route.

Our interventions consisted of adding the vectors learned by the probes to specific positions of the agent’s cell state with the aim of inducing the desired behaviour<d-footnote>Recall that a 1x1 probe 
projects activations along a vector $w_k \in \mathbb{R}^{32}$ to compute a logit for class $k$ of some multi-class concept $C$. We thus encourage the agent to represent square $(x,y)$ as class $k$ for concept $C$ by adding $w_k$ to position $(x,y)$ of the agent's cell state $g_{x,y}$:$$    g'_{x,y} \leftarrow g_{x,y} + w_k$$ If the agent indeed uses $C_A$ and $C_A$ for planning, altering the agent's square-level representations of these concepts ought to modify its internal plan and, subsequently, its long-term behavior.</d-footnote>. Our interventions were successful, with the agent being induced to take the long path in 98.8% of the former types of levels and 80.6% of the latter types of levels. Furthermore, when visualising the agent’s internal plans in levels with and without interventions, we can clearly see that the reason the agent’s behaviour changes after the intervention is because the intervention causes it to formulate an alternate internal plan. This would be expected if the agent indeed used its representations of Agent Approach Direction and Box Push Direction as part of an internal planning process. This is illustrated in Figures 12 and 13 below

<div style="text-align: center;">
  <img src="/assets/img/projects/planning/interv_ag.gif" style="width: 80%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 12: An example of the effect of an intervention on the path that the agent plans to follow. Teal arrows correspond to squares that the agent expects to step onto, and the direction is expects to step onto them from. The left-most gif shows the agent's plan over the initial steps of an episode when no intervention is performed. The middle gif shows the squares intervened upon, with our interventions encouraging the agent to step onto the square with the white arrow, and discouraging the agent from stepping onto the squares with the white crosses. The right-most gif shows the agent's plan over the initial steps of an episode when the intervention is performed.
  </p>
</div>

<div style="text-align: center;">
  <img src="/assets/img/projects/planning/interv_box.gif" style="width: 80%; height: auto;" >
   <p style="font-size: 0.75em; font-style: italic; text-align: center; width: 90%; margin: 0 auto;">
  Figure 13: An example of the effect of an intervention on the route that the agent plans to push boxes. Purple arrows correspond to squares that the agent expects to push boxes off of, and the direction is expects to push them when it does so. The left-most gif shows the agent's plan over the initial steps of an episode when no intervention is performed. The middle gif shows the squares intervened upon, with our interventions encouraging the agent to push the box in the direction of the white arrow, and discouraging the agent from pushing boxes off of the squares with the white crosses. The right-most gif shows the agent's plan over the initial steps of an episode when the intervention is performed.
  </p>
</div>

## 5 - Discussion
What is the takeaway from all of this? Well, in seeming to plan via searching over potential future actions and their environmental impacts, the agent appears to have leaned to a plan in a manner analogous to model-based planning. However, this raises an obvious question: how can a nominally model-free agent have learned to plan in this way? 

We think the answer here is that the inductive bias of ConvLSTMs has encouraged the agent to learn to organise its internal representations in a manner that corresponds to a learned (implicit) environment model. Recall that the agent has learned a spatial correspondence between its cell states and the Sokoban grid such that the agent represents the spatially-localised concepts of Agent Approach Direction and Box Push Direction at the corresponding positions of its cell state. We contend this means that the agent’s 3D recurrent state can be seen as containing a learned “implicit” model of the environment. This is, of course, not a true world model in the sense that the agent is not explicitly approximating the full dynamics of the environment. However, the agent’s recurrent state does seemingly "do enough" to play the role of a world model in that it allows the agent to (i) formulate potential sequences of future actions and (ii) predict their environmental impacts. 


## 6 - Conclusion

Our work provides the first non-behavioural evidence that agents can learn to perform decision-time planning without either an explicit world model or an explicit planning algorithm. Specifically, we provide evidence that appears to indicate that a Sokoban-playing DRC agent internally performs a process with similarities to bi-directional search-based planning. This represents a further blurring of the classic distinction between model-based and model-free RL, and confirms that -- at least with a specific architecture in a specific environment -- model-free agents can learn to perform planning.

However, our work leaves many questions unanswered. For instance, what are the conditions under which an agent can learn to plan without an explicit world model? For instance, as exemplified by its multiple internal recurrent ticks, the agent we study has a slightly atypical architecture. Likewise, the 3D nature of the agent's ConvLSTM cell states means that the agent is especially well-suited to learning to plan in the context of Sokoban's localised, grid-based transition dynamics. While we suspect our findings hold more broadly<d-footnote>In an appendix to our paper we find evidence (1) that DRC agents internally plan in Sokoban even when only performing a single tick of computation per step; (2) that a ResNet agent can learn to plan in Sokoban; (3) that a DRC agent can learn to plan in Mini Pacman, a game with less-localised transition dynamics. </d-footnote>, we leave it to future work to confirm this.

### Acknowledgements

This project would not have been possible without the amazing guidance and mentorship I have recieved from Stephen Chung, Usman Anwar, Adria Garriga-Alonso and David Krueger to who I am deeply grateful. I would also like to thank Alex Cloud for helpful feedback on this post.