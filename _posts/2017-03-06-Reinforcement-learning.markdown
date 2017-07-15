---
layout: post
comments: true
mathjax: true
priority: 680
title: “Reinforcement learning”
excerpt: “Reinforcement learning with deep learning”
date: 2017-03-06 14:00:00
---
### Overview

In autonomous driving, the computer takes actions based on what it sees. It stops on a red light or makes a turn in a T junction. In a chess game, we make moves based on the chess pieces on the board. In reinforcement learning, we study the actions that maximize the total rewards. In stock trading, we evaluate our trading strategy to maximize the rewards which is the total return. Interestingly, rewards may be realized long after an action. Like in a chess game, we may make sacrifice moves to maximize the long term gain. 

>  In reinforcement learning, we create a **policy** to determine what **action** to take in a specific **state** that can maximize the **rewards**.

In blackjack, the state of the game is the sum of your cards and the value of the face up card of the dealer. The actions are stick or hit. The policy is whether you stick or hit based on the value of your cards and the dealer's face up card. The rewards are the total money you win or loss.
 
### Known model

#### Iterative policy evaluation

Credit: (David Silver RL course)
<div class="imgcap">
<img src="/assets/rl/value.png" style="border:none;width:40%">
</div>

We divide a room into 16 grids and name each grid as above. At each time frame, we can move up, dow, left or right one step. However, we penalize each move by 1. i.e. we give a negative reward (-1) for every move. The game finishes when we reach the top left corner (grid 0) or the bottom right corner (grid 15). Obviously, with a negative reward, we are better off at grid 1 than grid 9 since grid 9 is further away from exits than grid 1. Here, we use a **value function $$v(s)$$** to measure how good to be at certain state. Let's evaluate this value function with a random policy. i.e. at every grid, we have an equal chance of going up, down, left or right.

We initialize $$v(s)$$ to be 0.
<div class="imgcap">
<img src="/assets/rl/s1.png" style="border:none;width:25%">
</div>
In our first iteration $$ k=1 $$, we compute a new value function for each state based on the next state after taking an action: We add the reward of each action to the value function of the next state. Then we compute the expected value function for all actions. For example, in grid 1, we can go down, left or right (not up) with a chance of 1/3 each. As defined, any action will have a reward of -1. When we move from grid 1 to grid 2, the reward is -1 and the $$v(s)$$ for the next state $$ grid_2$$ is 0. The sum will be $$ -1 + 0 = -1$$. We repeat the calculation for grid 1 to grid 0 (moving left) and grid 1 to grid 5 (moving down) which are all equal to -1. Then we multiple each value with the corresponding chance (1/3) to find the expected value:

$$ v(grid_1) = 1/3 * (-1+0) + 1/3 * (-1+0) + 1/3 * (-1+0) = -1 $$

<div class="imgcap">
<img src="/assets/rl/s2.png" style="border:none;width:25%">
</div>

In the next iteration 2, the new value function for grid 1 is 

$$v(grid_1) = 1/3 * (-1+0) + 1/3 * (-1-1) + 1/3 * (-1-1) = -1.666 \approx -1.7 $$

and when we keep the iteration, the value function will converge to the final result for the random policy. For example, at grid 9, it will take an average of 20 steps to reach the exit point.

<div class="imgcap">
<img src="/assets/rl/s3.png" style="border:none;width:50%">
</div>

With the value function, we can easily create an optimal policy to exit the room. We just need to follow a path with the next highest value function:
<div class="imgcap">
<img src="/assets/rl/s4.png" style="border:none;width:45%">
</div>

In general, the value function $$ v_{k+1}(s) $$ for state $$s$$ at iteration $$k+1$$ is defined as:

$$
V_{k+1}(s) = \sum_{a \in A}\pi(a|s) ( R^a_s + \gamma \sum_{s' \in S} P^a_{ss'}V_k(s'))
$$

Let us digest the equation further. The total estimated rewards (the right term) for taking action $$a$$ at state $$s$$ is: 

$$
R^a_s + \gamma \sum_{s' \in S} P^a_{ss'}V_k(s')
$$ 

$$V_k(s')$$ is the value function of the next state for taking a specific action. However, an action may transit the system to different states. So we define $$P^a_{ss'} $$ as the probability of transfer from state $$ s $$ to $$ s'$$ after taken action $$a$$. Hence, $$ \sum_{s' \in S} P^a_{ss'}V_k(s') $$ is the expected value function for all possible next states after action $$a$$ at state $$s$$. $$ \gamma $$  $$(0 < \gamma <=1) $$ gives an option to discount future rewards. For rewards further away from present, we may gradually discount it because of its un-certainty or to favor a more stable mathematical solution. For $$\gamma < 1$$, we pay less weight to future rewards than present.  $$R^a_s$$ is the reward given of taking action $$a$$ at state $$s$$. So the equation above is the total estimated rewards for taking action $$a$$ at state $$s$$.

In our example, each action only lead to a single state and hence there is only one output state and $$P^a_{ss'}=1 $$. The new estimated total reward is simplify to:

$$
R^a_s + \gamma V_k(s')
$$

> This simplified equation is the corner stone in understanding and estimating total rewards.

$$\pi$$ is our policy and 
$$\pi(a|s)$$ is the probability for taking action $$ a $$ given the state $$s$$. So the whole equation for $$V_{k+1}(s)$$ is the expected value functions for taken all possible actions for a state.

In our example, with $$\gamma=1$$ (no discount on future reward),  the equation is simplify to:

$$
V_{k+1}(s) = \sum_{a \in A}\pi(a|s) ( R^a_s + V_k(s'))
$$

$$
V_{2}(grid_1) = \sum_{a \in A} \pi(a|s) ( -1 + v_k(s')) = 1/3 * (-1+0) + 1/3 * (-1-1) + 1/3 * (-1-1) \approx -1.7
$$

which $$s'$$ is the next state (grid 5, grid 0, grid 2) for action down, left, right respectively.

#### Policy iteration
In policy iteration, our goal is to find the optimal policy rather then the value function of a particular policy. In last section, we hold the policy constant throughout the calculation. In policy iteration, we do not keep the policy constant. In fact, we update to a new policy in every iteration based on the value function:

Credit: (David Silver RL course)
<div class="imgcap">
<img src="/assets/rl/p2.png" style="border:none;width:40%">
</div>

At each iteration, we start with the value function $$ V_{k} $$ and policy $$ \pi $$. Then we compute the value function $$ V_{k+1} $$ for this iteration. Then based on the new value function, we replace the policy by a new greedy policy based on $$ V_{k+1} $$ 

$$ \pi = greedy(V_{k+1}) $$

At the right, we start with a random policy which updated at every iteration. As shown below, after 2 iterations, we can find the optimal policy to reach the exit the fastest.

<div class="imgcap">
<img src="/assets/rl/p1.png" style="border:none;width:45%">
</div>

### Model free

In the example above, we assume the reward and the state transition is well known. In reality, it is not often true in particular robotics. The chess game model is well known. But some games like the Atari ping-pong, the model is not known precisely by the gamer. In the Go game, the model is well known but the state space is so huge that it is computationally impossible to calculate the value function using the methods above. With a model free example, we need to execute an action in the real environment or a simulator to find out the reward and the next transition state for that action.

#### Monte-Carlo learning
We turn to sampling hoping that after playing enough games, we find the value function empirically. In Monte-Carlo, we start with a particular state and keep continue playing until the end of game.

For example, we have one episode with the following state, action, reward sequence. $$ S_1, A_1, R_2, ... , S_t, A_t, R_{t+1}, ...  $$

At the end of the game, we find the total reward $$ G_{t} $$ (total rewards from $$S_t$$ to the end) by adding rewards from $$ R_{t+1} $$ to the end of the game. We update the value function by adding a small portion of the difference between new estimate $$ G_t$$ and the current value $$ V(S_t) $$ back to itself. This approach creates a running mean from a stream of estimates. With many more episodes, $$V(S_t)$$ will converge to the true value.

$$
V(S_t) = V(S_t) + \alpha(G_{t} - V(S_t))
$$

#### Temporal-difference learning (TD)
Let's say we want to find the average driving time from San Francisco to L.A. We first estimate that the trip will take 6 hours and we have 6 check points suppose to be 1 hour apart. We update the estimation of SF to LA after the first check point.  For example, it takes 1.5 hour to reach the first check point Dublin. So the difference from the estimation is $$ (1.5 + 5) - 6 = 0.5 $$.  We add a small portion of the difference back to the estimation. In the next check point, we update the estimation time from Dublin to LA. 

In temporal-difference (TD), we look ahead 1 step to estimate a new total reward by adding the action reward to the value function of the next state (similar to 1.5+1 above):

$$ V'(S_t) = R_{t+1} + \gamma V(S_{t+1}) $$

We compute the difference of $$ V(S_t) $$ and $$ V'(S_t) $$. Then we add a portion of the difference back to $$ V(S_t) $$.

$$
\delta = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

$$
V(S_t) = V(S_t) + \alpha \delta
$$

For a single episode, $$ S_1, A_1, R_2, ... , S_t, A_t, R_{t+1}, ...., S_{last}, A_{last},$$, we update $$ V(S_1)$$ to $$V(S_{last}) $$ each with a one step look ahead as above. After repeating many episodes, we expect $$V$$ to converge to a good estimate.

#### TD(λ)

In TD, we use a 1 step look ahead: use the reward $$ R_{t+1} $$ to update $$ V_{t} $$. 

$$
G^1_t = R_{t+1} + \gamma V(S_{t+1}) 
$$

$$
V(S_t) = V(S_t) + \alpha ( G^1_t - V(S_t))
$$

We can have a 2 step look ahead: Take 2 actions and use the next 2 rewards to compute the difference.

$$
G^2_t = R_{t+1} + \gamma (R_{t+2} + \gamma V(S_{t+2})) 
$$

$$
V(S_t) = V(S_t) + \alpha ( G^2_t - V(S_t))
$$

In fact, we can has n-step look ahead with n approaches infinity. To update $$V(S_t)$$, we can use the average of all these look ahead.

$$
V(S_t) = V(S_t) + \alpha \frac{1}{N}(\delta_1(S_t) + ... + \delta_N(S_t)))
$$

In TD(λ), however, we don't take the average. The weight drops as k increase in $$ G^k_t$$ (in kth steps look ahead). The weight we use is:

$$
G_t(\lambda) = (1-\lambda) \sum^{\infty}_{n=1} \lambda^{n-1} G^n_t
$$ 

The value function is:

$$
V(S_t) = V(S_t) + \alpha ( G_t(\lambda) - V(S_t))
$$


When $$\lambda=0$$, this is TD and when $$\lambda=1$$, it is Monte-carlo. The following demonstrates how the weight decrease for each kth-step.

Source: David Silver RL class.
<div class="imgcap">
<img src="/assets/rl/td.png" style="border:none;width:65%">
</div>

#### Eligibility traces & backward view TD(λ)

We do not compute TD(λ) directly in practice. Mathematically, TD(λ) is the same as Eligibility traces.

First we compute the eligibility traces:

$$
E_0(s) = 0
$$

$$
E_t(s) = \gamma \lambda E_{t-1}(s) + 1( S_t = s)
$$

When we visit a state $$s$$, we add 1 to  $$E_t(s)$$ which record the current trace level for state $$s$$ at time $$t$$. But at the same time, we decay the trace as time pass. In the diagram below, the vertical bar at the bottom indicates when state $$s$$ is visited. We can see the trace value jumps up by approximately 1 unit. But it the same time, we gradually decay the trace by $$ \gamma \lambda E_{t-1}(s)$$ Hence we see the trace decay when there is not new visit. The intuition is to update the estimate more aggressive for states visited closer to when a reward is granted.

<div class="imgcap">
<img src="/assets/rl/et.png" style="border:none;width:65%">
</div>
Source: David Silver RL course.

 We first compute the differences in the estimates. Then we use the eligibility traces as weight of how to update the previous visited states in this episode. 

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

$$
V(s) = V(s) + \alpha \delta_t E_t(s)
$$


#### Model free control

With a model, we optimize policy by taking an action using the greedy algorithm. i.e. in each iteration, we look at the neighboring states and determines which path gives us the best total rewards. With a model, we can derive the action that take us to that state. Without a model, $$V(s)$$ is not enough to tell which action to take to get to the next best state. Now instead of estimate value function for a state $$v(s)$$, we estimate the action value function for a state and an action $$Q(s, a)$$. This value tells us how good to take action $$a$$ at state $$s$$. This value will help us to pick which action to execute during the sampling.

#### Monte-Carlo

The corresponding equation to update $$Q$$ with Monte-Carlo is similar to that for $$ V(s) $$

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha(G_{t} - Q(S_t, A_t))
$$

which $$G_t$$ is the total rewards after taking the action $$A_t$$

However, we do not use greedy algorithm to select action within each episode. Similar to all sampling algorithm, we do not want outliner samples to stop us from exploring the sample action. A greedy algorithm may pre-mature stop us for actions that look bad in a few sample but in general performs very well. But when we process more and more episodes, we will have more exploitation for known good actions rather than exploring more less promising actions to avoid outliner samples.

To sample the kth episode, $$ S_1, A_1, R_2, ... , S_t, A_t, R_{t+1}, ...., S_{last}, A_{last},$$, we use a ε-greedy algorithm below to pick which action to sample next. For example, at $$S_1$$, we use it to compute the policy $$ \pi(a \vert s_1) $$. Then we sample an action as $$ A_1$$ based on this distribution.

$$
ε = 1/k
$$

$$
\begin{equation} 
\pi(a \vert s)=
    \begin{cases}
      ε/m + 1 - ε, & \text{if}\ a^{*} = argmax_a Q(s,a) \\
      ε/m, & \text{otherwise}
    \end{cases}
\end{equation}
$$

As $$k$$ increases, the policy distribution will favor action with higher $$ Q $$ value.

Eventually, similar to the previous policy iteration,
$$
argmax_a Q(s,a)
$$ will be the optimal policy.

<div class="imgcap">
<img src="/assets/rl/e2.png" style="border:none;width:35%">
</div>

#### Sarsa

Sarsa is the corresponding TD method for estimate $$Q$$ without a model. At state S, we sample an action based on $$ Q(s, a)$$ using ε-greedy. We received an reward and transit to $$s'$$. Once again, we sample another action $$ a' $$ using ε-greedy and use the $$ Q(s', a') $$ to update $$ Q(s, a)$$.
<div class="imgcap">
<img src="/assets/rl/sa.png" style="border:none;width:15%">
</div>

$$
Q(S, A) = Q(S, A) + \alpha (R + \gamma Q(S', A') - Q(S, A)) 
$$

Here is the algorithm
<div class="imgcap">
<img src="/assets/rl/a1.png" style="border:none;width:75%">
</div>
Source: David Silver RL course

#### Sarsa(λ) 

> Because its close similarity with TD(λ), we will just show the equations.

To update the $$Q$$ value:

$$
Q_t(\lambda) = (1-\lambda) \sum^{\infty}_{n=1} \lambda^{n-1} Q^n_t
$$ 

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha ( Q_t(\lambda) - Q(S_t, A_t))
$$

#### Sarsa with eligibility traces

Equation for the eligibility trace:

$$
E_0(s, a) = 0
$$

$$
E_t(s, a) = \gamma \lambda E_{t-1}(s, a) + 1( S_t = s, A_t=a)
$$

Update the $$Q$$:

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A{t+1}) - Q(S_t, A_t)
$$

$$
Q(s, a) = Q(s, a) + \alpha \delta_t E_t(s, a)
$$

Here is the algorithm
<div class="imgcap">
<img src="/assets/rl/a2.png" style="border:none;width:75%">
</div>

Source: David Silver RL course


#### Q-learning (Sarsa Max)

In Sarsa, we use

$$
Q(S, A) = Q(S, A) + \alpha (R + \gamma Q(S', A') - Q(S, A)) 
$$

Both $$A$$ and $$A'$$ are sampled using ε-greedy algorithm. However, in Q-learning, $$a$$ is picked by ε-greedy algorithm but $$a'$$ is picked by

$$
argmax_{a'} Q(s', a')
$$

i.e. select the second action with the highest action value function $$ Q(s', a') $$
 
<div class="imgcap">
<img src="/assets/rl/sa4.png" style="border:none;width:20%">
</div>
Credit: David Silver RL course
 
The formular will therefore change to:
 
$$
Q(S, A) = Q(S, A) + \alpha (R + \gamma argmax_{a'} Q(s', a') - Q(S, A)) 
$$
 
The ε-greedy algorithm explores states with high un-certainty in earlier episode. But it also lead the training to very bad state more frequently. Q-learning using greedy algorithm in their second pick. With the combination of ε-greedy algorithm, at least in the empirical data, it helps the training to avoid some very disaster states. It therefore trains the model better.

Here is the algorithm
<div class="imgcap">
<img src="/assets/rl/a3.png" style="border:none;width:75%">
</div>
Credit: David Silver RL course

### Value function estimator
In the previous sections, our focus is to estamate $$\text{V or Q}$$ accurately. If the state of the system is huge, this is not practical. Deep learning has been frequently used to estimate function. So we will use a deep network to estimate the function value instead.

So we need to build a deep network with parameters $$w$$ to estimate $$\text{V or Q}$$. We define a cost function to compute the mean square error between the estimate and the calculated value from the samples in the episodes. We then backpropagate the gradient to change $$w$$ to create a better model to match the sampling episodes.

The cost function is defined as:

$$
J(w) = E_\pi \left[ (v_\pi(s)-\hat{v}(S,w))^2 \right]
$$

which $$\hat{v}(S,w)$$ is the estimate from the deep network and $$v_\pi(s)$$ can be the estimate compute by sampling using Monte-Carlo, TD, TD(λ) or eligibility traces.

 $$
 J(w) = E_\pi \left[ (v_\pi(s)-\hat{v}(S,w))^2 \right]
 $$
 
 $$
 \bigtriangleup w = -\frac{1}{2} \alpha \bigtriangledown_w J(w)
 $$
 
 $$
 \bigtriangleup w = \alpha (v_\pi(s)-\hat{v}(S,w)) \bigtriangledown_w \hat{v}(S,w)
 $$
 
So we can use backpropagate to compute the gradient $$ \bigtriangledown_w \hat{v}(S,w) $$ and change $$w$$.
 
We can using different method to compute $$v_\pi(s)$$

Using Monte-Carlo

 $$
 \bigtriangleup w = \alpha (G_t-\hat{v}(S_t,w)) \bigtriangledown_w \hat{v}(S_t,w)
 $$

Use TD(0)

$$
 \bigtriangleup w = \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1},w) -\hat{v}(S_t,w)) \bigtriangledown_w \hat{v}(S_t,w)
$$

Use TD(λ)

$$
 \bigtriangleup w = \alpha (G_t(λ)-\hat{v}(S_t,w)) \bigtriangledown_w \hat{v}(S_t,w)
$$

For control using Monte-Carlo:

 $$
 \bigtriangleup w = \alpha (G_t-\hat{q}(S_t,A_t, w)) \bigtriangledown_w \hat{q}(S_t,A_t,w)
 $$

Because of the similarity, we will not list the equations for eligibility tracing, Sarsa and Q-learning. Those should look extremely similar with previous equations.

> We can updates $$w$$ immediately at each step of an episode. However, to have a more stable solution, we may store the new $$w$$ values separately and update it at once at the end of an episode.

### Policy descent

In previous sections, the algorithms focus on finding the value functions $$ \text{V or Q}$$ and derive the optimal policy from it. Some algorithms recompute the value functions of the whole state space at each iteration. Unfortunately, for a problem with a huge state space, this is not efficient. Often a lot of state space is not interesting to us. The model free control uses ε-greedy algorithm to reduce the amount of state space to search. We focus more on the action function value $$Q$$ that looks more promising to us. i.e action that either we have not explore much or actions that have high $$Q$$ value.

Policy descent on the contrary focuses on finding the optimal policy $$ \pi_\theta(a \vert s)$$. We are trying to use deep network to predict the best action for a specific state. The principles are pretty simple. If certain state and action can produce good result, we make sure our deep network predict the same action. If the following shooting sequence scores a 3-point, we make sure the deep network will suggest the same action at each frame of the picture. Each frame represents a state and the action is how to move the joints and the muscle.

<div class="imgcap">
<img src="/assets/rl/curry.jpg" style="border:none;width:75%">
</div>
Source ESPN

Technically, we adjust the deep network to make policy that yields better total rewards. Through Monte-Carlo or TD(λ), we can estimate how good to take a specific action $$a_i$$ at a particular state $$s$$. If it is good, we backpropagate the gradient to make changes to $$\theta$$ such that the score predicted by the deep network $$ \pi_\theta(a_i \vert s) $$ is high.

#### Total rewards

Let's calculate the total rewards $$J(\theta)$$ of a single step system. i.e. we sample a state $$s$$ from a state probability distribution $$d(s)$$
from this system, take an action $$a$$, get a reward $$r=R_{s,a}$$ and then immediately terminate. The total rewards will be:

$$
J(\theta) = E_{\pi_\theta} (r)
$$

$$
J(\theta) = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta(s, a) R_{s, a}
$$

$$
\bigtriangledown_\theta J(\theta) = \sum_{s \in S} d(s) \sum_{a \in A} \bigtriangledown_\theta \pi_\theta(s, a) R_{s, a}
$$

$$
\bigtriangledown_\theta J(\theta) = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta(s, a) \frac{\bigtriangledown_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} R_{s, a}
$$

$$
\bigtriangledown_\theta J(\theta) = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta(s, a) (\bigtriangledown_\theta \log \pi_\theta(s, a) ) R_{s, a}
$$

$$
\bigtriangledown_\theta J(\theta) = E_{\pi_\theta} ((\bigtriangledown_\theta \log \pi_\theta(s, a) ) r)
$$

$$
\bigtriangledown_\theta J(\theta) = E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) r)
$$

The term $$\log \pi_\theta(s, a)$$ should be familiar in deep learning. This is the log of a probability. i.e. the logit value in a deep network.  Usually, that is the score before we pass it to a softmax. Our objective is to maximize $$ J (\theta)$$. i.e. build a deep network making accurate predictions and those predictions make the highest total rewards. That is why $$\bigtriangledown_\theta J(\theta)$$ is depends on $$ R_{s, a} $$ and $$score_a $$ which the first one measure how good is the action and the second on how good the deep network predict that action. When $$ R_{s, a} $$ is high, we backpropagate the signal to change $$\theta$$ in the deep network to make $$score_a $$ higher. Eventually, the deep network will make accurate predictions on actions that make high total rewards.

Without proofing, the equivalent equations for multiple steps system is:

$$
\bigtriangledown_\theta J(\theta) = E_{\pi_\theta} ((\bigtriangledown_\theta \log \pi_\theta(s, a) ) Q^{\pi_\theta} (s, a))
$$

$$
\bigtriangledown_\theta J(\theta) = E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) Q^{\pi_\theta} (s, a))
$$

#### Policy Gradient using Monte-Carlo

We can sample $$  Q^{\pi_\theta} (s, a) $$ using Monte-Carlo. The algorithm is:

<div class="imgcap">
<img src="/assets/rl/mc.png" style="border:none;width:50%">
</div>
Source David Silver Course

#### Advantage function

$$ Q^{\pi_\theta} (s, a) $$ is high variance with Monte-Carlo. One value calculated from one sampling path in Monte-Carlo can have very different value in another sampling path. For example, just make a change in one move in chess can produce total different result. We try to reduce the variance by subtracting it with a baseline value $$ V(s) $$. We can proof that it produces the same solution for $$ \bigtriangledown_\theta J(\theta)$$ even we replace $$ Q^{\pi_\theta} (s, a) $$  with $$ Q^{\pi_\theta} (s, a) - V^{\pi_\theta}(s)$$ 

Let's proof that the following term is equal to 0 first:

$$
 E_{\pi_\theta} ((\bigtriangledown_\theta \log \pi_\theta(s, a)) B(S)) = \sum_{s \in S} d(s) \sum_{a \in A} \bigtriangledown_\theta  \pi_\theta(s, a) B(s)
 $$

$$
 = \sum_{s \in S} d(s) B(s)  \bigtriangledown_\theta \sum_{a \in A} \pi_\theta(s, a) 
 $$

$$
= \sum_{s \in S} d(s) B(s)  \bigtriangledown_\theta 1
$$

$$
=0
$$

So we proof that instead of using $$ Q^{\pi_\theta} (s, a) $$, we can use $$ Q^{\pi_\theta} (s, a) - V^{\pi_\theta}(s)$$  to optimize $$ \theta $$

$$
E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) (Q^{\pi_\theta} (s, a) - V{\pi_\theta}(s)))
$$

$$
= E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) (Q^{\pi_\theta} (s, a))) - E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) V{\pi_\theta}(s))
$$

$$
= E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) (Q^{\pi_\theta} (s, a))) 
$$

We call this the advantage function:

$$ 
A^{\pi_\theta} (s, a) = Q^{\pi_\theta} (s, a) - V^{\pi_\theta}(s)$$ 

We use it to replace $$ Q^{\pi_\theta} (s, a) $$ in training the deep network.

Here, we can proof that we can use TD(λ) method as the advantage function.

$$
\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta} (s') - V^{\pi_\theta} (s)
$$

$$
E_{\pi_\theta}(\delta^{\pi_\theta} \vert s, a ) = E_{\pi_\theta} ( r + \gamma V^{\pi_\theta} (s') ) - V^{\pi_\theta} (s)
$$

$$
E_{\pi_\theta}(\delta^{\pi_\theta} \vert s, a ) = Q^{\pi_\theta} (s, a) - V^{\pi_\theta} (s, a)
$$

$$
E_{\pi_\theta}(\delta^{\pi_\theta} \vert s, a ) = A^{\pi_\theta} (s, a)
$$

Here are the different methods to train $$\theta$$ of the deep network using Monte-Carlo, TD, TD(λ) and eligibility trace.
<div class="imgcap">
<img src="/assets/rl/m1.png" style="border:none;width:40%">
</div>
Source David Silver Course

#### Policy gradient using Actor and Critic

Policy gradient needs to know how good is an action with a specific state. $$ Q^{\pi_\theta} (s, a) $$

$$
\bigtriangledown_\theta J(\theta) = E_{\pi_\theta} ((\bigtriangledown_\theta score_a ) Q^{\pi_\theta} (s, a))
$$

In previous section, we use Monte-Carlo, TD, etc... to sample the value. However, we actually can have a second deep network to estimate this value. So we have one actor deep network to predict the policy and an critic deep network to predict the action value function. So critics learns how to evaluate how good is $$ Q(s, a) $$ and the actor deep network learns how to make good policy prediction $$ \pi_\theta(a \vert s)$$.

This is the algorithm:
<div class="imgcap">
<img src="/assets/rl/m3.png" style="border:none;width:40%">
</div>

### Learning and planning

In previous sections, we sample states and take actions to be played in the real world or a simulator to collect information on rewards and transition states. (learning) Even we start with a modeless problem, we can use the observed data to peek into the model to have our algorithms to work better (planning).

Consider a system with state A and B. We use Monte-Carlo to build experience with the system. The left side is the state we sampled with the corresponding rewards. For example, we start with state A, the system gives us 0 reward when we transition to B. We transit to a terminate state and again with 0 reward. We play 7 more episodes which all starts at state B. Using Monte-Carlo, $$v(A)=0$$ and $$v(B)=0.75$$.

<div class="imgcap">
<img src="/assets/rl/dy.png" style="border:none;width:60%">
</div>

From the experience, we can build a model like the middle. Based on this model, we can sample data as in the right hand side. With the sampled experience, $$v(A)=1$$ and $$v(B)=0.75$$. The new $$v(A)=1$$ is closer to the real $$ v(a) $$ than one with just the real experience. So this kind of experience sampling and replay usually gives us better estimation of the value function.

#### Dyna

Here is the Dyna algorithm than use learning and planning to have a better estimate for $$ Q $$.

<div class="imgcap">
<img src="/assets/rl/a4.png" style="border:none;width:60%">
</div>





