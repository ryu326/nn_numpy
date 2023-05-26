## Implementation of Layers
First, we define a class called `Module` which will serve as the base class for definining any other Neural Network modules or layers further on. This, in general, is a good practice since we can define all the necessary functions which are common to all modules that are defined or will be defined. This allows users to define their own modules without re-defining all the common necessary functions. However, in our case, the class `Module` is a dummy class which basically passes the input (in the forward pass) and the gradients (in the backward pass) ahead without modifying them. This is because we re-define all the three basic functions `__init__`, `forward` and `backward` in all the layer classes later.
### Fully Connected Layer
In the `forward` pass, the Fully Connected or the `Linear` layer class applies a linear transformation to the input $\mathbf{x}$ using a weight matrix $\mathbf{W}$ and a bias vector $\mathbf{b}$ as follows:

$$ \mathbf{z} = \mathbf{x}\mathbf{W}^T + \mathbf{b} $$

where $\mathbf{x}$ is a row vector of size $F_{in}$, $\mathbf{z}$ is a row vector of size $F_{out}$, $\mathbf{W}$ is a matrix of size $(F_{out}, F_{in})$ and $\mathbf{b}$ is a row vector of size $F_{out}$. The weight matrix and the bias vector are initialized similarly as in the PyTorch framework.

**Note**: In mini-batch gradient descent, the size of the input would be $(N, F_{in})$ and the size of the output will be $(N, F_{out})$ with $N$ being the size of the mini-batch. However, for the purpose of illustration we use single samples.

<br><br>

In the `backward` pass, the `Linear` layer needs to calculate the output derivatives with respect to the weight matrix $\mathbf{W}$, the bias vector $\mathbf{b}$ and the input $\mathbf{x}$. These output derivatives are multiplied to the loss gradients received from the further layers to get the loss gradients with respect to the layer parameters. The layer stores the loss gradients with respect to the weight matrix and the bias vector and uses them later to update the parameters. The loss gradients with respect to the input are passed on to the previous layers. 

The output derivatives described above are as follows:

$$
\begin{align*}
\frac{\partial \mathbf{z}}{\partial \mathbf{W}} &= \mathbf{x}^T \\
\frac{\partial \mathbf{z}}{\partial \mathbf{b}} &= \mathbf{I} \\
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} &= \mathbf{W} \\
\end{align*}
$$

where $\mathbf{I}$ is an Identity matrix of size $(F_{out}, F_{out})$.

Since the value of the input is required during the backward pass, the layer stores the input in a `cache` variable during the forward pass which is then used during the backward pass.

**Note**: In mini-batch gradient descent, the loss gradients are computed and passed on for each sample in the mini-batch. The loss gradients with respect to the layer parameters are averaged over all the samples in the mini-batch and this average loss gradient value is stored for updating the parameters later.

<br><br>

Once the loss gradients are computed, the layer parameters are updated in the `step` function according to the Gradient Descent update rule:

$$ \Theta = \Theta -  \alpha \frac{\partial L}{\partial \Theta}$$

where $\Theta = \{\mathbf{W}, \mathbf{b}\}$, $\alpha$ is the learning rate and $L$ is the loss. 

<br><br>

Also, the `num_params` function gives the total number of trainable parameters of the `Linear` layer. It is the number of elements in the weight matrix $\mathbf{W}$ i.e. $F_{out} F_{in}$ added to the number of elements in the bias vector $\mathbf{b}$ i.e. $F_{out}$. 

In our case, only the `Linear` layer has trainable parameters and therefore the `step` and the `num_param` functions are defined only for the `Linear` layer.
### ReLU and Softmax Activation Functions
In the `forward` pass, the `ReLU` layer applies the ReLU operation elementwise on the input $\mathbf{z}$. The ReLU operation is defined as:

$$ \mathbf{a} = \text{ReLU}(\mathbf{z}) = \max(0,\mathbf{z}) $$

where $\mathbf{z}$ and $\mathbf{a}$ are row vectors of size $F$. The ReLU operation clips all the negative values to $0$ while keeping the positive values unmodified. 

<br><br>

In the `backward` pass, the `ReLU` layer calculates the output derivative with respect to the input $\mathbf{z}$ as follows:  

$$
\begin{align*}
\frac{\partial \mathbf{a}}{\partial \mathbf{z}} &= \text{diag}\left(\frac{\partial a_0}{\partial z_0}, \dots, \frac{\partial a_i}{\partial z_i}, \dots, \frac{\partial a_{F-1}}{\partial z_{F-1}} \right)\\
\frac{\partial a_i}{\partial z_i} &= 
\begin{cases} 
  0 & \max(0,z_i) = 0 \equiv z_i \leq 0\\
  1 & \max(0,z_i) = z_i \equiv z_i > 0
\end{cases}
\end{align*}
$$

Since we only need to know whether the ReLU was active ($z_i > 0$) or inactive ($z_i \leq 0$) at a position $i$ for computing the output derivatives in the backward pass, we store only a boolean map of size $F$ denoting whether the ReLU was active at that position in the `cache` variable.

<br><br>

The `ReLU` layer has no trainable parameters. 

<br><br>

In the `forward` pass, the `Softmax` layer applies the Softmax operation on the input $\mathbf{z}$. The Softmax operation is used to convert the input $\mathbf{z}$ to a probability distribution so that the elements in $\mathbf{z}$ sum to $1$ and each element lies between $0$ and $1$. The Softmax operation is defined as:

$$ 
\begin{align*}
\mathbf{a} &= \text{Softmax}(\mathbf{z})\\
a_i &= \frac{e^{z_i}}{\sum_{j=0}^{F-1} e^{z_j}} \quad \forall i \in \{0, \dots, F-1\}
\end{align*}
$$

where $\mathbf{z}$ and $\mathbf{a}$ are row vectors of size $F$. 

If the values of elements of $\mathbf{z}$ are high, the exponent values may cross the `float` datatype limits and thus we would get `nan` in the softmax output. Thus for numerical stability, we modify the softmax operation along with a correction factor $D$ as follows:

$$ a_i = \frac{e^{z_i - D}}{\sum_{j=0}^{F-1} e^{z_j - D}} \quad \forall i \in \{0, \dots, F-1\} $$

The correction factor $D$ is generally taken to be $\max(z_0, \dots, z_{F-1})$. Using the correction factor, we avoid `nan` in output while not changing the actual Softmax output values.

<br><br>

In the `backward` pass, the `Softmax` layer calculates the output derivative with respect to the input $\mathbf{z}$ as follows:  


\begin{align*}
\left( \frac{\partial \mathbf{a}}{\partial \mathbf{z}} \right)_{ij} &= \frac{\partial a_i}{\partial z_j} \quad \forall i,j \in \{ 0,\dots,F-1 \}\\
\frac{\partial a_i}{\partial z_j} &= a_i(\delta_{ij} - a_j)
\end{align*}


where $\delta_{ij}$ is the Kroneker Delta defined as:

$$ 
\delta_{ij} = 
\begin{cases} 
  0 & i \neq j\\
  1 & i = j
\end{cases}
$$

Since we need the output values for the output derivative computation in backward pass, we store the outputs computed in the forward pass in the `cache` variable which is then used during the backward pass.

<br><br>

The `Softmax` layer has no trainable parameters.
