\section{Mathematical Formulation}

Let the training dataset be defined as:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
$$

where \( x_i \) and \( y_i \) represent the input and output sequences, respectively. The difficulty measure \( d_i \) for each data point is given by the sum of the lengths of the input and output sequences:

$$
d_i = |x_i| + |y_i|
$$

Here, \( |x_i| \) and \( |y_i| \) denote the lengths of the sequences \( x_i \) and \( y_i \).

The dataset is sorted in ascending order based on the difficulty measure:

$$
d_1 \leq d_2 \leq \dots \leq d_N
$$

Next, the sorted dataset is divided into \( K \) curriculum stages. Each stage \( \mathcal{D}_k \) consists of a subset of the dataset, defined as:

$$
\mathcal{D}_k = \{(x_i, y_i) \mid i = 1, 2, \dots, k \cdot \frac{N}{K}\}
$$

where \( k = 1, 2, \dots, K \).

The model \( \theta \) is trained iteratively, starting with simpler examples in early stages, and progressively including more complex examples. The training process for each stage is expressed as:

$$
\theta^{(k)} = \text{Train}(\theta^{(k-1)}, \mathcal{D}_k)
$$

where \textit{Train} represents the training procedure, such as gradient descent, on the \( k \)-th curriculum stage.
