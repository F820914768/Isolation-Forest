## Principal Component Analysis

Before going into the key concept of Principle component analysis, we need to first address several questions. 
* What is principle component analysis?
* Why are we doing principle component analysis?
* How can be understand principle component intuitively?

Problem often occurs when variables in a datasets have high correlations. A way to address that is to create orthogonal vectors. 
A common way to create orthogonal vectors is QR decomposition and SVD decomposition. However, we don't want to expand our data into any random orthogonal base vectors. If our data points are grouped like a ellipse, we want our new orthogonal base vectors to be the main axis of ellipse.

Now consider a situation where $x_1 = [1,0]$ and $x_2 = [1,0.1]$. If we orthogonized it, it will become
$y_1 = [1,0]$ $y_2 = [0,0.1]$. As we can see, the new base vectors have very different norm. Since $y_2$ doesn't contain much information, we may want to delete it.
