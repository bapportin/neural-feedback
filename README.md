# neural-feedback

read some papers about alternatives to back propagation and hit:

https://github.com/metataro/DirectFeedbackAlignment

After understanding the idea, we were left with the question, if not anything 
would be as good as a random matrix. Therefore we decided to try different 'cheaper'
alternatives to just distribute the error over the neurons of a layer. Especially 
distributing the error linear (resampling) worked well.


with further tests we realized that using different activation functions caused problems
because of the reachable value range. A last layer with a linear activation would
case a previous sigmoid or tanh activated layer to be in very quickly out of range. Therefore
we thought about alternative activationh functions. What do we actually require or at least wish
an activation function should be?

1. Non linearity
2. crossing the origin
3. steepest gradiend at the origin
3. derivable
4. degrading gradiend, but never zero
5. limes in both directions infinity

and we found one:

y=np.sing(x)*np.log(np.abs(x)+1)

it is the logarithm shifted to cross the origin and mirrored at the line y=-1*x. Now with this activation function any value domain should be learnable.





