# neural-feedback

read some papers about alternatives to back propagation and hit:

https://github.com/metataro/DirectFeedbackAlignment

After understanding the idea, we were left with the question, if not anything 
would be as good as a random matrix. Therefore we decided to try different 'cheaper'
alternatives to just distribute the error over the neurons of a layer. Especially 
distributing the error linear (resampling) worked well.


