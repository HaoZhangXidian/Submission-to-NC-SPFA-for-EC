# Submission-in-NC-SPFA-for-EC

This is the code for our submission in NC: supervised topic modeling for EC design

We run our code based on Python 3.7 and pytorch 0.4 on Windows system.

The descriptions in the folder:

1) main.py: This is the main function.

2) libCrt_Multi_Sample.dll, libCrt_Sample.dll, libMulti_Sample.dll: These are dynamic link functions for sampling in our method. 
The dynamic link functions should be used in Windows system with visual studio as compiler for python.

3) PGBN_sampler.py, train.py, Optimizer.py, build_model.py: These are codes for building the model and optimization, which are used in main.py

Note that, since the data is privacy, we can not upload the data without permission. You can try our method on your data or ask for the permision from 
OneFlorida Clinical Research Consortium (https://www.ctsi.ufl.edu/ctsa-consortium-projects/oneflorida/).
In ths Line 19, 20 in main.py, we explain how to organize the data.

If you have any questions, please contact us:
haz4007@med.cornell.edu
jix4002@med.cornell.edu
few2001@med.cornell.edu

