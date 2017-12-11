import urllib, os

if not (os.path.isdir('cifar-10-batches-py') | os.path.isfile('cifar-10-python.tar.gz')):
    print 'no tar.gz or dir found so downloading tar.gz \n'
    os.system('wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
if not os.path.isdir('cifar-10-batches-py'):
    print 'no dir found so extracting tar.gz \n'
    os.system('tar -xvzf cifar-10-python.tar.gz')