wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz
tar -zxvf Python-3.10.6.tgz
cd Python-3.10.6/
mkdir ~/.localpython
./configure --prefix=/home/baker/inverse-tracr/Python-3.10.6
make
make install