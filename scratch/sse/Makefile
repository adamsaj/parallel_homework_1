# On Trestles we will check versus your performance versus Intel MKL library's BLAS. 

CC = cc 
OPT = -O3
CFLAGS = -Wall -std=gnu99 -funroll-loops $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm


targets = benchmark-blocked
objects = benchmark.o dgemm-blocked.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
